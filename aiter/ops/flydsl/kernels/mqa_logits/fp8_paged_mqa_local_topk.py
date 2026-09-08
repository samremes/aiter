# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental paged FP8 MQA scorer with an exact LDS local-TopK reservoir."""

# FlyDSL argument annotations must remain concrete at trace time.
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..candidate_topk_common import (
    BLOCK_THREADS,
    NUM_HIST_BINS,
    NUM_RADIX_PASSES,
    NUM_WAVES,
    RADIX_SIGN_BIT,
    block_exclusive_prefix_i32,
    f32_to_ordered_i32,
    make_streaming_topk_storage,
    prefix_radix_mask,
    radix_bucket,
    radix_pass_bits,
)
from ..kernels_common import atomic_add_i32
from ..tensor_shim import GTensor, _run_compiled, _to_raw

HEADS = 32
HEAD_DIM = 128
BLOCK_N = 128
WAVE_SIZE = 64
WAVES = BLOCK_THREADS // WAVE_SIZE
MFMA_M = 16
MFMA_N = 16
MFMA_K = 128
M_TILES = HEADS // MFMA_M
N_TILES_PER_WAVE = (BLOCK_N // MFMA_N) // WAVES
DREG = 4
INCOMING_CAPACITY = 3328
TILES_PER_BATCH = INCOMING_CAPACITY // BLOCK_N
SUPPORTED_K = (128, 512, 1024, 2048)

_RETAINED = 0
_SCORE_PREFIX = 1
_SCORE_MASK = 2
_REMAINING = 3
_STATE_SIZE = 4
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1
_NEUTRAL_E8M0 = 0x7F7F7F7F


def _udiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _umod(a, b):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(b))


def _imin(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return (a <= b).select(a, b)


def _concat_i32x4(lo, hi):
    lo, hi = fx.Vector(lo), fx.Vector(hi)
    return fx.Vector.from_elements(
        [lo[i].ir_value() for i in range_constexpr(4)]
        + [hi[i].ir_value() for i in range_constexpr(4)],
        fx.Int32,
    )


def _load_fp8x32(i32_view, byte_base, lane_div_16):
    """Load one lane's contiguous K32 fragment for 16x16x128 scaled MFMA."""
    off = (byte_base + lane_div_16 * 32) // fx.Int32(4)
    lo = i32_view.vec_load((off,), vec_size=4)
    hi = i32_view.vec_load((off + 4,), vec_size=4)
    return _concat_i32x4(lo, hi)


def _load_preshuffled_fp8x32(
    i32_view,
    page_base,
    token_in_page,
    lane_div_16,
):
    """Load the two 16-byte MFMA fragments from shuffle_weight layout."""
    token_block = token_in_page // fx.Int32(16)
    token_lane = token_in_page % fx.Int32(16)
    dim_block = lane_div_16 * fx.Int32(2)
    byte_offset = (
        page_base
        + token_block * fx.Int32(16 * HEAD_DIM)
        + dim_block * fx.Int32(16 * 16)
        + token_lane * fx.Int32(16)
    )
    lo = i32_view.vec_load((byte_offset // fx.Int32(4),), vec_size=4)
    hi = i32_view.vec_load(
        ((byte_offset + fx.Int32(16 * 16)) // fx.Int32(4),),
        vec_size=4,
    )
    return _concat_i32x4(lo, hi)


@lru_cache(maxsize=32)
def compile_fp8_paged_mqa_local_topk(
    *,
    topk: int,
    arch: str,
    preshuffled: bool,
    page_size: int,
):
    """Compile the H32D128 one-row/one-split Stage-A specialization."""
    if topk not in SUPPORTED_K:
        raise ValueError(f"topk must be one of {SUPPORTED_K}, got {topk}")
    if arch != "gfx950":
        raise RuntimeError(
            "the initial native-e4m3fn Stage-A specialization supports gfx950 only"
        )

    pool_capacity = topk + INCOMING_CAPACITY
    pool_steps = (pool_capacity + BLOCK_THREADS - 1) // BLOCK_THREADS
    output_steps = (topk + BLOCK_THREADS - 1) // BLOCK_THREADS
    storage_type = make_streaming_topk_storage(
        pool_capacity,
        _STATE_SIZE,
        fx.Uint16,
    )
    layout_name = "preshuffled" if preshuffled else "rowmajor"
    kernel_name = (
        f"fp8_paged_mqa_local_topk_h32d128_k{topk}_{layout_name}_{arch}"
    )

    if preshuffled:

        def _load_k(kv_i32, physical_page, token_in_page, physical, page_size, lane):
            return _load_preshuffled_fp8x32(
                kv_i32,
                physical_page * page_size * fx.Int32(HEAD_DIM),
                token_in_page,
                lane,
            )

    else:

        def _load_k(kv_i32, physical_page, token_in_page, physical, page_size, lane):
            return _load_fp8x32(
                kv_i32,
                physical * fx.Int32(HEAD_DIM),
                lane,
            )

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        kv_cache: fx.Tensor,
        k_scales: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        rows: fx.Int32,
        next_n: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
    ):
        page_size_i32 = fx.Int32(page_size)
        tid = fx.Int32(gpu.thread_idx.x)
        row = fx.Int32(gpu.block_idx.x)
        split = fx.Int32(gpu.block_idx.y)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(kv_cache, dtype=T.i32, shape=(-1,))
        scales = GTensor(k_scales, dtype=T.f32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        output_scores_row = fx.slice(candidate_scores, (row, split, None))
        output_positions_row = fx.slice(candidate_positions, (row, split, None))

        storage = fx.SharedAllocator().allocate(storage_type)
        pool_values = storage.pool_values.peek().view(fx.make_layout(pool_capacity, 1))
        pool_indices = storage.pool_indices.peek().view(
            fx.make_layout(pool_capacity, 1)
        )
        histogram = storage.histogram.peek().view(fx.make_layout(NUM_HIST_BINS, 1))
        scan = storage.scan.peek().view(fx.make_layout(NUM_WAVES + 1, 1))
        state = storage.state.peek().view(fx.make_layout(_STATE_SIZE, 1))

        valid_len = fx.Int32(lengths[row])
        split_begin = _udiv(valid_len * split, num_splits)
        split_end = _udiv(valid_len * (split + fx.Int32(1)), num_splits)
        request = _udiv(row, next_n)

        if tid == 0:
            state[_RETAINED] = 0
        gpu.barrier()

        q_tiles = [None] * M_TILES
        weight_frag = [[None] * DREG for _ in range_constexpr(M_TILES)]
        q_row_bytes = row * fx.Int32(HEADS * HEAD_DIM)
        for mi in range_constexpr(M_TILES):
            q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
            q_tiles[mi] = _load_fp8x32(
                q_i32,
                q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                lane_div_16,
            )
            for ii in range_constexpr(DREG):
                head = mi * MFMA_M + lane_div_16 * DREG + ii
                weight_frag[mi][ii] = fx.Float32(weight_t[row, head])

        def _compact(
            pool_count,
            state,
            histogram,
            scan,
            pool_values,
            pool_indices,
        ):
            if tid == 0:
                state[_SCORE_PREFIX] = 0
                state[_SCORE_MASK] = 0
                state[_REMAINING] = fx.Int32(topk)
            gpu.barrier()

            for radix_pass in range_constexpr(NUM_RADIX_PASSES):
                pass_bits = radix_pass_bits(radix_pass)
                num_bins = 1 << pass_bits
                bins_per_thread = num_bins // BLOCK_THREADS
                for hist_step in range_constexpr(
                    (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    histogram[hist_step * BLOCK_THREADS + tid] = 0
                gpu.barrier()
                score_prefix = state[_SCORE_PREFIX]
                score_mask = state[_SCORE_MASK]
                xor_value = RADIX_SIGN_BIT if radix_pass == 0 else 0

                for step in range_constexpr(pool_steps):
                    pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                    live = pool_pos < pool_count
                    safe_pos = live.select(pool_pos, fx.Int32(0))
                    score_ord = f32_to_ordered_i32(pool_values[safe_pos])
                    prefix_match = live & ((score_ord & score_mask) == score_prefix)
                    if prefix_match:
                        bucket = radix_bucket(score_ord, radix_pass) ^ fx.Int32(
                            xor_value
                        )
                        atomic_add_i32(histogram, 1, bucket, "workgroup")
                gpu.barrier()

                selected_high = (
                    fx.Int32(num_bins - 1) - tid * fx.Int32(bins_per_thread)
                )
                bin_counts = [
                    histogram[selected_high - fx.Int32(bin_item)]
                    for bin_item in range_constexpr(bins_per_thread)
                ]
                group_count = fx.Int32(0)
                for bin_item in range_constexpr(bins_per_thread):
                    group_count = group_count + bin_counts[bin_item]
                before_group, _ = block_exclusive_prefix_i32(
                    tid, group_count, scan
                )
                before_bin = before_group
                remaining = state[_REMAINING]
                for bin_item in range_constexpr(bins_per_thread):
                    selected = selected_high - fx.Int32(bin_item)
                    bin_count = bin_counts[bin_item]
                    if (before_bin < remaining) & (
                        before_bin + bin_count >= remaining
                    ):
                        actual = selected ^ fx.Int32(xor_value)
                        pass_mask, shift = prefix_radix_mask(radix_pass)
                        state[_SCORE_PREFIX] = score_prefix | (
                            actual << fx.Int32(shift)
                        )
                        state[_SCORE_MASK] = score_mask | pass_mask
                        state[_REMAINING] = remaining - before_bin
                    before_bin = before_bin + bin_count
                gpu.barrier()

            threshold = state[_SCORE_PREFIX]
            equal_needed = state[_REMAINING]
            write_cursor = fx.Int32(0)
            equal_seen = fx.Int32(0)
            for step in range_constexpr(pool_steps):
                pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                live = pool_pos < pool_count
                safe_pos = live.select(pool_pos, fx.Int32(0))
                value = pool_values[safe_pos]
                score_ord = f32_to_ordered_i32(value)
                logical_position = fx.Uint16(pool_indices[safe_pos])
                better = live & (score_ord > threshold)
                equal = live & (score_ord == threshold)
                better_i32 = better.select(fx.Int32(1), fx.Int32(0))
                equal_i32 = equal.select(fx.Int32(1), fx.Int32(0))
                packed = (better_i32 << fx.Int32(_PACK_SHIFT)) + equal_i32
                packed_before, packed_total = block_exclusive_prefix_i32(
                    tid, packed, scan
                )
                better_before = packed_before >> fx.Int32(_PACK_SHIFT)
                equal_before = packed_before & fx.Int32(_PACK_MASK)
                better_total = packed_total >> fx.Int32(_PACK_SHIFT)
                equal_total = packed_total & fx.Int32(_PACK_MASK)
                room = equal_needed - equal_seen
                room = (room < 0).select(fx.Int32(0), room)
                admit_equal = equal & (equal_before < room)
                keep = better | admit_equal
                admitted_equal_before = (equal_before < room).select(equal_before, room)
                destination = write_cursor + better_before + admitted_equal_before
                if keep:
                    pool_values[destination] = value
                    pool_indices[destination] = logical_position
                admitted_equal_total = (equal_total < room).select(equal_total, room)
                write_cursor = write_cursor + better_total + admitted_equal_total
                equal_seen = equal_seen + equal_total
            if tid == 0:
                state[_RETAINED] = fx.Int32(topk)
            gpu.barrier()

        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tile_number = fx.Int32(0)
        for col0 in range(split_begin, split_end, fx.Int32(BLOCK_N)):
            batch_tile = _umod(tile_number, TILES_PER_BATCH)
            batch_first = col0 - batch_tile * fx.Int32(BLOCK_N)
            wave_tile_base = wave * fx.Int32(N_TILES_PER_WAVE)
            logical_tiles = []
            k_packs = []
            scale_tiles = []
            for ni in range_constexpr(N_TILES_PER_WAVE):
                logical = (
                    col0
                    + (wave_tile_base + fx.Int32(ni)) * fx.Int32(MFMA_N)
                    + lane_mod_16
                )
                safe_logical = _imin(logical, split_end - fx.Int32(1))
                table_offset = request * max_pages + _udiv(
                    safe_logical, page_size_i32
                )
                physical_page = fx.Int32(tables[table_offset])
                token_in_page = _umod(safe_logical, page_size_i32)
                physical = physical_page * page_size_i32 + token_in_page
                k_pack = _load_k(
                    kv_i32,
                    physical_page,
                    token_in_page,
                    physical,
                    page_size_i32,
                    lane_div_16,
                )
                scale = fx.Float32(scales[physical])
                logical_tiles.append(logical)
                k_packs.append(k_pack)
                scale_tiles.append(scale)

            for ni in range_constexpr(N_TILES_PER_WAVE):
                logical = logical_tiles[ni]
                k_pack = k_packs[ni]
                scale = scale_tiles[ni]
                total = fx.Float32(0.0)
                for mi in range_constexpr(M_TILES):
                    acc = fx.Vector.filled(DREG, 0.0, fx.Float32)
                    acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        result_type,
                        [q_tiles[mi], k_pack, acc, 0, 0, 0, neutral, 0, neutral],
                    )
                    frag = fx.Vector(acc)
                    for ii in range_constexpr(DREG):
                        scaled = fx.Float32(frag[ii]) * scale
                        activated = scaled.maximumf(fx.Float32(0.0))
                        total = total + activated * weight_frag[mi][ii]
                total = total + total.shuffle_xor(16, WAVE_SIZE)
                total = total + total.shuffle_xor(32, WAVE_SIZE)

                is_writer = (lane_div_16 == 0) & (logical < split_end)
                destination = state[_RETAINED] + logical - batch_first

                @flyc.jit
                def _write_candidate(
                    _pred=is_writer,
                    _dst=destination,
                    _score=total,
                    _logical=logical,
                    _values=pool_values,
                    _indices=pool_indices,
                ):
                    if _pred:
                        _values[_dst] = _score
                        _indices[_dst] = fx.Uint16(_logical - split_begin)

                _write_candidate()

            end_of_batch = batch_tile == fx.Int32(TILES_PER_BATCH - 1)
            final_tile = col0 + fx.Int32(BLOCK_N) >= split_end
            if end_of_batch | final_tile:
                gpu.barrier()
                batch_end = _imin(col0 + fx.Int32(BLOCK_N), split_end)
                pool_count = state[_RETAINED] + batch_end - batch_first
                if pool_count > fx.Int32(topk):
                    _compact(
                        pool_count,
                        state,
                        histogram,
                        scan,
                        pool_values,
                        pool_indices,
                    )
                else:
                    if tid == 0:
                        state[_RETAINED] = pool_count
                    gpu.barrier()
            tile_number = tile_number + fx.Int32(1)

        retained = state[_RETAINED]
        for step in range_constexpr(output_steps):
            slot = fx.Int32(step * BLOCK_THREADS) + tid
            if slot < fx.Int32(topk):
                live = slot < retained
                safe_slot = live.select(slot, fx.Int32(0))
                output_scores_row[slot] = live.select(
                    pool_values[safe_slot], fx.Float32(float("-inf"))
                )
                output_positions_row[slot] = live.select(
                    split_begin + fx.Int32(pool_indices[safe_slot]),
                    fx.Int32(-1),
                )
        if tid == 0:
            fx.ptr_store(
                retained,
                fx.add_offset(fx.get_iter(candidate_counts), row * num_splits + split),
            )

    @flyc.jit
    def launch(
        q_fp8: fx.Tensor,
        kv_cache: fx.Tensor,
        k_scales: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        rows: fx.Int32,
        next_n: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(rows))
        gy = arith.index_cast(T.index, _to_raw(num_splits))
        kernel(
            q_fp8,
            kv_cache,
            k_scales,
            weights,
            context_lens,
            block_tables,
            candidate_scores,
            candidate_positions,
            candidate_counts,
            rows,
            next_n,
            num_splits,
            max_pages,
        ).launch(grid=(gx, gy, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": True}
    return launch


def launch_fp8_paged_mqa_local_topk(
    q_fp8,
    kv_cache,
    k_scales,
    weights,
    context_lens,
    block_tables,
    candidate_scores,
    candidate_positions,
    candidate_counts,
    *,
    topk,
    num_splits,
    preshuffled,
    arch,
    stream,
):
    page_size = kv_cache.shape[1]
    launcher = compile_fp8_paged_mqa_local_topk(
        topk=topk,
        arch=arch,
        preshuffled=preshuffled,
        page_size=page_size,
    )
    batch, next_n, _, _ = q_fp8.shape
    rows = batch * next_n
    max_pages = block_tables.shape[1]
    _run_compiled(
        launcher,
        q_fp8,
        kv_cache,
        k_scales,
        weights,
        context_lens,
        block_tables,
        candidate_scores,
        candidate_positions,
        candidate_counts,
        rows,
        next_n,
        num_splits,
        max_pages,
        stream,
    )
