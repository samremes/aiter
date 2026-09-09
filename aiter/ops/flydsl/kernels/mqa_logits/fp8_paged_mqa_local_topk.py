# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental paged FP8 MQA scorer with an exact LDS local-TopK reservoir."""

# FlyDSL argument annotations must remain concrete at trace time.
import os
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..candidate_topk_common import (
    NUM_HIST_BINS,
    NUM_RADIX_PASSES,
    RADIX_SIGN_BIT,
    WAVE_SIZE,
    f32_to_ordered_i32,
    make_block_exclusive_prefix_i32,
    make_streaming_topk_storage,
    prefix_radix_mask,
    radix_bucket,
    radix_pass_bits,
)
from ..kernels_common import atomic_add_i32
from ..tensor_shim import GTensor, _run_compiled, _to_raw

HEADS = 32
HEAD_DIM = 128
PROFILE_WAVES = int(os.environ.get("AITER_STAGE_A_PROFILE_WAVES", "8"))
if PROFILE_WAVES not in (4, 8):
    raise ValueError("AITER_STAGE_A_PROFILE_WAVES must be 4 or 8")
BLOCK_THREADS = PROFILE_WAVES * WAVE_SIZE
BLOCK_N = PROFILE_WAVES * 32
WAVES = BLOCK_THREADS // WAVE_SIZE
MFMA_M = 16
MFMA_N = 16
MFMA_K = 128
M_TILES = HEADS // MFMA_M
N_TILES_PER_WAVE = (BLOCK_N // MFMA_N) // WAVES
DREG = 4
# Keep the 4-wave compact cadence (26 score tiles) when BLOCK_N grows.
TILES_PER_COMPACT = 26
INCOMING_CAPACITY = TILES_PER_COMPACT * BLOCK_N
MTP2_TILES_PER_COMPACT = int(
    os.environ.get("AITER_STAGE_A_MTP2_TILES_PER_COMPACT", "32")
)
if MTP2_TILES_PER_COMPACT <= 0:
    raise ValueError("AITER_STAGE_A_MTP2_TILES_PER_COMPACT must be positive")
MTP4_TILES_PER_COMPACT = int(
    os.environ.get("AITER_STAGE_A_MTP4_TILES_PER_COMPACT", "16")
)
if MTP4_TILES_PER_COMPACT <= 0:
    raise ValueError("AITER_STAGE_A_MTP4_TILES_PER_COMPACT must be positive")
MTP_REUSE_MAX_ROWS = int(os.environ.get("AITER_STAGE_A_MTP_REUSE_MAX_ROWS", "8"))
if MTP_REUSE_MAX_ROWS <= 0:
    raise ValueError("AITER_STAGE_A_MTP_REUSE_MAX_ROWS must be positive")


def grouped_rows_per_cta(next_n: int, rows: int) -> int:
    """Launch even ``next_n`` as consecutive two-row CTAs when the grid is small.

    Pairs stay inside one request because ``next_n`` is even. Above
    ``MTP_REUSE_MAX_ROWS`` one-row CTAs are faster: the two-row full-split
    reservoir needs 64 splits at L=1M, which oversubscribes, and streaming
    two-row compact is slower than the one-row path.
    """
    if (
        next_n >= 2
        and next_n % 2 == 0
        and rows % 2 == 0
        and rows <= MTP_REUSE_MAX_ROWS
    ):
        return 2
    return 1


def _env_flag(name: str, *, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback) in ("1", "true", "True", "yes", "YES")


# Diagnostic-only compile switches. Default keeps exact Stage A.
SKIP_COMPACT = _env_flag("AITER_STAGE_A_SKIP_COMPACT")
SKIP_CANDIDATE_WRITES = _env_flag("AITER_STAGE_A_SKIP_CANDIDATE_WRITES")
SKIP_EPILOGUE = _env_flag("AITER_STAGE_A_SKIP_EPILOGUE")
# The measured winner for two-row CTAs. Set the environment variable to 0 to
# retain the streaming implementation for diagnostics.
MTP2_FULL_SPLIT_SELECT = _env_flag(
    "AITER_STAGE_A_MTP2_FULL_SPLIT_SELECT", default=True
)
# 0 identity, 1 swap grid axes, 2 HipKittens Alg1 C=2, 3 Alg1 C=4,
# 4 pack each split's row-groups onto one assumed XCD (id%8).
# Map 1 is the measured default: same-split CTAs share an XCD when
# num_splits % 8 == 0.
XCD_MAP = int(os.environ.get("AITER_STAGE_A_XCD_MAP", "1"))
if XCD_MAP not in (0, 1, 2, 3, 4):
    raise ValueError("AITER_STAGE_A_XCD_MAP must be 0..4")
NUM_XCD = 8
WORKGROUPS_PER_CU = 16 // PROFILE_WAVES
SUPPORTED_K = (128, 512, 1024, 2048)
FULL_SPLIT_CAPACITY = 16384

_RETAINED = 0
_SCORE_PREFIX = 1
_SCORE_MASK = 2
_REMAINING = 3
_STATE_SIZE = 4
_MERGE_PREFIX = 0
_MERGE_MASK = 1
_MERGE_REMAINING = 2
_MERGE_WRITE_COUNTER = 3
_MERGE_EQ_COUNTER = 4
_MERGE_DIRECT = 5
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1
_NEUTRAL_E8M0 = 0x7F7F7F7F


def _make_full_split_storage(state_size):
    """Create score-only LDS storage for two full 16K splits."""

    @fx.struct
    class FullSplitStorage:
        pool_values: fx.Array[fx.Float32, 2 * FULL_SPLIT_CAPACITY, 16]
        pool_indices: fx.Array[fx.Uint16, 1, 16]
        histogram: fx.Array[fx.Int32, NUM_HIST_BINS, 16]
        scan: fx.Array[fx.Int32, WAVES + 1, 16]
        state: fx.Array[fx.Int32, state_size, 16]

    return FullSplitStorage


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


def _alg1_xy2(flat, chunk):
    xcd = _umod(flat, NUM_XCD)
    local = _udiv(flat, NUM_XCD)
    chunk_idx = _udiv(local, chunk)
    pos = _umod(local, chunk)
    return (
        chunk_idx * fx.Int32(NUM_XCD * chunk)
        + xcd * fx.Int32(chunk)
        + pos
    )


@lru_cache(maxsize=32)
def compile_fp8_paged_mqa_local_topk(
    *,
    topk: int,
    arch: str,
    preshuffled: bool,
    page_size: int,
    rows_per_cta: int = 1,
    prepare_merge: bool = False,
    full_split_select: bool = False,
    skip_compact: bool = False,
    skip_candidate_writes: bool = False,
    skip_epilogue: bool = False,
    xcd_map: int = 0,
):
    """Compile an H32D128 Stage-A specialization."""
    if topk not in SUPPORTED_K:
        raise ValueError(f"topk must be one of {SUPPORTED_K}, got {topk}")
    if rows_per_cta not in (1, 2, 3, 4):
        raise ValueError(f"rows_per_cta must be in [1,4], got {rows_per_cta}")
    if full_split_select and rows_per_cta != 2:
        raise ValueError("full_split_select requires rows_per_cta=2")
    if xcd_map not in (0, 1, 2, 3, 4):
        raise ValueError(f"xcd_map must be 0..4, got {xcd_map}")
    if arch != "gfx950":
        raise RuntimeError(
            "the initial native-e4m3fn Stage-A specialization supports gfx950 only"
        )

    tiles_per_compact = {
        1: TILES_PER_COMPACT,
        2: MTP2_TILES_PER_COMPACT,
        3: MTP4_TILES_PER_COMPACT,
        4: MTP4_TILES_PER_COMPACT,
    }[rows_per_cta]
    incoming_capacity = tiles_per_compact * BLOCK_N
    pool_capacity = (
        FULL_SPLIT_CAPACITY if full_split_select else topk + incoming_capacity
    )
    index_capacity = 1 if full_split_select else rows_per_cta * pool_capacity
    pool_steps = (pool_capacity + BLOCK_THREADS - 1) // BLOCK_THREADS
    output_steps = (topk + BLOCK_THREADS - 1) // BLOCK_THREADS
    storage_type = (
        _make_full_split_storage(rows_per_cta * _STATE_SIZE)
        if full_split_select
        else make_streaming_topk_storage(
            rows_per_cta * pool_capacity,
            rows_per_cta * _STATE_SIZE,
            fx.Uint16,
            num_waves=WAVES,
        )
    )
    block_exclusive_prefix_i32 = make_block_exclusive_prefix_i32(WAVES)
    layout_name = "preshuffled" if preshuffled else "rowmajor"
    kernel_name = (
        f"fp8_paged_mqa_local_topk_h32d128_k{topk}_{layout_name}_"
        f"w{WAVES}_bn{BLOCK_N}_r{rows_per_cta}_inc{incoming_capacity}_"
        f"{f'fs{int(full_split_select)}_' if rows_per_cta == 2 else ''}"
        f"pm{int(prepare_merge)}_sc{int(skip_compact)}_"
        f"sw{int(skip_candidate_writes)}_se{int(skip_epilogue)}_"
        f"xcd{int(xcd_map)}_{arch}"
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
        merge_histogram: fx.Tensor,
        merge_state: fx.Tensor,
        rows: fx.Int32,
        next_n: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
    ):
        page_size_i32 = fx.Int32(page_size)
        tid = fx.Int32(gpu.thread_idx.x)
        bx = fx.Int32(gpu.block_idx.x)
        by = fx.Int32(gpu.block_idx.y)
        row_groups = _udiv(rows, fx.Int32(rows_per_cta))
        row_group = bx
        split = by
        if const_expr(xcd_map == 1):
            row_group = by
            split = bx
        elif const_expr(xcd_map == 2) or const_expr(xcd_map == 3):
            flat = bx + row_groups * by
            total = row_groups * num_splits
            chunk = 2 if xcd_map == 2 else 4
            stride = fx.Int32(NUM_XCD * chunk)
            can_alg1 = _umod(total, stride) == fx.Int32(0)
            xy2 = _alg1_xy2(flat, chunk)
            row_group = can_alg1.select(_umod(xy2, row_groups), bx)
            split = can_alg1.select(_udiv(xy2, row_groups), by)
        elif const_expr(xcd_map == 4):
            flat = bx + row_groups * by
            can_pack = _umod(num_splits, fx.Int32(NUM_XCD)) == fx.Int32(0)
            xcd = _umod(flat, NUM_XCD)
            local = _udiv(flat, NUM_XCD)
            packed_row = _umod(local, row_groups)
            packed_split = xcd + _udiv(local, row_groups) * fx.Int32(NUM_XCD)
            row_group = can_pack.select(packed_row, bx)
            split = can_pack.select(packed_split, by)
        row_base = row_group * fx.Int32(rows_per_cta)
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

        storage = fx.SharedAllocator().allocate(storage_type)
        pool_values = storage.pool_values.peek().view(
            fx.make_layout(rows_per_cta * pool_capacity, 1)
        )
        pool_indices = storage.pool_indices.peek().view(
            fx.make_layout(index_capacity, 1)
        )
        histogram = storage.histogram.peek().view(fx.make_layout(NUM_HIST_BINS, 1))
        scan = storage.scan.peek().view(fx.make_layout(WAVES + 1, 1))
        state = storage.state.peek().view(fx.make_layout(rows_per_cta * _STATE_SIZE, 1))

        valid_lens = [
            fx.Int32(lengths[row_base + fx.Int32(row_in_group)])
            for row_in_group in range_constexpr(rows_per_cta)
        ]
        group_valid_len = valid_lens[0]
        for row_in_group in range_constexpr(1, rows_per_cta):
            group_valid_len = (valid_lens[row_in_group] > group_valid_len).select(
                valid_lens[row_in_group], group_valid_len
            )
        split_begin = _udiv(group_valid_len * split, num_splits)
        split_end = _udiv(
            group_valid_len * (split + fx.Int32(1)),
            num_splits,
        )
        request = _udiv(row_base, next_n)

        if tid == 0:
            for row_in_group in range_constexpr(rows_per_cta):
                state[row_in_group * _STATE_SIZE + _RETAINED] = 0
        gpu.barrier()

        q_tiles = [[None] * M_TILES for _ in range_constexpr(rows_per_cta)]
        weight_frags = [
            [[None] * DREG for _ in range_constexpr(M_TILES)]
            for _ in range_constexpr(rows_per_cta)
        ]
        for row_in_group in range_constexpr(rows_per_cta):
            row = row_base + fx.Int32(row_in_group)
            q_row_bytes = row * fx.Int32(HEADS * HEAD_DIM)
            for mi in range_constexpr(M_TILES):
                q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
                q_tiles[row_in_group][mi] = _load_fp8x32(
                    q_i32,
                    q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                    lane_div_16,
                )
                for ii in range_constexpr(DREG):
                    head = mi * MFMA_M + lane_div_16 * DREG + ii
                    weight_frags[row_in_group][mi][ii] = fx.Float32(weight_t[row, head])

        def _compact(
            pool_count,
            pool_base,
            state_base,
            state,
            histogram,
            scan,
            pool_values,
            pool_indices,
        ):
            if tid == 0:
                state[state_base + _SCORE_PREFIX] = 0
                state[state_base + _SCORE_MASK] = 0
                state[state_base + _REMAINING] = fx.Int32(topk)
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
                score_prefix = state[state_base + _SCORE_PREFIX]
                score_mask = state[state_base + _SCORE_MASK]
                xor_value = RADIX_SIGN_BIT if radix_pass == 0 else 0

                for step in range_constexpr(pool_steps):
                    pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                    live = pool_pos < pool_count
                    safe_pos = live.select(pool_pos, fx.Int32(0))
                    score_ord = f32_to_ordered_i32(pool_values[pool_base + safe_pos])
                    prefix_match = live & ((score_ord & score_mask) == score_prefix)
                    if prefix_match:
                        bucket = radix_bucket(score_ord, radix_pass) ^ fx.Int32(
                            xor_value
                        )
                        atomic_add_i32(histogram, 1, bucket, "workgroup")
                gpu.barrier()

                selected_high = fx.Int32(num_bins - 1) - tid * fx.Int32(bins_per_thread)
                bin_counts = [
                    histogram[selected_high - fx.Int32(bin_item)]
                    for bin_item in range_constexpr(bins_per_thread)
                ]
                group_count = fx.Int32(0)
                for bin_item in range_constexpr(bins_per_thread):
                    group_count = group_count + bin_counts[bin_item]
                before_group, _ = block_exclusive_prefix_i32(tid, group_count, scan)
                before_bin = before_group
                remaining = state[state_base + _REMAINING]
                for bin_item in range_constexpr(bins_per_thread):
                    selected = selected_high - fx.Int32(bin_item)
                    bin_count = bin_counts[bin_item]
                    if (before_bin < remaining) & (before_bin + bin_count >= remaining):
                        actual = selected ^ fx.Int32(xor_value)
                        pass_mask, shift = prefix_radix_mask(radix_pass)
                        state[state_base + _SCORE_PREFIX] = score_prefix | (
                            actual << fx.Int32(shift)
                        )
                        state[state_base + _SCORE_MASK] = score_mask | pass_mask
                        state[state_base + _REMAINING] = remaining - before_bin
                    before_bin = before_bin + bin_count
                gpu.barrier()

            threshold = state[state_base + _SCORE_PREFIX]
            equal_needed = state[state_base + _REMAINING]
            write_cursor = fx.Int32(0)
            equal_seen = fx.Int32(0)
            for step in range_constexpr(pool_steps):
                pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                live = pool_pos < pool_count
                safe_pos = live.select(pool_pos, fx.Int32(0))
                value = pool_values[pool_base + safe_pos]
                score_ord = f32_to_ordered_i32(value)
                logical_position = fx.Uint16(pool_indices[pool_base + safe_pos])
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
                    pool_values[pool_base + destination] = value
                    pool_indices[pool_base + destination] = logical_position
                admitted_equal_total = (equal_total < room).select(equal_total, room)
                write_cursor = write_cursor + better_total + admitted_equal_total
                equal_seen = equal_seen + equal_total
            if tid == 0:
                state[state_base + _RETAINED] = fx.Int32(topk)
            gpu.barrier()

        def _full_split_select(
            pool_count,
            pool_base,
            state_base,
            state,
            histogram,
            scan,
            pool_values,
            output_scores_row,
            output_positions_row,
            merge_histogram_row,
            merge_state_row,
            valid_len,
        ):
            if tid == 0:
                state[state_base + _SCORE_PREFIX] = 0
                state[state_base + _SCORE_MASK] = 0
                state[state_base + _REMAINING] = fx.Int32(topk)
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
                score_prefix = state[state_base + _SCORE_PREFIX]
                score_mask = state[state_base + _SCORE_MASK]
                xor_value = RADIX_SIGN_BIT if radix_pass == 0 else 0

                for step in range_constexpr(pool_steps):
                    pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                    live = pool_pos < pool_count
                    safe_pos = live.select(pool_pos, fx.Int32(0))
                    score_ord = f32_to_ordered_i32(pool_values[pool_base + safe_pos])
                    prefix_match = live & ((score_ord & score_mask) == score_prefix)
                    if prefix_match:
                        bucket = radix_bucket(score_ord, radix_pass) ^ fx.Int32(
                            xor_value
                        )
                        atomic_add_i32(histogram, 1, bucket, "workgroup")
                gpu.barrier()

                selected_high = fx.Int32(num_bins - 1) - tid * fx.Int32(bins_per_thread)
                bin_counts = [
                    histogram[selected_high - fx.Int32(bin_item)]
                    for bin_item in range_constexpr(bins_per_thread)
                ]
                group_count = fx.Int32(0)
                for bin_item in range_constexpr(bins_per_thread):
                    group_count = group_count + bin_counts[bin_item]
                before_group, _ = block_exclusive_prefix_i32(tid, group_count, scan)
                before_bin = before_group
                remaining = state[state_base + _REMAINING]
                for bin_item in range_constexpr(bins_per_thread):
                    selected = selected_high - fx.Int32(bin_item)
                    bin_count = bin_counts[bin_item]
                    if (before_bin < remaining) & (before_bin + bin_count >= remaining):
                        actual = selected ^ fx.Int32(xor_value)
                        pass_mask, shift = prefix_radix_mask(radix_pass)
                        state[state_base + _SCORE_PREFIX] = score_prefix | (
                            actual << fx.Int32(shift)
                        )
                        state[state_base + _SCORE_MASK] = score_mask | pass_mask
                        state[state_base + _REMAINING] = remaining - before_bin
                    before_bin = before_bin + bin_count
                gpu.barrier()

            threshold = state[state_base + _SCORE_PREFIX]
            equal_needed = state[state_base + _REMAINING]
            write_cursor = fx.Int32(0)
            equal_seen = fx.Int32(0)
            for step in range_constexpr(output_steps):
                slot = fx.Int32(step * BLOCK_THREADS) + tid
                if slot < fx.Int32(topk):
                    output_scores_row[slot] = fx.Float32(float("-inf"))
                    output_positions_row[slot] = fx.Int32(-1)
            for hist_step in range_constexpr(
                (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
            ):
                histogram[hist_step * BLOCK_THREADS + tid] = 0
            if prepare_merge and split == 0 and tid == 0:
                merge_state_row[_MERGE_PREFIX] = 0
                merge_state_row[_MERGE_MASK] = 0
                merge_state_row[_MERGE_REMAINING] = _imin(
                    valid_len,
                    fx.Int32(topk),
                )
                merge_state_row[_MERGE_WRITE_COUNTER] = 0
                merge_state_row[_MERGE_EQ_COUNTER] = 0
                merge_state_row[_MERGE_DIRECT] = 0
            gpu.barrier()

            for step in range_constexpr(pool_steps):
                pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                live = pool_pos < pool_count
                safe_pos = live.select(pool_pos, fx.Int32(0))
                value = pool_values[pool_base + safe_pos]
                score_ord = f32_to_ordered_i32(value)
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
                    output_scores_row[destination] = value
                    output_positions_row[destination] = split_begin + pool_pos
                    if prepare_merge:
                        bucket = radix_bucket(score_ord, 0) ^ fx.Int32(RADIX_SIGN_BIT)
                        atomic_add_i32(histogram, 1, bucket, "workgroup")
                admitted_equal_total = (equal_total < room).select(equal_total, room)
                write_cursor = write_cursor + better_total + admitted_equal_total
                equal_seen = equal_seen + equal_total
            gpu.barrier()

            if prepare_merge:
                for hist_step in range_constexpr(
                    (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    hist_bin = hist_step * BLOCK_THREADS + tid
                    count = histogram[hist_bin]
                    if count != 0:
                        atomic_add_i32(
                            merge_histogram_row,
                            count,
                            hist_bin,
                            "agent",
                        )
                gpu.barrier()
            if tid == 0:
                state[state_base + _RETAINED] = fx.Int32(topk)

        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tile_number = fx.Int32(0)
        score_keep = fx.Float32(0.0)
        for col0 in range(split_begin, split_end, fx.Int32(BLOCK_N)):
            if const_expr(full_split_select):
                batch_tile = fx.Int32(0)
                batch_first = split_begin
            else:
                batch_tile = _umod(tile_number, tiles_per_compact)
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
                table_offset = request * max_pages + _udiv(safe_logical, page_size_i32)
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
                for row_in_group in range_constexpr(rows_per_cta):
                    row = row_base + fx.Int32(row_in_group)
                    total = fx.Float32(0.0)
                    for mi in range_constexpr(M_TILES):
                        q_pack = q_tiles[row_in_group][mi]
                        acc = fx.Vector.filled(DREG, 0.0, fx.Float32)
                        acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            result_type,
                            [
                                q_pack,
                                k_pack,
                                acc,
                                0,
                                0,
                                0,
                                neutral,
                                0,
                                neutral,
                            ],
                        )
                        frag = fx.Vector(acc)
                        for ii in range_constexpr(DREG):
                            scaled = fx.Float32(frag[ii]) * scale
                            activated = scaled.maximumf(fx.Float32(0.0))
                            weight = weight_frags[row_in_group][mi][ii]
                            total = total + activated * weight
                    total = total + total.shuffle_xor(16, WAVE_SIZE)
                    total = total + total.shuffle_xor(32, WAVE_SIZE)

                    state_base = fx.Int32(row_in_group * _STATE_SIZE)
                    pool_base = fx.Int32(row_in_group * pool_capacity)
                    is_writer = (
                        (lane_div_16 == 0)
                        & (logical < split_end)
                        & (logical < valid_lens[row_in_group])
                    )
                    destination = (
                        pool_base
                        + (
                            fx.Int32(0)
                            if const_expr(full_split_select)
                            else state[state_base + _RETAINED]
                        )
                        + logical
                        - batch_first
                    )

                    if not const_expr(skip_candidate_writes):
                        if const_expr(full_split_select):

                            @flyc.jit
                            def _write_full_split_score(
                                _pred=is_writer,
                                _dst=destination,
                                _score=total,
                                _values=pool_values,
                            ):
                                if _pred:
                                    _values[_dst] = _score

                            _write_full_split_score()
                        else:

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
                    else:
                        score_keep = score_keep + total

            end_of_batch = batch_tile == fx.Int32(tiles_per_compact - 1)
            final_tile = col0 + fx.Int32(BLOCK_N) >= split_end
            if (
                not const_expr(skip_compact)
                and not const_expr(full_split_select)
                and (end_of_batch | final_tile)
            ):
                gpu.barrier()
                batch_end = _imin(col0 + fx.Int32(BLOCK_N), split_end)
                for row_in_group in range_constexpr(rows_per_cta):
                    state_base = fx.Int32(row_in_group * _STATE_SIZE)
                    pool_base = fx.Int32(row_in_group * pool_capacity)
                    incoming_end = _imin(
                        batch_end,
                        valid_lens[row_in_group],
                    )
                    incoming_count = (incoming_end > batch_first).select(
                        incoming_end - batch_first,
                        fx.Int32(0),
                    )
                    pool_count = state[state_base + _RETAINED] + incoming_count
                    if pool_count > fx.Int32(topk):
                        _compact(
                            pool_count,
                            pool_base,
                            state_base,
                            state,
                            histogram,
                            scan,
                            pool_values,
                            pool_indices,
                        )
                    else:
                        if tid == 0:
                            state[state_base + _RETAINED] = pool_count
                        gpu.barrier()
            tile_number = tile_number + fx.Int32(1)

        if const_expr(skip_candidate_writes) and tid == 0:
            scan[0] = f32_to_ordered_i32(score_keep)

        if const_expr(skip_epilogue):
            return

        if const_expr(full_split_select):
            for row_in_group in range_constexpr(rows_per_cta):
                row = row_base + fx.Int32(row_in_group)
                state_base = fx.Int32(row_in_group * _STATE_SIZE)
                pool_base = fx.Int32(row_in_group * pool_capacity)
                valid_len = valid_lens[row_in_group]
                split_row_end = _imin(split_end, valid_len)
                pool_count = (split_row_end > split_begin).select(
                    split_row_end - split_begin,
                    fx.Int32(0),
                )
                output_scores_row = fx.slice(
                    candidate_scores,
                    (row, split, None),
                )
                output_positions_row = fx.slice(
                    candidate_positions,
                    (row, split, None),
                )
                merge_histogram_row = fx.slice(
                    merge_histogram,
                    (row, 0, None),
                )
                merge_state_row = fx.slice(
                    merge_state,
                    (row, None),
                )
                if not const_expr(skip_compact):
                    if pool_count > fx.Int32(topk):
                        _full_split_select(
                            pool_count,
                            pool_base,
                            state_base,
                            state,
                            histogram,
                            scan,
                            pool_values,
                            output_scores_row,
                            output_positions_row,
                            merge_histogram_row,
                            merge_state_row,
                            valid_len,
                        )
                    else:
                        if const_expr(prepare_merge):
                            for hist_step in range_constexpr(
                                (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                            ):
                                histogram[hist_step * BLOCK_THREADS + tid] = 0
                            if split == 0 and tid == 0:
                                merge_state_row[_MERGE_PREFIX] = 0
                                merge_state_row[_MERGE_MASK] = 0
                                merge_state_row[_MERGE_REMAINING] = _imin(
                                    valid_len,
                                    fx.Int32(topk),
                                )
                                merge_state_row[_MERGE_WRITE_COUNTER] = 0
                                merge_state_row[_MERGE_EQ_COUNTER] = 0
                                merge_state_row[_MERGE_DIRECT] = 0
                            gpu.barrier()
                        for step in range_constexpr(output_steps):
                            slot = fx.Int32(step * BLOCK_THREADS) + tid
                            if slot < fx.Int32(topk):
                                live = slot < pool_count
                                safe_slot = live.select(slot, fx.Int32(0))
                                candidate_score = pool_values[pool_base + safe_slot]
                                output_scores_row[slot] = live.select(
                                    candidate_score,
                                    fx.Float32(float("-inf")),
                                )
                                output_positions_row[slot] = live.select(
                                    split_begin + slot,
                                    fx.Int32(-1),
                                )
                                if const_expr(prepare_merge):

                                    @flyc.jit
                                    def _count_direct_candidate(
                                        _pred=live,
                                        _score=candidate_score,
                                        _histogram=histogram,
                                    ):
                                        if _pred:
                                            bucket = radix_bucket(
                                                f32_to_ordered_i32(_score),
                                                0,
                                            ) ^ fx.Int32(RADIX_SIGN_BIT)
                                            atomic_add_i32(
                                                _histogram,
                                                1,
                                                bucket,
                                                "workgroup",
                                            )

                                    _count_direct_candidate()
                        if const_expr(prepare_merge):
                            gpu.barrier()
                            for hist_step in range_constexpr(
                                (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                            ):
                                hist_bin = hist_step * BLOCK_THREADS + tid
                                count = histogram[hist_bin]
                                if count != 0:
                                    atomic_add_i32(
                                        merge_histogram_row,
                                        count,
                                        hist_bin,
                                        "agent",
                                    )
                            gpu.barrier()
                        if tid == 0:
                            state[state_base + _RETAINED] = pool_count
                        gpu.barrier()
                else:
                    for step in range_constexpr(output_steps):
                        slot = fx.Int32(step * BLOCK_THREADS) + tid
                        if slot < fx.Int32(topk):
                            output_scores_row[slot] = fx.Float32(float("-inf"))
                            output_positions_row[slot] = fx.Int32(-1)
                    if tid == 0:
                        state[state_base + _RETAINED] = 0
                    gpu.barrier()
                if tid == 0:
                    fx.ptr_store(
                        state[state_base + _RETAINED],
                        fx.add_offset(
                            fx.get_iter(candidate_counts),
                            row * num_splits + split,
                        ),
                    )
            return

        for row_in_group in range_constexpr(rows_per_cta):
            row = row_base + fx.Int32(row_in_group)
            state_base = fx.Int32(row_in_group * _STATE_SIZE)
            pool_base = fx.Int32(row_in_group * pool_capacity)
            retained = state[state_base + _RETAINED]
            valid_len = valid_lens[row_in_group]
            output_scores_row = fx.slice(
                candidate_scores,
                (row, split, None),
            )
            output_positions_row = fx.slice(
                candidate_positions,
                (row, split, None),
            )
            merge_histogram_row = fx.slice(
                merge_histogram,
                (row, 0, None),
            )
            merge_state_row = fx.slice(
                merge_state,
                (row, None),
            )
            if const_expr(prepare_merge):
                for hist_step in range_constexpr(
                    (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    histogram[hist_step * BLOCK_THREADS + tid] = 0
                if split == 0 and tid == 0:
                    merge_state_row[_MERGE_PREFIX] = 0
                    merge_state_row[_MERGE_MASK] = 0
                    merge_state_row[_MERGE_REMAINING] = _imin(
                        valid_len,
                        fx.Int32(topk),
                    )
                    merge_state_row[_MERGE_WRITE_COUNTER] = 0
                    merge_state_row[_MERGE_EQ_COUNTER] = 0
                    merge_state_row[_MERGE_DIRECT] = 0
                gpu.barrier()

            for step in range_constexpr(output_steps):
                slot = fx.Int32(step * BLOCK_THREADS) + tid
                if slot < fx.Int32(topk):
                    live = slot < retained
                    safe_slot = live.select(slot, fx.Int32(0))
                    candidate_score = pool_values[pool_base + safe_slot]
                    output_scores_row[slot] = live.select(
                        candidate_score,
                        fx.Float32(float("-inf")),
                    )
                    output_positions_row[slot] = live.select(
                        split_begin + fx.Int32(pool_indices[pool_base + safe_slot]),
                        fx.Int32(-1),
                    )
                    if const_expr(prepare_merge):

                        @flyc.jit
                        def _count_candidate(
                            _pred=live,
                            _score=candidate_score,
                            _histogram=histogram,
                        ):
                            if _pred:
                                bucket = radix_bucket(
                                    f32_to_ordered_i32(_score),
                                    0,
                                ) ^ fx.Int32(RADIX_SIGN_BIT)
                                atomic_add_i32(
                                    _histogram,
                                    1,
                                    bucket,
                                    "workgroup",
                                )

                        _count_candidate()
            if const_expr(prepare_merge):
                gpu.barrier()
                for hist_step in range_constexpr(
                    (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    hist_bin = hist_step * BLOCK_THREADS + tid
                    count = histogram[hist_bin]
                    if count != 0:
                        atomic_add_i32(
                            merge_histogram_row,
                            count,
                            hist_bin,
                            "agent",
                        )
                gpu.barrier()
            if tid == 0:
                fx.ptr_store(
                    retained,
                    fx.add_offset(
                        fx.get_iter(candidate_counts),
                        row * num_splits + split,
                    ),
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
        merge_histogram: fx.Tensor,
        merge_state: fx.Tensor,
        rows: fx.Int32,
        next_n: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        stream: fx.Stream,
    ):
        row_groups = _udiv(rows, fx.Int32(rows_per_cta))
        if const_expr(xcd_map == 1):
            gx = arith.index_cast(T.index, _to_raw(num_splits))
            gy = arith.index_cast(T.index, _to_raw(row_groups))
        else:
            gx = arith.index_cast(T.index, _to_raw(row_groups))
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
            merge_histogram,
            merge_state,
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
    prepare_merge=False,
    merge_histogram=None,
    merge_state=None,
):
    page_size = kv_cache.shape[1]
    batch, next_n, _, _ = q_fp8.shape
    rows = batch * next_n
    rows_per_cta = grouped_rows_per_cta(next_n, rows)
    max_split_span = (block_tables.shape[1] * page_size + num_splits - 1) // num_splits
    full_split_select = (
        MTP2_FULL_SPLIT_SELECT
        and rows_per_cta == 2
        and max_split_span <= FULL_SPLIT_CAPACITY
    )
    launcher = compile_fp8_paged_mqa_local_topk(
        topk=topk,
        arch=arch,
        preshuffled=preshuffled,
        page_size=page_size,
        rows_per_cta=rows_per_cta,
        prepare_merge=prepare_merge,
        full_split_select=full_split_select,
        skip_compact=SKIP_COMPACT,
        skip_candidate_writes=SKIP_CANDIDATE_WRITES,
        skip_epilogue=SKIP_EPILOGUE,
        xcd_map=XCD_MAP,
    )
    max_pages = block_tables.shape[1]
    if merge_histogram is None:
        merge_histogram = candidate_scores
    if merge_state is None:
        merge_state = candidate_counts
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
        merge_histogram,
        merge_state,
        rows,
        next_n,
        num_splits,
        max_pages,
        stream,
    )
