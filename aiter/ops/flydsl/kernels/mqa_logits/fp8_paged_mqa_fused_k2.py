# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compact τ-filtered harvest and occupancy-bounded fused post-K1 selector."""

from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, rocdl as mlir_rocdl
from flydsl.expr import (
    Array,
    Float32,
    Int32,
    arith,
    as_ir_value,
    gpu,
    range_constexpr,
)
from flydsl.expr.typing import T

from ..candidate_topk_common import (
    NUM_HIST_BINS,
    NUM_WAVES,
    RADIX_SIGN_BIT,
    block_exclusive_prefix_i32,
    f32_to_ordered_i32,
    prefix_radix_mask,
    radix_bucket,
    radix_pass_bits,
)
from ..kernels_common import atomic_add_i32
from ..split_topk_merge_layout import COUNTER_ARRIVALS, COUNTER_PASS_DONE
from ..tensor_shim import GTensor, _run_compiled, _to_raw
from .fp8_paged_mqa_group_harvest import HARVEST_SPLITS
from .fp8_paged_mqa_local_topk import (
    BLOCK_THREADS,
    DREG,
    HEAD_DIM,
    HEADS,
    MFMA_M,
    MFMA_N,
    WAVE_SIZE,
    _load_fp8x32,
    _load_preshuffled_fp8x32,
    _udiv,
    _umod,
)
from .fp8_paged_mqa_score_common import canonical_score_tile

PAGE_SIZE = 64
GROUP_SIZE = 16
_NEUTRAL_E8M0 = 0x7F7F7F7F
_VEC = 4
_NEG_INF = float("-inf")
_POS_INF = float("inf")
GATHER_THREADS = 256


@fx.struct
class CompactTauStorage:
    wave_tau: Array[Float32, 8, 4]


FUSED_COUNTER_GROUP = 32
FUSED_HIST0 = FUSED_COUNTER_GROUP
FUSED_HIST1 = FUSED_HIST0 + NUM_HIST_BINS
FUSED_HIST2 = FUSED_HIST1 + NUM_HIST_BINS
FUSED_STATE = FUSED_HIST2 + NUM_HIST_BINS
FUSED_ROW_STRIDE = FUSED_STATE + 32
FUSED_TAU = 5
FUSED_ABOVE = 6
FUSED_PREFIX = 8
FUSED_MASK = 9
FUSED_REMAINING = 10


def fused_k2_workspace_elems(rows: int) -> int:
    return int(rows) * FUSED_ROW_STRIDE


def fused_k2_parts(rows: int, cu_count: int, occupancy: int = 1) -> int:
    """Largest per-row part count that fits one occupancy envelope."""
    if rows <= 0 or cu_count <= 0:
        return 0
    envelope = cu_count * max(1, occupancy)
    parts = envelope // int(rows)
    if parts < 2:
        return 0
    if rows * parts > envelope:
        return 0
    return int(parts)


def _global_ptr(memref, offset):
    return fx.to_llvm_ptr(fx.get_iter(memref) + offset)


def _atomic_add_ordered(memref, val, offset, ordering):
    val = fx.Int32(val) if isinstance(val, int) else val
    old = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        _global_ptr(memref, offset),
        as_ir_value(val),
        ordering,
        syncscope="agent",
        alignment=4,
    ).result
    return fx.Int32(old)


def _atomic_xchg_ordered(memref, val, offset, ordering):
    val = fx.Int32(val) if isinstance(val, int) else val
    old = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.xchg,
        _global_ptr(memref, offset),
        as_ir_value(val),
        ordering,
        syncscope="agent",
        alignment=4,
    ).result
    return fx.Int32(old)


def _atomic_load_ordered(memref, offset, ordering):
    return fx.Int32(
        llvm.LoadOp(
            T.i32,
            _global_ptr(memref, offset),
            alignment=4,
            volatile_=True,
            ordering=ordering,
            syncscope="agent",
        ).result
    )


def _make_fused_storage(k):
    @fx.struct
    class FusedK2Storage:
        histogram: Array[Int32, NUM_HIST_BINS, 16]
        scan: Array[Int32, NUM_WAVES + 1, 16]
        tau: Array[Float32, 1, 4]
        above_need: Array[Int32, 1, 4]
        equal_need: Array[Int32, 1, 4]
        remaining: Array[Int32, 1, 4]
        prefix: Array[Int32, 1, 4]
        mask: Array[Int32, 1, 4]
        selected_n: Array[Int32, 1, 4]
        above_idxs: Array[Int32, k, 16]
        equal_idxs: Array[Int32, k, 16]
        above_count: Array[Int32, 1, 4]
        equal_count: Array[Int32, 1, 4]
        above_base: Array[Int32, 1, 4]
        equal_base: Array[Int32, 1, 4]

    return FusedK2Storage


@lru_cache(maxsize=8)
def compile_fp8_paged_mqa_compact_harvest(*, arch: str):
    """Rescore selected groups and retain only scores >= the group cutoff τ."""
    if arch != "gfx950":
        raise RuntimeError("compact harvest supports gfx950 only")
    kernel_name = f"fp8_paged_mqa_compact_harvest_h32d128_g16_c8_w8_{arch}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        indices: fx.Tensor,
        selected_group_ids: fx.Tensor,
        selected_group_counts: fx.Tensor,
        group_max: fx.Tensor,
        harvest_scores: fx.Tensor,
        harvest_positions: fx.Tensor,
        harvest_lengths: fx.Tensor,
        rows: fx.Int32,
        request_capacity: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        selected_width: fx.Int32,
        candidate_cap: fx.Int32,
        group_width: fx.Int32,
    ):
        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)
        row = fx.Int32(gpu.block_idx.x)
        split = fx.Int32(gpu.block_idx.y)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(packed_kv, dtype=T.i32, shape=(-1,))
        scale_f32 = GTensor(packed_kv, dtype=T.f32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        row_indices = GTensor(indices, dtype=T.i32, shape=(-1,))
        group_ids = GTensor(selected_group_ids, dtype=T.i32, shape=(-1,))
        group_counts = GTensor(selected_group_counts, dtype=T.i32, shape=(-1,))
        maxima = GTensor(group_max, dtype=T.f32, shape=(-1,))

        row_ok = row < rows
        safe_row = row_ok.select(row, fx.Int32(0))
        request = fx.Int32(row_indices[safe_row])
        request_ok = (request >= 0) & (request < request_capacity)
        safe_request = request_ok.select(request, fx.Int32(0))
        table_span = max_pages * fx.Int32(PAGE_SIZE)
        length = fx.Int32(lengths[safe_row])
        length = (length < 0).select(fx.Int32(0), length)
        length = (length < table_span).select(length, table_span)
        count = fx.Int32(group_counts[safe_row])
        count = (count < 0).select(fx.Int32(0), count)
        count = (count < selected_width).select(count, selected_width)

        tau = fx.Float32(_POS_INF)
        for idx in range(tid, selected_width, fx.Int32(BLOCK_THREADS)):
            live = row_ok & (idx < count)
            safe_idx = live.select(idx, fx.Int32(0))
            group = fx.Int32(group_ids[safe_row * selected_width + safe_idx])
            group_ok = live & (group >= 0)
            safe_group = group_ok.select(group, fx.Int32(0))
            gmax = fx.Float32(maxima[safe_row * group_width + safe_group])
            tau = group_ok.select(gmax.minimumf(tau), tau)
        for xor_off in (1, 2, 4, 8, 16, 32):
            tau = tau.minimumf(tau.shuffle_xor(xor_off, WAVE_SIZE))
        tau_storage = fx.SharedAllocator().allocate(CompactTauStorage)
        wave_tau = tau_storage.wave_tau.peek().view(fx.make_layout(8, 1))
        if lane == 0:
            wave_tau[wave] = tau
        gpu.barrier()
        if tid == 0:
            block_tau = wave_tau[0]
            for wave_idx in range_constexpr(7):
                block_tau = block_tau.minimumf(wave_tau[fx.Int32(wave_idx + 1)])
            if count == 0:
                block_tau = fx.Float32(_NEG_INF)
            wave_tau[0] = block_tau
        gpu.barrier()
        tau = wave_tau[0]

        q_tiles = [None] * (HEADS // MFMA_M)
        weight_frags = [[None] * DREG for _ in range(HEADS // MFMA_M)]
        q_row_bytes = safe_row * fx.Int32(HEADS * HEAD_DIM)
        for mi in range_constexpr(HEADS // MFMA_M):
            q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
            q_tiles[mi] = _load_fp8x32(
                q_i32,
                q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                lane_div_16,
            )
            for ii in range_constexpr(DREG):
                head = mi * MFMA_M + lane_div_16 * DREG + ii
                weight_frags[mi][ii] = fx.Float32(weight_t[safe_row, head])

        ordinal_begin = _udiv(selected_width * split, HARVEST_SPLITS)
        ordinal_end = _udiv(selected_width * (split + fx.Int32(1)), HARVEST_SPLITS)
        for ordinal_base in range(
            ordinal_begin,
            ordinal_end,
            fx.Int32(HARVEST_SPLITS),
        ):
            ordinal = ordinal_base + wave
            ordinal_ok = row_ok & (ordinal < ordinal_end) & (ordinal < count)
            safe_ordinal = ordinal_ok.select(ordinal, fx.Int32(0))
            group = fx.Int32(group_ids[safe_row * selected_width + safe_ordinal])
            group_ok = ordinal_ok & (group >= 0)
            safe_group = group_ok.select(group, fx.Int32(0))
            logical = safe_group * fx.Int32(GROUP_SIZE) + lane_mod_16
            logical_page = safe_group >> fx.Int32(2)
            table_ok = request_ok & group_ok & (logical_page < max_pages)
            table_offset = safe_request * max_pages + table_ok.select(
                logical_page, fx.Int32(0)
            )
            physical_page = fx.Int32(tables[table_offset])
            page_ok = table_ok & (physical_page >= 0) & (physical_page < num_pages)
            safe_page = page_ok.select(physical_page, fx.Int32(0))
            page_i32 = (safe_page << 11) + (safe_page << 6)
            token_in_page = (safe_group & fx.Int32(3)) * fx.Int32(GROUP_SIZE)
            token_in_page = token_in_page + lane_mod_16
            k_pack = _load_preshuffled_fp8x32(
                kv_i32,
                page_i32,
                token_in_page,
                lane_div_16,
            )
            scale = fx.Float32(
                scale_f32[
                    page_i32 + fx.Int32(PAGE_SIZE * HEAD_DIM // 4) + token_in_page
                ]
            )
            position_ok = table_ok & (logical < length)
            score = canonical_score_tile(
                q_tiles,
                weight_frags,
                k_pack,
                scale,
                page_ok & position_ok,
                result_type,
                neutral,
            )
            full_groups = _udiv(length + fx.Int32(GROUP_SIZE - 1), fx.Int32(GROUP_SIZE))
            keep_all = count == full_groups
            keep = (
                (lane_div_16 == 0)
                & (ordinal < ordinal_end)
                & position_ok
                & (keep_all | (score >= tau))
            )
            if keep:
                slot = atomic_add_i32(harvest_lengths, 1, safe_row, "agent")
                in_range = slot < candidate_cap
                safe_slot = in_range.select(slot, fx.Int32(0))
                output = safe_row * candidate_cap + safe_slot
                if in_range:
                    fx.ptr_store(
                        score, fx.add_offset(fx.get_iter(harvest_scores), output)
                    )
                    fx.ptr_store(
                        logical,
                        fx.add_offset(fx.get_iter(harvest_positions), output),
                    )

    @flyc.jit
    def launch(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        indices: fx.Tensor,
        selected_group_ids: fx.Tensor,
        selected_group_counts: fx.Tensor,
        group_max: fx.Tensor,
        harvest_scores: fx.Tensor,
        harvest_positions: fx.Tensor,
        harvest_lengths: fx.Tensor,
        rows: fx.Int32,
        request_capacity: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        selected_width: fx.Int32,
        candidate_cap: fx.Int32,
        group_width: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(rows))
        kernel(
            q_fp8,
            packed_kv,
            weights,
            context_lens,
            block_tables,
            indices,
            selected_group_ids,
            selected_group_counts,
            group_max,
            harvest_scores,
            harvest_positions,
            harvest_lengths,
            rows,
            request_capacity,
            max_pages,
            num_pages,
            selected_width,
            candidate_cap,
            group_width,
        ).launch(
            grid=(gx, HARVEST_SPLITS, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": True}
    return launch


def launch_fp8_paged_mqa_compact_harvest(
    q_fp8,
    packed_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    selected_group_ids,
    selected_group_counts,
    group_max,
    harvest_scores,
    harvest_positions,
    harvest_lengths,
    *,
    arch,
    stream,
):
    launcher = compile_fp8_paged_mqa_compact_harvest(arch=arch)
    harvest_lengths.zero_()
    _run_compiled(
        launcher,
        q_fp8,
        packed_kv,
        weights,
        context_lens,
        block_tables,
        indices,
        selected_group_ids,
        selected_group_counts,
        group_max,
        harvest_scores,
        harvest_positions,
        harvest_lengths,
        q_fp8.shape[0],
        block_tables.shape[0],
        block_tables.shape[1],
        packed_kv.shape[0],
        selected_group_ids.shape[1],
        harvest_scores.shape[1],
        group_max.shape[1],
        stream,
    )


@lru_cache(maxsize=4)
def compile_fp8_paged_mqa_gather_positions(*, arch: str):
    kernel_name = f"fp8_paged_mqa_gather_positions_{arch}"

    @flyc.kernel(name=kernel_name, known_block_size=[GATHER_THREADS, 1, 1])
    def kernel(
        compact_positions: fx.Tensor,
        ordinals: fx.Tensor,
        compact_counts: fx.Tensor,
        output_positions: fx.Tensor,
        rows: fx.Int32,
        compact_width: fx.Int32,
        output_width: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        row = fx.Int32(gpu.block_idx.x)
        compact = GTensor(compact_positions, dtype=T.i32, shape=(-1,))
        ords = GTensor(ordinals, dtype=T.i32, shape=(-1,))
        counts = GTensor(compact_counts, dtype=T.i32, shape=(-1,))
        row_ok = row < rows
        safe_row = row_ok.select(row, fx.Int32(0))
        count = fx.Int32(counts[safe_row])
        count = (count < 0).select(fx.Int32(0), count)
        count = (count < compact_width).select(count, compact_width)
        for column in range(tid, output_width, fx.Int32(GATHER_THREADS)):
            ordinal = fx.Int32(ords[safe_row * output_width + column])
            live = row_ok & (ordinal >= 0) & (ordinal < count)
            safe_ord = live.select(ordinal, fx.Int32(0))
            position = fx.Int32(compact[safe_row * compact_width + safe_ord])
            fx.ptr_store(
                live.select(position, fx.Int32(-1)),
                fx.add_offset(
                    fx.get_iter(output_positions),
                    row * output_width + column,
                ),
            )

    @flyc.jit
    def launch(
        compact_positions: fx.Tensor,
        ordinals: fx.Tensor,
        compact_counts: fx.Tensor,
        output_positions: fx.Tensor,
        rows: fx.Int32,
        compact_width: fx.Int32,
        output_width: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(rows))
        kernel(
            compact_positions,
            ordinals,
            compact_counts,
            output_positions,
            rows,
            compact_width,
            output_width,
        ).launch(
            grid=(gx, 1, 1),
            block=(GATHER_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"fast_fp_math": True}
    return launch


def launch_fp8_paged_mqa_gather_positions(
    compact_positions,
    ordinals,
    compact_counts,
    output_positions,
    *,
    arch,
    stream,
):
    launcher = compile_fp8_paged_mqa_gather_positions(arch=arch)
    _run_compiled(
        launcher,
        compact_positions,
        ordinals,
        compact_counts,
        output_positions,
        ordinals.shape[0],
        compact_positions.shape[1],
        ordinals.shape[1],
        stream,
    )


@lru_cache(maxsize=8)
def compile_fp8_paged_mqa_fused_k2(*, arch: str, k: int, parts: int):
    """One occupancy-bounded kernel: group threshold, compact rescore, item TopK."""
    if arch != "gfx950":
        raise RuntimeError("fused K2 supports gfx950 only")
    if k % _VEC:
        raise ValueError("k must be divisible by 4")
    if parts < 2:
        raise ValueError("fused K2 requires at least 2 parts")
    kernel_name = f"fp8_paged_mqa_fused_k2_h32d128_k{k}_p{parts}_w8_v2_{arch}"
    hist_steps = NUM_HIST_BINS // BLOCK_THREADS
    output_steps = (k + BLOCK_THREADS - 1) // BLOCK_THREADS
    storage_type = _make_fused_storage(k)
    hist_offs = (FUSED_HIST0, FUSED_HIST1, FUSED_HIST2)

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        indices: fx.Tensor,
        group_max: fx.Tensor,
        group_ends: fx.Tensor,
        workspace: fx.Tensor,
        out_scores: fx.Tensor,
        out_positions: fx.Tensor,
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        rows: fx.Int32,
        request_capacity: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_width: fx.Int32,
        candidate_cap: fx.Int32,
        k_i32: fx.Int32,
    ):
        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)
        part = fx.Int32(gpu.block_idx.x)
        row = fx.Int32(gpu.block_idx.y)
        parts_i32 = fx.Int32(parts)
        row_base = row * fx.Int32(FUSED_ROW_STRIDE)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(packed_kv, dtype=T.i32, shape=(-1,))
        scale_f32 = GTensor(packed_kv, dtype=T.f32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        row_indices = GTensor(indices, dtype=T.i32, shape=(-1,))
        maxima = GTensor(group_max, dtype=T.f32, shape=(-1,))

        storage = fx.SharedAllocator().allocate(storage_type)
        histogram = storage.histogram.peek().view(fx.make_layout(NUM_HIST_BINS, 1))
        scan = storage.scan.peek().view(fx.make_layout(NUM_WAVES + 1, 1))
        s_tau = storage.tau.peek().view(fx.make_layout(1, 1))
        s_remaining = storage.remaining.peek().view(fx.make_layout(1, 1))
        s_prefix = storage.prefix.peek().view(fx.make_layout(1, 1))
        s_mask = storage.mask.peek().view(fx.make_layout(1, 1))
        s_selected = storage.selected_n.peek().view(fx.make_layout(1, 1))
        s_above_idxs = storage.above_idxs.peek().view(fx.make_layout(k, 1))
        s_equal_idxs = storage.equal_idxs.peek().view(fx.make_layout(k, 1))
        s_above_count = storage.above_count.peek().view(fx.make_layout(1, 1))
        s_equal_count = storage.equal_count.peek().view(fx.make_layout(1, 1))
        s_above_base = storage.above_base.peek().view(fx.make_layout(1, 1))
        s_equal_base = storage.equal_base.peek().view(fx.make_layout(1, 1))
        s_above_need = storage.above_need.peek().view(fx.make_layout(1, 1))
        s_equal_need = storage.equal_need.peek().view(fx.make_layout(1, 1))

        row_ok = row < rows
        safe_row = row_ok.select(row, fx.Int32(0))
        request = fx.Int32(row_indices[safe_row])
        request_ok = (request >= 0) & (request < request_capacity)
        safe_request = request_ok.select(request, fx.Int32(0))
        table_span = max_pages * fx.Int32(PAGE_SIZE)
        length = fx.Int32(lengths[safe_row])
        length = (length < 0).select(fx.Int32(0), length)
        length = (length < table_span).select(length, table_span)
        row_groups = fx.Int32(group_ends[safe_row])
        row_groups = (row_groups < 0).select(fx.Int32(0), row_groups)
        row_groups = (row_groups < group_width).select(row_groups, group_width)
        selected_n = (row_groups < k_i32).select(row_groups, k_i32)
        if tid == 0:
            s_selected[0] = selected_n
        gpu.barrier()
        selected_n = s_selected[0]

        def spin_until_ge(offset, target):
            cur = _atomic_load_ordered(
                workspace, offset, llvm.AtomicOrdering.monotonic
            )
            while cur < target:
                mlir_rocdl.s_sleep(1)
                cur = _atomic_load_ordered(
                    workspace, offset, llvm.AtomicOrdering.monotonic
                )
            _atomic_load_ordered(workspace, offset, llvm.AtomicOrdering.acquire)

        def row_barrier(token):
            gpu.barrier()

            @flyc.jit
            def _arrive(
                _pred=(tid == 0) & row_ok,
                _token=fx.Int32(token),
                _workspace=workspace,
                _row_base=row_base,
                _parts=parts_i32,
            ):
                if _pred:
                    arrivals = _row_base + fx.Int32(COUNTER_ARRIVALS)
                    done = _row_base + fx.Int32(COUNTER_PASS_DONE)
                    prev = _atomic_add_ordered(
                        _workspace, 1, arrivals, llvm.AtomicOrdering.acq_rel
                    )
                    last = (prev + fx.Int32(1)) == _token * _parts
                    if last:
                        _atomic_xchg_ordered(
                            _workspace,
                            _token,
                            done,
                            llvm.AtomicOrdering.release,
                        )
                    else:
                        spin_until_ge(done, _token)

            _arrive()
            gpu.barrier()

        def clear_histogram(num_bins):
            steps = num_bins // BLOCK_THREADS
            for step in range_constexpr(steps):
                histogram[fx.Int32(step * BLOCK_THREADS) + tid] = 0
            gpu.barrier()

        def flush_histogram(hist_off, num_bins):
            steps = num_bins // BLOCK_THREADS
            for step in range_constexpr(steps):
                hist_bin = fx.Int32(step * BLOCK_THREADS) + tid
                count = histogram[hist_bin]

                @flyc.jit
                def _flush(
                    _pred=(count != 0) & (hist_bin < fx.Int32(num_bins)),
                    _workspace=workspace,
                    _offset=row_base + fx.Int32(hist_off) + hist_bin,
                    _count=count,
                ):
                    if _pred:
                        atomic_add_i32(_workspace, _count, _offset, "agent")

                _flush()
            gpu.barrier()

        def copy_hist_from_workspace(hist_off, num_bins):
            steps = num_bins // BLOCK_THREADS
            for step in range_constexpr(steps):
                hist_bin = fx.Int32(step * BLOCK_THREADS) + tid
                histogram[hist_bin] = workspace[
                    row_base + fx.Int32(hist_off) + hist_bin
                ]
            gpu.barrier()

        def select_from_histogram(pass_index, prefix, mask, remaining):
            num_bins = 1 << radix_pass_bits(pass_index)
            bins_per_thread = num_bins // BLOCK_THREADS
            xor_value = RADIX_SIGN_BIT if pass_index == 0 else 0
            selected_high = fx.Int32(num_bins - 1) - tid * fx.Int32(bins_per_thread)
            group_count = fx.Int32(0)
            bin_counts = fx.make_rmem_tensor(bins_per_thread, Int32)
            for bin_item in range_constexpr(bins_per_thread):
                value = histogram[selected_high - fx.Int32(bin_item)]
                bin_counts[bin_item] = value
                group_count = group_count + value
            before_group, _ = block_exclusive_prefix_i32(tid, group_count, scan)
            before_bin = before_group
            for bin_item in range_constexpr(bins_per_thread):
                selected = selected_high - fx.Int32(bin_item)
                bin_count = bin_counts[bin_item]

                @flyc.jit
                def _maybe_select(
                    _pred=(before_bin < remaining)
                    & (before_bin + bin_count >= remaining),
                    _selected=selected,
                    _xor=fx.Int32(xor_value),
                    _prefix=prefix,
                    _mask=mask,
                    _remaining=remaining,
                    _before=before_bin,
                    _s_prefix=s_prefix,
                    _s_mask=s_mask,
                    _s_remaining=s_remaining,
                    _pass=pass_index,
                ):
                    if _pred:
                        actual = _selected ^ _xor
                        pass_mask, shift = prefix_radix_mask(_pass)
                        _s_prefix[0] = _prefix | (actual << fx.Int32(shift))
                        _s_mask[0] = _mask | pass_mask
                        _s_remaining[0] = _remaining - _before

                _maybe_select()
                before_bin = before_bin + bin_count
            gpu.barrier()
            return s_prefix[0], s_mask[0], s_remaining[0]

        group_begin = _udiv(group_width * part, parts_i32)
        group_end = _udiv(group_width * (part + fx.Int32(1)), parts_i32)

        if selected_n == 0:
            for step in range_constexpr(output_steps):
                slot = fx.Int32(step * BLOCK_THREADS) + tid
                if (part == 0) & (slot < k_i32):
                    out_scores[safe_row, slot] = fx.Float32(_NEG_INF)
                    out_positions[safe_row, slot] = fx.Int32(-1)
            return

        prefix = fx.Int32(0)
        mask = fx.Int32(0)
        remaining = selected_n
        if tid == 0:
            s_prefix[0] = 0
            s_mask[0] = 0
            s_remaining[0] = remaining
        gpu.barrier()

        for pass_index in range_constexpr(3):
            num_bins = 1 << radix_pass_bits(pass_index)
            clear_histogram(num_bins)
            xor_value = RADIX_SIGN_BIT if pass_index == 0 else 0
            for group in range(group_begin + tid, group_end, fx.Int32(BLOCK_THREADS)):
                live = row_ok & (group < row_groups)
                safe_group = live.select(group, fx.Int32(0))
                value = fx.Float32(maxima[safe_row * group_width + safe_group])
                ords = f32_to_ordered_i32(value)
                matched = live & ((ords & mask) == prefix)

                @flyc.jit
                def _hist(
                    _pred=matched,
                    _histogram=histogram,
                    _ords=ords,
                    _pass=pass_index,
                    _xor=fx.Int32(xor_value),
                ):
                    if _pred:
                        bucket = radix_bucket(_ords, _pass) ^ fx.Int32(_xor)
                        atomic_add_i32(_histogram, 1, bucket, "workgroup")

                _hist()
            gpu.barrier()
            flush_histogram(hist_offs[pass_index], num_bins)
            row_barrier(pass_index + 1)
            copy_hist_from_workspace(hist_offs[pass_index], num_bins)
            prefix, mask, remaining = select_from_histogram(
                pass_index, prefix, mask, remaining
            )
            if tid == 0:
                s_prefix[0] = prefix
                s_mask[0] = mask
                s_remaining[0] = remaining
            gpu.barrier()
            prefix = s_prefix[0]
            mask = s_mask[0]
            remaining = s_remaining[0]

        threshold = prefix
        if tid == 0:
            s_above_count[0] = 0
            s_equal_count[0] = 0
        gpu.barrier()
        for group in range(group_begin + tid, group_end, fx.Int32(BLOCK_THREADS)):
            live = row_ok & (group < row_groups)
            safe_group = live.select(group, fx.Int32(0))
            value = fx.Float32(maxima[safe_row * group_width + safe_group])
            ords = f32_to_ordered_i32(value)
            above = live & (ords > threshold)
            equal = live & (ords == threshold)

            @flyc.jit
            def _claim_above(
                _pred=above,
                _s_above=s_above_count,
                _s_idx=s_above_idxs,
                _group=safe_group,
                _k=k_i32,
            ):
                if _pred:
                    slot = atomic_add_i32(_s_above, 1, 0, "workgroup")
                    if slot < _k:
                        _s_idx[slot] = _group

            _claim_above()

            @flyc.jit
            def _claim_equal(
                _pred=equal,
                _s_equal=s_equal_count,
                _s_idx=s_equal_idxs,
                _group=safe_group,
                _k=k_i32,
            ):
                if _pred:
                    slot = atomic_add_i32(_s_equal, 1, 0, "workgroup")
                    if slot < _k:
                        _s_idx[slot] = _group

            _claim_equal()
        gpu.barrier()
        row_barrier(4)
        id_counter = row_base + fx.Int32(16)
        compact_counter = row_base + fx.Int32(17)
        above_local = s_above_count[0]
        for local in range(tid, k, BLOCK_THREADS):
            live = local < above_local
            if live:
                gid = s_above_idxs[local]
                out_slot = atomic_add_i32(workspace, 1, id_counter, "agent")
                if out_slot < k_i32:
                    candidate_positions[safe_row, out_slot] = gid
        gpu.barrier()
        row_barrier(5)
        already = _atomic_load_ordered(
            workspace, id_counter, llvm.AtomicOrdering.acquire
        )
        equal_local = s_equal_count[0]
        for local in range(tid, k, BLOCK_THREADS):
            live = (local < equal_local) & (already < selected_n)
            if live:
                gid = s_equal_idxs[local]
                out_slot = atomic_add_i32(workspace, 1, id_counter, "agent")
                if out_slot < selected_n:
                    candidate_positions[safe_row, out_slot] = gid
        row_barrier(6)

        local_tau = fx.Float32(_POS_INF)
        for idx in range(tid, selected_n, fx.Int32(BLOCK_THREADS)):
            gid = candidate_positions[safe_row, idx]
            gid_ok = gid >= 0
            safe_gid = gid_ok.select(gid, fx.Int32(0))
            gmax = fx.Float32(maxima[safe_row * group_width + safe_gid])
            local_tau = gid_ok.select(gmax.minimumf(local_tau), local_tau)
        for xor_off in (1, 2, 4, 8, 16, 32):
            local_tau = local_tau.minimumf(local_tau.shuffle_xor(xor_off, WAVE_SIZE))
        if lane == 0:
            histogram[wave] = local_tau.bitcast(fx.Int32)
        gpu.barrier()
        if (part == 0) & (tid == 0) & row_ok:
            block_tau = histogram[0].bitcast(fx.Float32)
            for wave_idx in range_constexpr(7):
                block_tau = block_tau.minimumf(
                    histogram[fx.Int32(wave_idx + 1)].bitcast(fx.Float32)
                )
            if selected_n == 0:
                block_tau = fx.Float32(_NEG_INF)
            s_tau[0] = block_tau
            workspace[row_base + fx.Int32(FUSED_TAU)] = block_tau.bitcast(fx.Int32)
        gpu.barrier()
        for idx in range(tid, selected_n, fx.Int32(BLOCK_THREADS)):
            out_positions[safe_row, idx] = candidate_positions[safe_row, idx]
        row_barrier(7)
        tau = workspace[row_base + fx.Int32(FUSED_TAU)].bitcast(fx.Float32)

        q_tiles = [None] * (HEADS // MFMA_M)
        weight_frags = [[None] * DREG for _ in range(HEADS // MFMA_M)]
        q_row_bytes = safe_row * fx.Int32(HEADS * HEAD_DIM)
        for mi in range_constexpr(HEADS // MFMA_M):
            q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
            q_tiles[mi] = _load_fp8x32(
                q_i32,
                q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                lane_div_16,
            )
            for ii in range_constexpr(DREG):
                head = mi * MFMA_M + lane_div_16 * DREG + ii
                weight_frags[mi][ii] = fx.Float32(weight_t[safe_row, head])

        for ordinal_base in range(
            part * fx.Int32(NUM_WAVES),
            selected_n,
            parts_i32 * fx.Int32(NUM_WAVES),
        ):
            ordinal = ordinal_base + wave
            ordinal_ok = row_ok & (ordinal < selected_n)
            safe_ordinal = ordinal_ok.select(ordinal, fx.Int32(0))
            group = out_positions[safe_row, safe_ordinal]
            group_ok = ordinal_ok & (group >= 0)
            safe_group = group_ok.select(group, fx.Int32(0))
            logical = safe_group * fx.Int32(GROUP_SIZE) + lane_mod_16
            logical_page = safe_group >> fx.Int32(2)
            table_ok = request_ok & group_ok & (logical_page < max_pages)
            table_offset = safe_request * max_pages + table_ok.select(
                logical_page, fx.Int32(0)
            )
            physical_page = fx.Int32(tables[table_offset])
            page_ok = table_ok & (physical_page >= 0) & (physical_page < num_pages)
            safe_page = page_ok.select(physical_page, fx.Int32(0))
            page_i32 = (safe_page << 11) + (safe_page << 6)
            token_in_page = (safe_group & fx.Int32(3)) * fx.Int32(GROUP_SIZE)
            token_in_page = token_in_page + lane_mod_16
            k_pack = _load_preshuffled_fp8x32(
                kv_i32,
                page_i32,
                token_in_page,
                lane_div_16,
            )
            scale = fx.Float32(
                scale_f32[
                    page_i32 + fx.Int32(PAGE_SIZE * HEAD_DIM // 4) + token_in_page
                ]
            )
            position_ok = table_ok & (logical < length)
            score = canonical_score_tile(
                q_tiles,
                weight_frags,
                k_pack,
                scale,
                page_ok & position_ok,
                result_type,
                neutral,
            )
            keep_all = selected_n == row_groups
            keep = (
                (lane_div_16 == 0)
                & ordinal_ok
                & position_ok
                & (keep_all | (score >= tau))
            )
            if keep:
                compact_counter = row_base + fx.Int32(17)
                slot = atomic_add_i32(workspace, 1, compact_counter, "agent")
                in_range = slot < candidate_cap
                safe_slot = in_range.select(slot, fx.Int32(0))
                if in_range:
                    candidate_scores[safe_row, safe_slot] = score
                    candidate_positions[safe_row, safe_slot] = logical
        row_barrier(8)

        compact_n = _atomic_load_ordered(
            workspace,
            row_base + fx.Int32(17),
            llvm.AtomicOrdering.acquire,
        )
        compact_n = (compact_n < candidate_cap).select(compact_n, candidate_cap)
        emit_n = (compact_n < k_i32).select(compact_n, k_i32)
        if compact_n <= k_i32:
            for step in range_constexpr(output_steps):
                slot = fx.Int32(step * BLOCK_THREADS) + tid
                if (part == 0) & (slot < k_i32):
                    live = slot < compact_n
                    safe_slot = live.select(slot, fx.Int32(0))
                    out_scores[safe_row, slot] = live.select(
                        candidate_scores[safe_row, safe_slot],
                        fx.Float32(_NEG_INF),
                    )
                    out_positions[safe_row, slot] = live.select(
                        candidate_positions[safe_row, safe_slot],
                        fx.Int32(-1),
                    )
            return

        # Item radix over the compact bag, same occupancy-bounded parts.
        prefix = fx.Int32(0)
        mask = fx.Int32(0)
        remaining = k_i32
        if tid == 0:
            s_prefix[0] = 0
            s_mask[0] = 0
            s_remaining[0] = remaining
        gpu.barrier()
        cand_begin = _udiv(compact_n * part, parts_i32)
        cand_end = _udiv(compact_n * (part + fx.Int32(1)), parts_i32)
        for pass_index in range_constexpr(3):
            num_bins = 1 << radix_pass_bits(pass_index)
            clear_histogram(num_bins)
            xor_value = RADIX_SIGN_BIT if pass_index == 0 else 0
            for idx in range(cand_begin + tid, cand_end, fx.Int32(BLOCK_THREADS)):
                live = row_ok & (idx < compact_n)
                safe_idx = live.select(idx, fx.Int32(0))
                value = candidate_scores[safe_row, safe_idx]
                ords = f32_to_ordered_i32(value)
                matched = live & ((ords & mask) == prefix)

                @flyc.jit
                def _item_hist(
                    _pred=matched,
                    _histogram=histogram,
                    _ords=ords,
                    _pass=pass_index,
                    _xor=fx.Int32(xor_value),
                ):
                    if _pred:
                        bucket = radix_bucket(_ords, _pass) ^ fx.Int32(_xor)
                        atomic_add_i32(_histogram, 1, bucket, "workgroup")

                _item_hist()
            gpu.barrier()
            flush_histogram(hist_offs[pass_index], num_bins)
            row_barrier(9 + pass_index)
            copy_hist_from_workspace(hist_offs[pass_index], num_bins)
            prefix, mask, remaining = select_from_histogram(
                pass_index, prefix, mask, remaining
            )
            if tid == 0:
                s_prefix[0] = prefix
                s_mask[0] = mask
                s_remaining[0] = remaining
            gpu.barrier()
            prefix = s_prefix[0]
            mask = s_mask[0]
            remaining = s_remaining[0]

        item_threshold = prefix
        if tid == 0:
            s_above_count[0] = 0
            s_equal_count[0] = 0
        gpu.barrier()
        out_counter = row_base + fx.Int32(20)
        for idx in range(cand_begin + tid, cand_end, fx.Int32(BLOCK_THREADS)):
            live = row_ok & (idx < compact_n)
            safe_idx = live.select(idx, fx.Int32(0))
            value = candidate_scores[safe_row, safe_idx]
            ords = f32_to_ordered_i32(value)
            above = live & (ords > item_threshold)
            equal = live & (ords == item_threshold)
            if above:
                slot = atomic_add_i32(workspace, 1, out_counter, "agent")
                if slot < k_i32:
                    out_scores[safe_row, slot] = value
                    out_positions[safe_row, slot] = candidate_positions[
                        safe_row, safe_idx
                    ]
        row_barrier(12)
        already = _atomic_load_ordered(
            workspace, out_counter, llvm.AtomicOrdering.acquire
        )
        for idx in range(cand_begin + tid, cand_end, fx.Int32(BLOCK_THREADS)):
            live = row_ok & (idx < compact_n) & (already < k_i32)
            safe_idx = live.select(idx, fx.Int32(0))
            value = candidate_scores[safe_row, safe_idx]
            ords = f32_to_ordered_i32(value)
            equal = live & (ords == item_threshold)
            if equal:
                slot = atomic_add_i32(workspace, 1, out_counter, "agent")
                if slot < k_i32:
                    out_scores[safe_row, slot] = value
                    out_positions[safe_row, slot] = candidate_positions[
                        safe_row, safe_idx
                    ]
        row_barrier(13)
        for step in range_constexpr(output_steps):
            slot = fx.Int32(step * BLOCK_THREADS) + tid
            if (part == 0) & (slot < k_i32) & (slot >= already):
                filled = _atomic_load_ordered(
                    workspace, out_counter, llvm.AtomicOrdering.acquire
                )
                if slot >= filled:
                    out_scores[safe_row, slot] = fx.Float32(_NEG_INF)
                    out_positions[safe_row, slot] = fx.Int32(-1)

    @flyc.jit
    def launch(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        indices: fx.Tensor,
        group_max: fx.Tensor,
        group_ends: fx.Tensor,
        workspace: fx.Tensor,
        out_scores: fx.Tensor,
        out_positions: fx.Tensor,
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        rows: fx.Int32,
        request_capacity: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_width: fx.Int32,
        candidate_cap: fx.Int32,
        k_i32: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, fx.Int32(parts))
        gy = arith.index_cast(T.index, _to_raw(rows))
        kernel(
            q_fp8,
            packed_kv,
            weights,
            context_lens,
            block_tables,
            indices,
            group_max,
            group_ends,
            workspace,
            out_scores,
            out_positions,
            candidate_scores,
            candidate_positions,
            rows,
            request_capacity,
            max_pages,
            num_pages,
            group_width,
            candidate_cap,
            k_i32,
        ).launch(
            grid=(gx, gy, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": True}
    return launch


def launch_fp8_paged_mqa_fused_k2(
    q_fp8,
    packed_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    group_max,
    group_ends,
    workspace,
    out_scores,
    out_positions,
    candidate_scores,
    candidate_positions,
    *,
    k,
    parts,
    arch,
    stream,
):
    workspace.zero_()
    candidate_positions.fill_(-1)
    candidate_scores.fill_(_NEG_INF)
    launcher = compile_fp8_paged_mqa_fused_k2(arch=arch, k=int(k), parts=int(parts))
    _run_compiled(
        launcher,
        q_fp8,
        packed_kv,
        weights,
        context_lens,
        block_tables,
        indices,
        group_max,
        group_ends,
        workspace,
        out_scores,
        out_positions,
        candidate_scores,
        candidate_positions,
        q_fp8.shape[0],
        block_tables.shape[0],
        block_tables.shape[1],
        packed_kv.shape[0],
        group_max.shape[1],
        candidate_scores.shape[1],
        k,
        stream,
    )
