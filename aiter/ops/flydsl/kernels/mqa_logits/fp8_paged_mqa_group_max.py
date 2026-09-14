# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K-stationary group-max pass for packed-varlen paged FP8 MQA."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..tensor_shim import GTensor, _run_compiled, _to_raw
from .fp8_paged_mqa_local_topk import (
    BLOCK_N,
    BLOCK_THREADS,
    DREG,
    HEAD_DIM,
    HEADS,
    MFMA_M,
    MFMA_N,
    N_TILES_PER_WAVE,
    WAVE_SIZE,
    _concat_i32x4,
    _imin,
    _load_fp8x32,
    _load_preshuffled_fp8x32,
    _udiv,
    _umod,
)
from .fp8_paged_mqa_score_common import canonical_score_tile

ROWS_PER_CTA = 2
SUPPORTED_ROWS_PER_CTA = (2, 4)
PAGE_SIZE = 64
GROUP_SIZE = 16
_NEUTRAL_E8M0 = 0x7F7F7F7F
_WEIGHT_LDS_ELEMS = 4 * HEADS
_Q_I32_ELEMS = 4 * HEADS * HEAD_DIM // 4
_Q_I32_PER_ROW = HEADS * HEAD_DIM // 4


def _load_fp8x32_lds(q_lds, byte_base, lane_div_16):
    off = _udiv(byte_base + lane_div_16 * fx.Int32(32), 4)
    elems = [q_lds[off + fx.Int32(i)] for i in range_constexpr(8)]
    return _concat_i32x4(
        fx.Vector.from_elements(elems[:4]),
        fx.Vector.from_elements(elems[4:]),
    )


def _load_q_tiles_from_lds(q_lds, row_index, lane_mod_16, lane_div_16):
    tiles = [None] * (HEADS // MFMA_M)
    row_bytes = fx.Int32(row_index) * fx.Int32(HEADS * HEAD_DIM)
    for mi in range_constexpr(HEADS // MFMA_M):
        q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
        tiles[mi] = _load_fp8x32_lds(
            q_lds,
            row_bytes + q_head * fx.Int32(HEAD_DIM),
            lane_div_16,
        )
    return tiles


def _load_weight_frags_from_lds(weight_lds, row_index, lane_div_16):
    weight_frags = [[None] * DREG for _ in range(HEADS // MFMA_M)]
    row_base = fx.Int32(row_index) * fx.Int32(HEADS)
    for mi in range_constexpr(HEADS // MFMA_M):
        for ii in range_constexpr(DREG):
            head = mi * MFMA_M + lane_div_16 * DREG + ii
            weight_frags[mi][ii] = fx.Float32(weight_lds[row_base + fx.Int32(head)])
    return weight_frags


def fp8_paged_mqa_group_max_kernel_name(
    *, arch: str, rows_per_cta: int = ROWS_PER_CTA
) -> str:
    return (
        "fp8_paged_mqa_group_max_h32d128_"
        f"r{rows_per_cta}_g{GROUP_SIZE}_w8_bn{BLOCK_N}_{arch}"
    )


def _bind_launch(kernel):
    @flyc.jit
    def launch(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        decode_lens: fx.Tensor,
        block_tables: fx.Tensor,
        group_max: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(num_splits))
        gy = arith.index_cast(
            T.index,
            _to_raw(request_capacity * row_groups_per_request),
        )
        kernel(
            q_fp8,
            packed_kv,
            weights,
            context_lens,
            query_start_loc,
            decode_lens,
            block_tables,
            group_max,
            rows_cap,
            request_capacity,
            row_groups_per_request,
            num_splits,
            max_pages,
            num_pages,
            group_stride,
        ).launch(grid=(gx, gy, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": True}
    return launch


def _compile_r2(arch: str):
    rows = 2
    kernel_name = fp8_paged_mqa_group_max_kernel_name(arch=arch, rows_per_cta=rows)

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        decode_lens: fx.Tensor,
        block_tables: fx.Tensor,
        group_max: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
    ):
        del rows_cap
        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)
        split = fx.Int32(gpu.block_idx.x)
        owner = fx.Int32(gpu.block_idx.y)
        request = _udiv(owner, row_groups_per_request)
        row_group = _umod(owner, row_groups_per_request)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(packed_kv, dtype=T.i32, shape=(-1,))
        scale_f32 = GTensor(packed_kv, dtype=T.f32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        starts = GTensor(query_start_loc, dtype=T.i32, shape=(-1,))
        widths = GTensor(decode_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))

        request_ok = request < request_capacity
        safe_request = request_ok.select(request, fx.Int32(0))
        row0 = fx.Int32(starts[safe_request]) + row_group * fx.Int32(rows)
        remaining = fx.Int32(widths[safe_request]) - row_group * fx.Int32(rows)
        live = _imin(
            (remaining < 0).select(fx.Int32(0), remaining),
            fx.Int32(rows),
        )
        live = request_ok.select(live, fx.Int32(0))

        if live != 0:
            row_ids = [None] * rows
            row_lengths = [None] * rows
            q_tiles = [[None] * (HEADS // MFMA_M) for _ in range(rows)]
            weight_frags = [
                [[None] * DREG for _ in range(HEADS // MFMA_M)] for _ in range(rows)
            ]
            max_groups = fx.Int32(0)
            table_span = max_pages * fx.Int32(PAGE_SIZE)
            for ri in range_constexpr(rows):
                row_live = fx.Int32(ri) < live
                row = row_live.select(row0 + fx.Int32(ri), row0)
                row_ids[ri] = row
                length = _imin(fx.Int32(lengths[row]), table_span)
                length = (length < 0).select(fx.Int32(0), length)
                length = row_live.select(length, fx.Int32(0))
                row_lengths[ri] = length
                groups = _udiv(length + fx.Int32(GROUP_SIZE - 1), GROUP_SIZE)
                max_groups = (groups > max_groups).select(groups, max_groups)
                q_row_bytes = row * fx.Int32(HEADS * HEAD_DIM)
                for mi in range_constexpr(HEADS // MFMA_M):
                    q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
                    q_tiles[ri][mi] = _load_fp8x32(
                        q_i32,
                        q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                        lane_div_16,
                    )
                    for ii in range_constexpr(DREG):
                        head = mi * MFMA_M + lane_div_16 * DREG + ii
                        weight_frags[ri][mi][ii] = fx.Float32(weight_t[row, head])

            group_begin = _udiv(max_groups * split, num_splits)
            group_end = _udiv(max_groups * (split + fx.Int32(1)), num_splits)
            groups_per_block = BLOCK_N // GROUP_SIZE
            for block_group in range(
                group_begin,
                group_end,
                fx.Int32(groups_per_block),
            ):
                wave_group_base = block_group + wave * fx.Int32(N_TILES_PER_WAVE)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    group = wave_group_base + fx.Int32(ni)
                    logical = group * fx.Int32(GROUP_SIZE) + lane_mod_16
                    safe_logical = _imin(
                        logical,
                        group_end * fx.Int32(GROUP_SIZE) - fx.Int32(1),
                    )
                    table_offset = safe_request * max_pages + _udiv(
                        safe_logical, fx.Int32(PAGE_SIZE)
                    )
                    physical_page = fx.Int32(tables[table_offset])
                    page_ok = (physical_page >= 0) & (physical_page < num_pages)
                    safe_page = page_ok.select(physical_page, fx.Int32(0))
                    page_i32 = (safe_page << 11) + (safe_page << 6)
                    token_in_page = _umod(safe_logical, fx.Int32(PAGE_SIZE))
                    k_pack = _load_preshuffled_fp8x32(
                        kv_i32,
                        page_i32,
                        token_in_page,
                        lane_div_16,
                    )
                    scale = fx.Float32(
                        scale_f32[
                            page_i32
                            + fx.Int32(PAGE_SIZE * HEAD_DIM // 4)
                            + token_in_page
                        ]
                    )
                    for ri in range_constexpr(rows):
                        row_ok = (fx.Int32(ri) < live) & (logical < row_lengths[ri])
                        score = canonical_score_tile(
                            q_tiles[ri],
                            weight_frags[ri],
                            k_pack,
                            scale,
                            page_ok & row_ok,
                            result_type,
                            neutral,
                        )
                        reduced = score
                        reduced = reduced.maximumf(reduced.shuffle_xor(1, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(2, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(4, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(8, WAVE_SIZE))
                        store = (
                            row_ok
                            & (lane_div_16 == 0)
                            & (lane_mod_16 == 0)
                            & (group < group_end)
                        )
                        if store:
                            fx.ptr_store(
                                reduced,
                                fx.add_offset(
                                    fx.get_iter(group_max),
                                    row_ids[ri] * group_stride + group,
                                ),
                            )

    return _bind_launch(kernel)


def _compile_r4_qvgpr(arch: str):
    rows = 4
    kernel_name = fp8_paged_mqa_group_max_kernel_name(arch=arch, rows_per_cta=rows)

    @fx.struct
    class WeightStorage:
        weights: fx.Array[fx.Float32, _WEIGHT_LDS_ELEMS, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        decode_lens: fx.Tensor,
        block_tables: fx.Tensor,
        group_max: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
    ):
        del rows_cap
        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)
        split = fx.Int32(
            rocdl.readfirstlane(T.i32, fx.Int32(gpu.block_idx.x))
        )
        owner = fx.Int32(
            rocdl.readfirstlane(T.i32, fx.Int32(gpu.block_idx.y))
        )
        request = _udiv(owner, row_groups_per_request)
        row_group = _umod(owner, row_groups_per_request)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(packed_kv, dtype=T.i32, shape=(-1,))
        scale_f32 = GTensor(packed_kv, dtype=T.f32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        starts = GTensor(query_start_loc, dtype=T.i32, shape=(-1,))
        widths = GTensor(decode_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        weight_lds = (
            fx.SharedAllocator()
            .allocate(WeightStorage)
            .weights.peek()
            .view(fx.make_layout(_WEIGHT_LDS_ELEMS, 1))
        )

        request_ok = request < request_capacity
        safe_request = request_ok.select(request, fx.Int32(0))
        row0 = fx.Int32(
            rocdl.readfirstlane(T.i32, fx.Int32(starts[safe_request]))
        ) + row_group * fx.Int32(rows)
        remaining = fx.Int32(
            rocdl.readfirstlane(T.i32, fx.Int32(widths[safe_request]))
        ) - row_group * fx.Int32(rows)
        live = _imin(
            (remaining < 0).select(fx.Int32(0), remaining),
            fx.Int32(rows),
        )
        live = request_ok.select(live, fx.Int32(0))

        if live != 0:
            row_lengths = [None] * rows
            q_tiles = [[None] * (HEADS // MFMA_M) for _ in range(rows)]
            max_groups = fx.Int32(0)
            table_span = max_pages * fx.Int32(PAGE_SIZE)
            for ri in range_constexpr(rows):
                row_live = fx.Int32(ri) < live
                row = row_live.select(row0 + fx.Int32(ri), row0)
                length = fx.Int32(
                    rocdl.readfirstlane(T.i32, fx.Int32(lengths[row]))
                )
                length = _imin(length, table_span)
                length = (length < 0).select(fx.Int32(0), length)
                length = row_live.select(length, fx.Int32(0))
                row_lengths[ri] = length
                groups = _udiv(length + fx.Int32(GROUP_SIZE - 1), GROUP_SIZE)
                max_groups = (groups > max_groups).select(groups, max_groups)
                q_row_bytes = row * fx.Int32(HEADS * HEAD_DIM)
                for mi in range_constexpr(HEADS // MFMA_M):
                    q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
                    q_tiles[ri][mi] = _load_fp8x32(
                        q_i32,
                        q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                        lane_div_16,
                    )

            weight_slots = fx.Int32(_WEIGHT_LDS_ELEMS)
            if tid < weight_slots:
                local_row = _udiv(tid, HEADS)
                head = _umod(tid, HEADS)
                row_live = local_row < live
                src_row = row_live.select(row0 + local_row, row0)
                weight_lds[tid] = fx.Float32(weight_t[src_row, head])
            gpu.barrier()

            group_begin = _udiv(max_groups * split, num_splits)
            group_end = _udiv(max_groups * (split + fx.Int32(1)), num_splits)
            groups_per_block = BLOCK_N // GROUP_SIZE
            for block_group in range(
                group_begin,
                group_end,
                fx.Int32(groups_per_block),
            ):
                wave_group_base = block_group + wave * fx.Int32(N_TILES_PER_WAVE)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    group = wave_group_base + fx.Int32(ni)
                    logical = group * fx.Int32(GROUP_SIZE) + lane_mod_16
                    safe_logical = _imin(
                        logical,
                        group_end * fx.Int32(GROUP_SIZE) - fx.Int32(1),
                    )
                    table_offset = safe_request * max_pages + _udiv(
                        safe_logical, fx.Int32(PAGE_SIZE)
                    )
                    physical_page = fx.Int32(tables[table_offset])
                    page_ok = (physical_page >= 0) & (physical_page < num_pages)
                    safe_page = page_ok.select(physical_page, fx.Int32(0))
                    page_i32 = (safe_page << 11) + (safe_page << 6)
                    token_in_page = _umod(safe_logical, fx.Int32(PAGE_SIZE))
                    k_pack = _load_preshuffled_fp8x32(
                        kv_i32,
                        page_i32,
                        token_in_page,
                        lane_div_16,
                    )
                    scale = fx.Float32(
                        scale_f32[
                            page_i32
                            + fx.Int32(PAGE_SIZE * HEAD_DIM // 4)
                            + token_in_page
                        ]
                    )
                    for ri in range_constexpr(rows):
                        row_ok = (fx.Int32(ri) < live) & (
                            logical < row_lengths[ri]
                        )
                        row_weights = _load_weight_frags_from_lds(
                            weight_lds, ri, lane_div_16
                        )
                        score = canonical_score_tile(
                            q_tiles[ri],
                            row_weights,
                            k_pack,
                            scale,
                            page_ok & row_ok,
                            result_type,
                            neutral,
                        )
                        reduced = score
                        reduced = reduced.maximumf(reduced.shuffle_xor(1, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(2, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(4, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(8, WAVE_SIZE))
                        store = (
                            row_ok
                            & (lane_div_16 == 0)
                            & (lane_mod_16 == 0)
                            & (group < group_end)
                        )
                        if store:
                            fx.ptr_store(
                                reduced,
                                fx.add_offset(
                                    fx.get_iter(group_max),
                                    (row0 + fx.Int32(ri)) * group_stride + group,
                                ),
                            )

    return _bind_launch(kernel)


def _compile_r4_qlds(arch: str):
    rows = 4
    kernel_name = fp8_paged_mqa_group_max_kernel_name(arch=arch, rows_per_cta=rows)
    kernel_name = kernel_name + "_qlds"

    @fx.struct
    class QWeightStorage:
        weights: fx.Array[fx.Float32, _WEIGHT_LDS_ELEMS, 16]
        q: fx.Array[fx.Int32, _Q_I32_ELEMS, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        packed_kv: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        decode_lens: fx.Tensor,
        block_tables: fx.Tensor,
        group_max: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
    ):
        del rows_cap
        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)
        split = fx.Int32(gpu.block_idx.x)
        owner = fx.Int32(gpu.block_idx.y)
        request = _udiv(owner, row_groups_per_request)
        row_group = _umod(owner, row_groups_per_request)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(packed_kv, dtype=T.i32, shape=(-1,))
        scale_f32 = GTensor(packed_kv, dtype=T.f32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        starts = GTensor(query_start_loc, dtype=T.i32, shape=(-1,))
        widths = GTensor(decode_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        storage = fx.SharedAllocator().allocate(QWeightStorage)
        weight_lds = storage.weights.peek().view(fx.make_layout(_WEIGHT_LDS_ELEMS, 1))
        q_lds = storage.q.peek().view(fx.make_layout(_Q_I32_ELEMS, 1))

        request_ok = request < request_capacity
        safe_request = request_ok.select(request, fx.Int32(0))
        row0 = fx.Int32(starts[safe_request]) + row_group * fx.Int32(rows)
        remaining = fx.Int32(widths[safe_request]) - row_group * fx.Int32(rows)
        live = _imin(
            (remaining < 0).select(fx.Int32(0), remaining),
            fx.Int32(rows),
        )
        live = request_ok.select(live, fx.Int32(0))

        if live != 0:
            max_groups = fx.Int32(0)
            table_span = max_pages * fx.Int32(PAGE_SIZE)
            for ri in range_constexpr(rows):
                row_live = fx.Int32(ri) < live
                row = row_live.select(row0 + fx.Int32(ri), row0)
                length = _imin(fx.Int32(lengths[row]), table_span)
                length = (length < 0).select(fx.Int32(0), length)
                length = row_live.select(length, fx.Int32(0))
                groups = _udiv(length + fx.Int32(GROUP_SIZE - 1), GROUP_SIZE)
                max_groups = (groups > max_groups).select(groups, max_groups)

            for k in range_constexpr(8):
                idx = tid * fx.Int32(8) + fx.Int32(k)
                local_row = _udiv(idx, _Q_I32_PER_ROW)
                in_row = _umod(idx, _Q_I32_PER_ROW)
                row_live = local_row < live
                src_row = row_live.select(row0 + local_row, row0)
                q_lds[idx] = q_i32[src_row * fx.Int32(_Q_I32_PER_ROW) + in_row]
            weight_slots = fx.Int32(_WEIGHT_LDS_ELEMS)
            if tid < weight_slots:
                local_row = _udiv(tid, HEADS)
                head = _umod(tid, HEADS)
                row_live = local_row < live
                src_row = row_live.select(row0 + local_row, row0)
                weight_lds[tid] = fx.Float32(weight_t[src_row, head])
            gpu.barrier()

            group_begin = _udiv(max_groups * split, num_splits)
            group_end = _udiv(max_groups * (split + fx.Int32(1)), num_splits)
            groups_per_block = BLOCK_N // GROUP_SIZE
            for block_group in range(
                group_begin,
                group_end,
                fx.Int32(groups_per_block),
            ):
                wave_group_base = block_group + wave * fx.Int32(N_TILES_PER_WAVE)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    group = wave_group_base + fx.Int32(ni)
                    logical = group * fx.Int32(GROUP_SIZE) + lane_mod_16
                    safe_logical = _imin(
                        logical,
                        group_end * fx.Int32(GROUP_SIZE) - fx.Int32(1),
                    )
                    table_offset = safe_request * max_pages + _udiv(
                        safe_logical, fx.Int32(PAGE_SIZE)
                    )
                    physical_page = fx.Int32(tables[table_offset])
                    page_ok = (physical_page >= 0) & (physical_page < num_pages)
                    safe_page = page_ok.select(physical_page, fx.Int32(0))
                    page_i32 = (safe_page << 11) + (safe_page << 6)
                    token_in_page = _umod(safe_logical, fx.Int32(PAGE_SIZE))
                    k_pack = _load_preshuffled_fp8x32(
                        kv_i32,
                        page_i32,
                        token_in_page,
                        lane_div_16,
                    )
                    scale = fx.Float32(
                        scale_f32[
                            page_i32
                            + fx.Int32(PAGE_SIZE * HEAD_DIM // 4)
                            + token_in_page
                        ]
                    )
                    for row_i in range(fx.Int32(0), live):
                        src_row = row0 + row_i
                        row_len = _imin(fx.Int32(lengths[src_row]), table_span)
                        row_len = (row_len < 0).select(fx.Int32(0), row_len)
                        row_ok = logical < row_len
                        row_weights = _load_weight_frags_from_lds(
                            weight_lds, row_i, lane_div_16
                        )
                        row_q = _load_q_tiles_from_lds(
                            q_lds, row_i, lane_mod_16, lane_div_16
                        )
                        score = canonical_score_tile(
                            row_q,
                            row_weights,
                            k_pack,
                            scale,
                            page_ok & row_ok,
                            result_type,
                            neutral,
                        )
                        reduced = score
                        reduced = reduced.maximumf(reduced.shuffle_xor(1, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(2, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(4, WAVE_SIZE))
                        reduced = reduced.maximumf(reduced.shuffle_xor(8, WAVE_SIZE))
                        store = (
                            row_ok
                            & (lane_div_16 == 0)
                            & (lane_mod_16 == 0)
                            & (group < group_end)
                        )
                        if store:
                            fx.ptr_store(
                                reduced,
                                fx.add_offset(
                                    fx.get_iter(group_max),
                                    src_row * group_stride + group,
                                ),
                            )

    return _bind_launch(kernel)


@lru_cache(maxsize=8)
def compile_fp8_paged_mqa_group_max(
    *,
    arch: str,
    rows_per_cta: int = ROWS_PER_CTA,
    q_in_lds: bool = False,
):
    """Compile one packed P1 pass specialization."""
    if arch != "gfx950":
        raise RuntimeError("the initial packed P1 pass supports gfx950 only")
    if rows_per_cta not in SUPPORTED_ROWS_PER_CTA:
        raise ValueError(f"rows_per_cta must be 2 or 4, got {rows_per_cta}")
    if q_in_lds and rows_per_cta != 4:
        raise ValueError("Q-in-LDS is only a R=4 occupancy fallback")
    if int(rows_per_cta) == 2:
        return _compile_r2(arch)
    if q_in_lds:
        return _compile_r4_qlds(arch)
    return _compile_r4_qvgpr(arch)


def launch_fp8_paged_mqa_group_max(
    q_fp8,
    packed_kv,
    weights,
    context_lens,
    query_start_loc,
    decode_lens,
    block_tables,
    group_max,
    *,
    max_decode_width,
    num_splits,
    arch,
    stream,
    rows_per_cta=ROWS_PER_CTA,
    q_in_lds=None,
):
    """Launch a fixed-grid packed pass-1 kernel."""
    row_capacity = int(rows_per_cta)
    if q_in_lds is None:
        q_in_lds = False
    launcher = compile_fp8_paged_mqa_group_max(
        arch=arch,
        rows_per_cta=row_capacity,
        q_in_lds=bool(q_in_lds),
    )
    row_groups_per_request = (int(max_decode_width) + row_capacity - 1) // row_capacity
    _run_compiled(
        launcher,
        q_fp8,
        packed_kv,
        weights,
        context_lens,
        query_start_loc,
        decode_lens,
        block_tables,
        group_max,
        q_fp8.shape[0],
        decode_lens.numel(),
        row_groups_per_request,
        int(num_splits),
        block_tables.shape[1],
        packed_kv.shape[0],
        group_max.shape[1],
        stream,
    )
