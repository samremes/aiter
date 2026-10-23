# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K-stationary group-max pass for packed-varlen paged FP8 MQA."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops
from ..tensor_shim import GTensor, _run_compiled, _to_raw
from ..dpp_utils import update_dpp_i32
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
SUPPORTED_ROWS_PER_CTA = (2, 3, 4)
PAGE_SIZE = 64
GROUP_SIZE = 16
_NEUTRAL_E8M0 = 0x7F7F7F7F
_WEIGHT_LDS_ELEMS = 4 * HEADS
_Q_I32_ELEMS = 4 * HEADS * HEAD_DIM // 4
_Q_I32_PER_ROW = HEADS * HEAD_DIM // 4


def _guarded(pred, fn):
    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()


def _store_dense_logit(
    output,
    row,
    stride,
    logical,
    value,
    pred,
    cache_modifier,
):
    byte_offset = fx.Int64(fx.Uint32(row)) * fx.Int64(fx.Uint32(stride)) * 4
    out = GTensor(
        output,
        dtype=T.f32,
        shape=(-1,),
        cache_modifier=cache_modifier,
        static_bytes_offset_i64=byte_offset,
    )

    def _write(_out=out, _logical=logical, _value=value):
        buffer_ops.buffer_store(
            _value,
            _out.rsrc,
            _logical,
            cache_modifier=cache_modifier,
        )

    _guarded(pred, _write)


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


# row_shl:1/2/4/8 DPP controls. Only lane 0 of each 16-lane row stores a group
# maximum, so a directional reduction is enough and costs one DPP move per step;
# an XOR butterfly would need two for offsets 4 and 8, which cross DPP bank
# boundaries. `shuffle_xor` is worse still: it lowers to ds_swizzle_b32 plus an
# `s_waitcnt lgkmcnt(0)` per step, putting an LDS round trip on the dependency
# chain.
_DPP_ROW_SHL_STEPS = (257, 258, 260, 264)


def _dpp_row_max_f32(value, dpp_ctrl):
    """Fold in ``value`` from a lane further up the same 16-lane DPP row.

    ``bound_ctrl`` is off, so lanes whose source lane does not exist keep their
    own value rather than reading zero — the identity for a maximum, and what
    makes the cascade safe for negative and -inf scores.
    """
    src = _to_raw(value.bitcast(fx.Int32))
    remote = update_dpp_i32(src, src, dpp_ctrl, 0xF, 0xF, False)
    return value.maximumf(fx.Int32(remote).bitcast(fx.Float32))


def _dpp_group_max_rows(scores, rows):
    """Reduce each row's 16 lane scores down to lane 0, all rows in lockstep.

    Stepping every row through one shift before moving to the next leaves the
    scheduler ``rows`` independent chains, so each DPP wait state is filled by a
    sibling row instead of an ``s_nop``.
    """
    reduced = list(scores)
    for dpp_ctrl in _DPP_ROW_SHL_STEPS:
        for ri in range_constexpr(rows):
            reduced[ri] = _dpp_row_max_f32(reduced[ri], dpp_ctrl)
    return reduced


def _dpp_group_max(score):
    """Reduce one row's 16 lane scores down to lane 0."""
    return _dpp_group_max_rows([score], 1)[0]


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
        group_ends: fx.Tensor,
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
            group_ends,
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


def _compile_r_vgpr(
    arch: str,
    rows: int,
    *,
    emit_logits: bool = False,
    store_cache_modifier: int = 0,
    kv_load_cache_modifier: int = 0,
):
    if rows not in (2, 3):
        raise ValueError(f"VGPR scorer supports R=2 or R=3, got {rows}")
    kernel_name = fp8_paged_mqa_group_max_kernel_name(arch=arch, rows_per_cta=rows)
    if emit_logits:
        kernel_name = kernel_name.replace("group_max", "logits") + (
            f"_sc{store_cache_modifier}_kc{kv_load_cache_modifier}"
        )

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
        group_ends: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
    ):
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
        kv_i32 = GTensor(
            packed_kv,
            dtype=T.i32,
            shape=(-1,),
            cache_modifier=kv_load_cache_modifier if emit_logits else 0,
        )
        scale_f32 = GTensor(
            packed_kv,
            dtype=T.f32,
            shape=(-1,),
            cache_modifier=kv_load_cache_modifier if emit_logits else 0,
        )
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        starts = GTensor(query_start_loc, dtype=T.i32, shape=(-1,))
        widths = GTensor(decode_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        # One CTA turns every row length into the group count pass 2
        # scans with. Pass 2 reads dead slots too, and a slot that was
        # live on an earlier step has to be overwritten even when no row
        # group owns it now, so this cannot fold into the per-row work.
        if const_expr(not emit_logits):
            if (split == 0) & (owner == 0):
                end_span = max_pages * fx.Int32(PAGE_SIZE)
                for end_row in range(tid, rows_cap, fx.Int32(BLOCK_THREADS)):
                    end_len = fx.Int32(lengths[end_row])
                    end_len = (end_len < 0).select(fx.Int32(0), end_len)
                    end_len = _imin(end_len, end_span)
                    fx.ptr_store(
                        _udiv(end_len + fx.Int32(GROUP_SIZE - 1), GROUP_SIZE),
                        fx.add_offset(fx.get_iter(group_ends), end_row),
                    )

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
                        if const_expr(emit_logits):
                            _store_dense_logit(
                                group_max,
                                row_ids[ri],
                                group_stride,
                                logical,
                                score,
                                row_ok & (lane_div_16 == 0) & (group < group_end),
                                store_cache_modifier,
                            )
                        else:
                            reduced = _dpp_group_max(score)
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


def _compile_r4_qvgpr(
    arch: str,
    *,
    emit_logits: bool = False,
    store_cache_modifier: int = 0,
    kv_load_cache_modifier: int = 0,
):
    rows = 4
    kernel_name = fp8_paged_mqa_group_max_kernel_name(arch=arch, rows_per_cta=rows)
    if emit_logits:
        kernel_name = kernel_name.replace("group_max", "logits") + (
            f"_sc{store_cache_modifier}_kc{kv_load_cache_modifier}"
        )

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
        group_ends: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
    ):
        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)
        tid = fx.Int32(gpu.thread_idx.x)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)
        split = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gpu.block_idx.x)))
        owner = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gpu.block_idx.y)))
        request = _udiv(owner, row_groups_per_request)
        row_group = _umod(owner, row_groups_per_request)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(
            packed_kv,
            dtype=T.i32,
            shape=(-1,),
            cache_modifier=kv_load_cache_modifier if emit_logits else 0,
        )
        scale_f32 = GTensor(
            packed_kv,
            dtype=T.f32,
            shape=(-1,),
            cache_modifier=kv_load_cache_modifier if emit_logits else 0,
        )
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        starts = GTensor(query_start_loc, dtype=T.i32, shape=(-1,))
        widths = GTensor(decode_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))
        # One CTA turns every row length into the group count pass 2
        # scans with. Pass 2 reads dead slots too, and a slot that was
        # live on an earlier step has to be overwritten even when no row
        # group owns it now, so this cannot fold into the per-row work.
        if const_expr(not emit_logits):
            if (split == 0) & (owner == 0):
                end_span = max_pages * fx.Int32(PAGE_SIZE)
                for end_row in range(tid, rows_cap, fx.Int32(BLOCK_THREADS)):
                    end_len = fx.Int32(lengths[end_row])
                    end_len = (end_len < 0).select(fx.Int32(0), end_len)
                    end_len = _imin(end_len, end_span)
                    fx.ptr_store(
                        _udiv(end_len + fx.Int32(GROUP_SIZE - 1), GROUP_SIZE),
                        fx.add_offset(fx.get_iter(group_ends), end_row),
                    )
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
                length = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(lengths[row])))
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
                    row_oks = [None] * rows
                    scores = [None] * rows
                    for ri in range_constexpr(rows):
                        row_oks[ri] = (fx.Int32(ri) < live) & (
                            logical < row_lengths[ri]
                        )
                        row_weights = _load_weight_frags_from_lds(
                            weight_lds, ri, lane_div_16
                        )
                        scores[ri] = canonical_score_tile(
                            q_tiles[ri],
                            row_weights,
                            k_pack,
                            scale,
                            page_ok & row_oks[ri],
                            result_type,
                            neutral,
                        )
                        # Keep the rows' MFMA accumulator lifetimes disjoint.
                        # Without this the scheduler overlaps all `rows`
                        # accumulator sets, which spills the tile into AGPRs and
                        # costs a workgroup per CU.
                        rocdl.sched_barrier(0)
                    if const_expr(emit_logits):
                        for ri in range_constexpr(rows):
                            store = (
                                row_oks[ri] & (lane_div_16 == 0) & (group < group_end)
                            )
                            _store_dense_logit(
                                group_max,
                                row0 + fx.Int32(ri),
                                group_stride,
                                logical,
                                scores[ri],
                                store,
                                store_cache_modifier,
                            )
                    else:
                        reduced_rows = _dpp_group_max_rows(scores, rows)
                        for ri in range_constexpr(rows):
                            store = (
                                row_oks[ri]
                                & (lane_div_16 == 0)
                                & (lane_mod_16 == 0)
                                & (group < group_end)
                            )
                            if store:
                                fx.ptr_store(
                                    reduced_rows[ri],
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
        group_ends: fx.Tensor,
        rows_cap: fx.Int32,
        request_capacity: fx.Int32,
        row_groups_per_request: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        group_stride: fx.Int32,
    ):
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
        # One CTA turns every row length into the group count pass 2
        # scans with. Pass 2 reads dead slots too, and a slot that was
        # live on an earlier step has to be overwritten even when no row
        # group owns it now, so this cannot fold into the per-row work.
        if (split == 0) & (owner == 0):
            end_span = max_pages * fx.Int32(PAGE_SIZE)
            for end_row in range(tid, rows_cap, fx.Int32(BLOCK_THREADS)):
                end_len = fx.Int32(lengths[end_row])
                end_len = (end_len < 0).select(fx.Int32(0), end_len)
                end_len = _imin(end_len, end_span)
                fx.ptr_store(
                    _udiv(end_len + fx.Int32(GROUP_SIZE - 1), GROUP_SIZE),
                    fx.add_offset(fx.get_iter(group_ends), end_row),
                )
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
                        reduced = _dpp_group_max(score)
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
        raise ValueError(f"rows_per_cta must be 2, 3, or 4, got {rows_per_cta}")
    if q_in_lds and rows_per_cta != 4:
        raise ValueError("Q-in-LDS is only a R=4 occupancy fallback")
    if int(rows_per_cta) in (2, 3):
        return _compile_r_vgpr(arch, int(rows_per_cta))
    if q_in_lds:
        return _compile_r4_qlds(arch)
    return _compile_r4_qvgpr(arch)


@lru_cache(maxsize=32)
def compile_fp8_paged_mqa_logits(
    *,
    arch: str,
    rows_per_cta: int = 4,
    store_cache_modifier: int = 0,
    kv_load_cache_modifier: int = 0,
):
    """Compile an R=2/R=3/R=4 K-stationary dense-logits scorer."""
    if arch != "gfx950":
        raise RuntimeError("the experimental logits scorer supports gfx950 only")
    if rows_per_cta not in SUPPORTED_ROWS_PER_CTA:
        raise ValueError(f"rows_per_cta must be 2, 3, or 4, got {rows_per_cta}")
    if store_cache_modifier not in (0, 1, 2, 3, 4):
        raise ValueError("store_cache_modifier must be in [0, 4]")
    if kv_load_cache_modifier not in (0, 1, 2, 3, 4):
        raise ValueError("kv_load_cache_modifier must be in [0, 4]")
    if rows_per_cta in (2, 3):
        return _compile_r_vgpr(
            arch,
            int(rows_per_cta),
            emit_logits=True,
            store_cache_modifier=int(store_cache_modifier),
            kv_load_cache_modifier=int(kv_load_cache_modifier),
        )
    return _compile_r4_qvgpr(
        arch,
        emit_logits=True,
        store_cache_modifier=int(store_cache_modifier),
        kv_load_cache_modifier=int(kv_load_cache_modifier),
    )


def launch_fp8_paged_mqa_group_max(
    q_fp8,
    packed_kv,
    weights,
    context_lens,
    query_start_loc,
    decode_lens,
    block_tables,
    group_max,
    group_ends,
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
        group_ends,
        q_fp8.shape[0],
        decode_lens.numel(),
        row_groups_per_request,
        int(num_splits),
        block_tables.shape[1],
        packed_kv.shape[0],
        group_max.shape[1],
        stream,
    )


def launch_fp8_paged_mqa_logits(
    q_fp8,
    packed_kv,
    weights,
    context_lens,
    query_start_loc,
    decode_lens,
    block_tables,
    logits,
    *,
    max_decode_width,
    num_splits,
    arch,
    stream,
    rows_per_cta=4,
    store_cache_modifier=0,
    kv_load_cache_modifier=0,
):
    """Launch the experimental R=2/R=4 scorer into dense logits."""
    launcher = compile_fp8_paged_mqa_logits(
        arch=arch,
        rows_per_cta=int(rows_per_cta),
        store_cache_modifier=int(store_cache_modifier),
        kv_load_cache_modifier=int(kv_load_cache_modifier),
    )
    row_groups_per_request = (int(max_decode_width) + rows_per_cta - 1) // rows_per_cta
    _run_compiled(
        launcher,
        q_fp8,
        packed_kv,
        weights,
        context_lens,
        query_start_loc,
        decode_lens,
        block_tables,
        logits,
        logits,
        q_fp8.shape[0],
        decode_lens.numel(),
        row_groups_per_request,
        int(num_splits),
        block_tables.shape[1],
        packed_kv.shape[0],
        logits.stride(0),
        stream,
    )
