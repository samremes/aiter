# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense indexed group harvest for packed-varlen paged FP8 MQA."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import T

from ..tensor_shim import GTensor, _run_compiled, _to_raw
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
HARVEST_SPLITS = 8
POSITION_MAP_THREADS = 256
_NEUTRAL_E8M0 = 0x7F7F7F7F


def fp8_paged_mqa_group_harvest_kernel_name(
    *, arch: str, emit_positions: bool = True
) -> str:
    suffix = "" if emit_positions else "_scores_only"
    return f"fp8_paged_mqa_group_harvest_h32d128_g16_c8_w8_{arch}{suffix}"


@lru_cache(maxsize=8)
def compile_fp8_paged_mqa_group_harvest(*, arch: str, emit_positions: bool = True):
    """Compile the gfx950 dense-harvest specialization."""
    if arch != "gfx950":
        raise RuntimeError("the initial packed group harvest supports gfx950 only")

    kernel_name = fp8_paged_mqa_group_harvest_kernel_name(
        arch=arch, emit_positions=emit_positions
    )

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
        harvest_scores: fx.Tensor,
        harvest_positions: fx.Tensor,
        harvest_lengths: fx.Tensor,
        rows: fx.Int32,
        request_capacity: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        selected_width: fx.Int32,
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
        if (split == 0) & (tid == 0) & row_ok:
            fx.ptr_store(
                count * fx.Int32(GROUP_SIZE),
                fx.add_offset(fx.get_iter(harvest_lengths), row),
            )

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
            if (lane_div_16 == 0) & (ordinal < ordinal_end):
                output = safe_row * selected_width * fx.Int32(GROUP_SIZE)
                output = output + ordinal * fx.Int32(GROUP_SIZE) + lane_mod_16
                fx.ptr_store(score, fx.add_offset(fx.get_iter(harvest_scores), output))
                if emit_positions:
                    position = position_ok.select(logical, fx.Int32(-1))
                    fx.ptr_store(
                        position,
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
        harvest_scores: fx.Tensor,
        harvest_positions: fx.Tensor,
        harvest_lengths: fx.Tensor,
        rows: fx.Int32,
        request_capacity: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        selected_width: fx.Int32,
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
            harvest_scores,
            harvest_positions,
            harvest_lengths,
            rows,
            request_capacity,
            max_pages,
            num_pages,
            selected_width,
        ).launch(
            grid=(gx, HARVEST_SPLITS, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": True}
    return launch


def launch_fp8_paged_mqa_group_harvest(
    q_fp8,
    packed_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    selected_group_ids,
    selected_group_counts,
    harvest_scores,
    harvest_positions,
    harvest_lengths,
    *,
    arch,
    stream,
    emit_positions=True,
):
    """Launch the fixed-C dense indexed harvest."""
    launcher = compile_fp8_paged_mqa_group_harvest(
        arch=arch, emit_positions=bool(emit_positions)
    )
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
        harvest_scores,
        harvest_positions,
        harvest_lengths,
        q_fp8.shape[0],
        block_tables.shape[0],
        block_tables.shape[1],
        packed_kv.shape[0],
        selected_group_ids.shape[1],
        stream,
    )


def fp8_paged_mqa_position_map_kernel_name(*, arch: str) -> str:
    return f"fp8_paged_mqa_group_position_map_g16_{arch}"


@lru_cache(maxsize=4)
def compile_fp8_paged_mqa_position_map(*, arch: str):
    """Compile ordinal-to-logical-position materialization."""
    if arch != "gfx950":
        raise RuntimeError("the initial position map supports gfx950 only")

    kernel_name = fp8_paged_mqa_position_map_kernel_name(arch=arch)

    @flyc.kernel(name=kernel_name, known_block_size=[POSITION_MAP_THREADS, 1, 1])
    def kernel(
        final_ordinals: fx.Tensor,
        selected_group_ids: fx.Tensor,
        selected_group_counts: fx.Tensor,
        context_lens: fx.Tensor,
        output_positions: fx.Tensor,
        rows: fx.Int32,
        selected_width: fx.Int32,
        output_width: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        row = fx.Int32(gpu.block_idx.x)
        ordinals = GTensor(final_ordinals, dtype=T.i32, shape=(-1,))
        group_ids = GTensor(selected_group_ids, dtype=T.i32, shape=(-1,))
        group_counts = GTensor(selected_group_counts, dtype=T.i32, shape=(-1,))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))

        row_ok = row < rows
        safe_row = row_ok.select(row, fx.Int32(0))
        count = fx.Int32(group_counts[safe_row])
        count = (count < 0).select(fx.Int32(0), count)
        count = (count < selected_width).select(count, selected_width)
        length = fx.Int32(lengths[safe_row])
        length = (length < 0).select(fx.Int32(0), length)
        for column in range(tid, output_width, fx.Int32(POSITION_MAP_THREADS)):
            ordinal = fx.Int32(ordinals[safe_row * output_width + column])
            ordinal_ok = (ordinal >= 0) & (
                ordinal < selected_width * fx.Int32(GROUP_SIZE)
            )
            group_ordinal = _udiv(ordinal_ok.select(ordinal, fx.Int32(0)), GROUP_SIZE)
            group_ordinal_ok = ordinal_ok & (group_ordinal < count)
            safe_group_ordinal = group_ordinal_ok.select(group_ordinal, fx.Int32(0))
            group = fx.Int32(group_ids[safe_row * selected_width + safe_group_ordinal])
            token = _umod(ordinal_ok.select(ordinal, fx.Int32(0)), GROUP_SIZE)
            logical = group * fx.Int32(GROUP_SIZE) + token
            valid = row_ok & group_ordinal_ok & (group >= 0) & (logical < length)
            fx.ptr_store(
                valid.select(logical, fx.Int32(-1)),
                fx.add_offset(
                    fx.get_iter(output_positions), row * output_width + column
                ),
            )

    @flyc.jit
    def launch(
        final_ordinals: fx.Tensor,
        selected_group_ids: fx.Tensor,
        selected_group_counts: fx.Tensor,
        context_lens: fx.Tensor,
        output_positions: fx.Tensor,
        rows: fx.Int32,
        selected_width: fx.Int32,
        output_width: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(rows))
        kernel(
            final_ordinals,
            selected_group_ids,
            selected_group_counts,
            context_lens,
            output_positions,
            rows,
            selected_width,
            output_width,
        ).launch(
            grid=(gx, 1, 1),
            block=(POSITION_MAP_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1}
    return launch


def launch_fp8_paged_mqa_position_map(
    final_ordinals,
    selected_group_ids,
    selected_group_counts,
    context_lens,
    output_positions,
    *,
    arch,
    stream,
):
    """Map selected dense-harvest ordinals to logical positions."""
    launcher = compile_fp8_paged_mqa_position_map(arch=arch)
    _run_compiled(
        launcher,
        final_ordinals,
        selected_group_ids,
        selected_group_counts,
        context_lens,
        output_positions,
        final_ordinals.shape[0],
        selected_group_ids.shape[1],
        final_ordinals.shape[1],
        stream,
    )
