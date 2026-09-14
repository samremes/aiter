# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed-varlen group-max pruning prototype for paged FP8 MQA."""

from dataclasses import dataclass

import torch

from ..topk import top_k_per_row_decode
from .fp8_paged_mqa_local_topk import (
    HEAD_DIM,
    INDEX_DIM,
    _arch_name,
    _require_cuda_contiguous,
)
from .kernels.mqa_logits.fp8_paged_mqa_group_harvest import (
    HARVEST_SPLITS,
    launch_fp8_paged_mqa_group_harvest,
    launch_fp8_paged_mqa_position_map,
)
from .kernels.mqa_logits.fp8_paged_mqa_group_max import (
    ROWS_PER_CTA,
    launch_fp8_paged_mqa_group_max,
)

HEADS = 32
PAGE_SIZE = 64
GROUP_SIZE = 16
SUPPORTED_ARCHES = ("gfx950",)


@dataclass
class GroupMaxTopKPrototypeResult:
    """Intermediate and final values from the PyTorch harvest proof."""

    canonical_scores: list[torch.Tensor]
    group_max: torch.Tensor
    group_ends: torch.Tensor
    selected_group_ids: torch.Tensor
    selected_group_counts: torch.Tensor
    scores: torch.Tensor
    positions: torch.Tensor


@dataclass
class GroupMaxTopKResult:
    """Outputs and observable intermediates from the dense-harvest bring-up."""

    scores: torch.Tensor
    positions: torch.Tensor
    group_max: torch.Tensor
    group_ends: torch.Tensor
    selected_group_scores: torch.Tensor | None
    selected_group_ids: torch.Tensor
    selected_group_counts: torch.Tensor
    harvest_scores: torch.Tensor
    harvest_positions: torch.Tensor | None
    harvest_lengths: torch.Tensor
    harvest_ordinals: torch.Tensor


def _unpack_preshuffled_k(
    packed_preshuffled_kv: torch.Tensor,
    q_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover logical K and scales from packed page-64 storage."""
    pages = int(packed_preshuffled_kv.shape[0])
    raw = packed_preshuffled_kv.view(torch.uint8).reshape(pages, -1)
    shuffled = raw[:, : PAGE_SIZE * HEAD_DIM].reshape(
        pages,
        PAGE_SIZE,
        HEAD_DIM,
    )
    # Inverse of shuffle_weight(..., layout=(16, 16)) for one-byte FP8.
    keys = (
        shuffled.view(pages, 4, 4, 2, 16, 16)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(q_dtype)
        .reshape(pages, PAGE_SIZE, HEAD_DIM)
    )
    scales = raw[:, PAGE_SIZE * HEAD_DIM :].contiguous().view(torch.float32)
    return keys, scales


def _validate_prototype_inputs(
    q_fp8,
    packed_preshuffled_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    query_start_loc,
    decode_lens,
    *,
    group_size,
):
    if indices is None:
        raise ValueError("indices is required for the packed P1 ABI")
    if query_start_loc is None:
        raise ValueError("query_start_loc is required for the packed P1 ABI")
    if decode_lens is None:
        raise ValueError("decode_lens is required for the packed P1 ABI")
    for name, tensor in (
        ("q_fp8", q_fp8),
        ("packed_preshuffled_kv", packed_preshuffled_kv),
        ("weights", weights),
        ("context_lens", context_lens),
        ("block_tables", block_tables),
        ("indices", indices),
        ("query_start_loc", query_start_loc),
        ("decode_lens", decode_lens),
    ):
        _require_cuda_contiguous(name, tensor)
    if q_fp8.ndim == 4:
        if q_fp8.shape[1:] != (1, HEADS, HEAD_DIM):
            raise ValueError(
                "q_fp8 must be packed [R,32,128] or [R,1,32,128], "
                f"got {tuple(q_fp8.shape)}"
            )
        rows = int(q_fp8.shape[0])
    elif q_fp8.ndim == 3 and q_fp8.shape[1:] == (HEADS, HEAD_DIM):
        rows = int(q_fp8.shape[0])
    else:
        raise ValueError(
            "q_fp8 must be packed [R,32,128] or [R,1,32,128], "
            f"got {tuple(q_fp8.shape)}"
        )
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q_fp8 must be float8_e4m3fn, got {q_fp8.dtype}")
    if packed_preshuffled_kv.ndim != 4 or packed_preshuffled_kv.shape[1:] != (
        PAGE_SIZE,
        1,
        INDEX_DIM,
    ):
        raise ValueError(
            "packed_preshuffled_kv must have shape [pages,64,1,132], "
            f"got {tuple(packed_preshuffled_kv.shape)}"
        )
    if packed_preshuffled_kv.dtype not in (torch.uint8, q_fp8.dtype):
        raise ValueError("packed_preshuffled_kv must be uint8 or match q_fp8 dtype")
    if weights.shape != (rows, HEADS) or weights.dtype != torch.float32:
        raise ValueError(f"weights must be float32 with shape {(rows, HEADS)}")
    if context_lens.shape != (rows,) or context_lens.dtype != torch.int32:
        raise ValueError(f"context_lens must be int32 with shape {(rows,)}")
    if indices.shape != (rows,) or indices.dtype != torch.int32:
        raise ValueError(f"indices must be int32 with shape {(rows,)}")
    requests = int(decode_lens.numel())
    if decode_lens.ndim != 1 or decode_lens.dtype != torch.int32:
        raise ValueError("decode_lens must be contiguous int32 [request_capacity]")
    if query_start_loc.shape != (requests + 1,) or query_start_loc.dtype != torch.int32:
        raise ValueError(
            "query_start_loc must be contiguous int32 [request_capacity + 1]"
        )
    if block_tables.ndim != 2 or block_tables.shape[0] != requests:
        raise ValueError("block_tables must have one row per request slot")
    if block_tables.dtype != torch.int32:
        raise ValueError("block_tables must be int32")
    if group_size != GROUP_SIZE:
        raise ValueError(f"group_size must be {GROUP_SIZE}, got {group_size}")
    if _arch_name(q_fp8.device) not in SUPPORTED_ARCHES:
        raise RuntimeError("the initial P1 prototype supports gfx950 only")
    if any(
        tensor.device != q_fp8.device
        for tensor in (
            packed_preshuffled_kv,
            weights,
            context_lens,
            block_tables,
            indices,
            query_start_loc,
            decode_lens,
        )
    ):
        raise ValueError("all inputs must be on the same device")
    return rows


def _canonical_scores_for_row(
    q_row: torch.Tensor,
    keys: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """Apply the P1 canonical epilogue to one logical row."""
    if length == 0:
        return torch.empty(0, dtype=torch.float32, device=q_row.device)
    logical = torch.arange(length, dtype=torch.int64, device=q_row.device)
    physical_pages = block_table[logical // PAGE_SIZE].to(torch.int64)
    page_ok = (physical_pages >= 0) & (physical_pages < keys.shape[0])
    safe_pages = torch.where(page_ok, physical_pages, 0)
    physical_tokens = logical % PAGE_SIZE
    selected_keys = keys[safe_pages, physical_tokens].float()
    selected_scales = scales[safe_pages, physical_tokens]
    dots = torch.sum(
        q_row.float()[:, None, :] * selected_keys[None, :, :],
        dim=-1,
    )
    total = torch.sum(weights[:, None] * torch.relu(dots), dim=0)
    nonnegative_scale = torch.fmax(
        selected_scales,
        torch.zeros((), dtype=torch.float32, device=q_row.device),
    )
    scores = total * nonnegative_scale
    return torch.where(
        page_ok & ~torch.isnan(scores),
        scores,
        torch.full_like(scores, float("-inf")),
    )


def torch_group_max_topk_prototype(
    q_fp8,
    packed_preshuffled_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    query_start_loc,
    decode_lens,
    *,
    k=2048,
    group_size=GROUP_SIZE,
) -> GroupMaxTopKPrototypeResult:
    """Prove exact unstable TopK by group-max pruning and direct re-score.

    This is intentionally a PyTorch bring-up implementation. It accepts only
    the packed varlen ABI that the eventual kernels will serve.
    """
    rows = _validate_prototype_inputs(
        q_fp8,
        packed_preshuffled_kv,
        weights,
        context_lens,
        block_tables,
        indices,
        query_start_loc,
        decode_lens,
        group_size=group_size,
    )
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    q = q_fp8.reshape(rows, HEADS, HEAD_DIM)
    keys, scales = _unpack_preshuffled_k(packed_preshuffled_kv, q_fp8.dtype)
    max_length = int(block_tables.shape[1]) * PAGE_SIZE
    group_width = (max_length + group_size - 1) // group_size
    group_max = torch.full(
        (rows, group_width),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    group_ends = torch.zeros(rows, dtype=torch.int32, device=q.device)
    selected_group_ids = torch.full(
        (rows, k),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    selected_group_counts = torch.zeros(rows, dtype=torch.int32, device=q.device)
    output_scores = torch.full(
        (rows, k),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    output_positions = torch.full(
        (rows, k),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    canonical_scores = []

    for row in range(rows):
        length = min(max(int(context_lens[row]), 0), max_length)
        request = int(indices[row])
        scores = _canonical_scores_for_row(
            q[row],
            keys,
            scales,
            weights[row],
            block_tables[request],
            length,
        )
        canonical_scores.append(scores)
        groups = (length + group_size - 1) // group_size
        group_ends[row] = groups
        if groups == 0:
            continue
        padded = torch.full(
            (groups * group_size,),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        padded[:length] = scores
        maxima = padded.view(groups, group_size).amax(dim=1)
        group_max[row, :groups] = maxima
        group_count = min(k, groups)
        group_ids = torch.topk(
            maxima,
            group_count,
            sorted=False,
        ).indices
        selected_group_ids[row, :group_count] = group_ids.to(torch.int32)
        selected_group_counts[row] = group_count

        harvested_positions = (
            group_ids[:, None] * group_size
            + torch.arange(group_size, device=q.device)[None, :]
        ).reshape(-1)
        harvested_positions = harvested_positions[harvested_positions < length]
        # Re-score through the same canonical primitive rather than gathering
        # the pass-1 values.
        rescored = _canonical_scores_for_row(
            q[row],
            keys,
            scales,
            weights[row],
            block_tables[request],
            length,
        )
        harvested_scores = rescored[harvested_positions]
        output_count = min(k, length)
        chosen = torch.topk(
            harvested_scores,
            output_count,
            sorted=False,
        ).indices
        output_scores[row, :output_count] = harvested_scores[chosen]
        output_positions[row, :output_count] = harvested_positions[chosen].to(
            torch.int32
        )

    return GroupMaxTopKPrototypeResult(
        canonical_scores=canonical_scores,
        group_max=group_max,
        group_ends=group_ends,
        selected_group_ids=selected_group_ids,
        selected_group_counts=selected_group_counts,
        scores=output_scores,
        positions=output_positions,
    )


def flydsl_fp8_paged_mqa_group_max(
    q_fp8,
    packed_preshuffled_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    query_start_loc,
    decode_lens,
    *,
    max_decode_width,
    num_splits=64,
    group_size=GROUP_SIZE,
    rows_per_cta=ROWS_PER_CTA,
    q_in_lds=None,
):
    """Run an experimental packed pass and return maxima plus ragged row ends."""
    rows = _validate_prototype_inputs(
        q_fp8,
        packed_preshuffled_kv,
        weights,
        context_lens,
        block_tables,
        indices,
        query_start_loc,
        decode_lens,
        group_size=group_size,
    )
    if not isinstance(max_decode_width, int) or max_decode_width < 2:
        raise ValueError("max_decode_width must be a captured integer >= 2")
    if rows_per_cta not in (2, 4):
        raise ValueError(f"rows_per_cta must be 2 or 4, got {rows_per_cta}")
    if not isinstance(num_splits, int) or num_splits <= 0:
        raise ValueError("num_splits must be a positive integer")
    max_length = int(block_tables.shape[1]) * PAGE_SIZE
    group_width = (max_length + group_size - 1) // group_size
    group_max = torch.empty(
        (rows, group_width),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    group_ends = (
        context_lens.clamp(min=0, max=max_length)
        .add(group_size - 1)
        .div(group_size, rounding_mode="floor")
        .to(torch.int32)
    )
    stream = torch.cuda.current_stream(q_fp8.device)
    with torch.cuda.device(q_fp8.device):
        launch_fp8_paged_mqa_group_max(
            q_fp8,
            packed_preshuffled_kv,
            weights,
            context_lens,
            query_start_loc,
            decode_lens,
            block_tables,
            group_max,
            max_decode_width=max_decode_width,
            num_splits=num_splits,
            arch=_arch_name(q_fp8.device),
            stream=stream,
            rows_per_cta=rows_per_cta,
            q_in_lds=q_in_lds,
        )
    return group_max, group_ends


def flydsl_fp8_paged_mqa_group_harvest(
    q_fp8,
    packed_preshuffled_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    query_start_loc,
    decode_lens,
    selected_group_ids,
    selected_group_counts,
    *,
    group_size=GROUP_SIZE,
    harvest_splits=HARVEST_SPLITS,
    harvest_scores=None,
    harvest_positions=None,
    harvest_lengths=None,
    emit_positions=True,
):
    """Dense-rescore selected groups into deterministic ordinal slots."""
    rows = _validate_prototype_inputs(
        q_fp8,
        packed_preshuffled_kv,
        weights,
        context_lens,
        block_tables,
        indices,
        query_start_loc,
        decode_lens,
        group_size=group_size,
    )
    if harvest_splits != HARVEST_SPLITS:
        raise ValueError(f"harvest_splits must be {HARVEST_SPLITS}")
    if (
        selected_group_ids.ndim != 2
        or selected_group_ids.shape[0] != rows
        or selected_group_ids.dtype != torch.int32
    ):
        raise ValueError("selected_group_ids must be contiguous int32 [rows,k]")
    if not selected_group_ids.is_contiguous():
        raise ValueError("selected_group_ids must be contiguous")
    if (
        selected_group_counts.shape != (rows,)
        or selected_group_counts.dtype != torch.int32
        or not selected_group_counts.is_contiguous()
    ):
        raise ValueError("selected_group_counts must be contiguous int32 [rows]")
    if (
        selected_group_ids.device != q_fp8.device
        or selected_group_counts.device != q_fp8.device
    ):
        raise ValueError("selected group metadata must be on the input device")

    width = int(selected_group_ids.shape[1]) * GROUP_SIZE
    expected_shape = (rows, width)
    if harvest_scores is None:
        harvest_scores = torch.empty(
            expected_shape, dtype=torch.float32, device=q_fp8.device
        )
    if emit_positions and harvest_positions is None:
        harvest_positions = torch.empty(
            expected_shape, dtype=torch.int32, device=q_fp8.device
        )
    if harvest_lengths is None:
        harvest_lengths = torch.empty((rows,), dtype=torch.int32, device=q_fp8.device)
    outputs = [
        ("harvest_scores", harvest_scores, torch.float32, expected_shape),
        ("harvest_lengths", harvest_lengths, torch.int32, (rows,)),
    ]
    if emit_positions:
        outputs.append(
            ("harvest_positions", harvest_positions, torch.int32, expected_shape)
        )
    for name, tensor, dtype, shape in outputs:
        if tensor.dtype != dtype or tensor.shape != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape}")
        if tensor.device != q_fp8.device:
            raise ValueError(f"{name} must be on the input device")

    stream = torch.cuda.current_stream(q_fp8.device)
    with torch.cuda.device(q_fp8.device):
        launch_fp8_paged_mqa_group_harvest(
            q_fp8,
            packed_preshuffled_kv,
            weights,
            context_lens,
            block_tables,
            indices,
            selected_group_ids,
            selected_group_counts,
            harvest_scores,
            harvest_positions if emit_positions else selected_group_ids,
            harvest_lengths,
            arch=_arch_name(q_fp8.device),
            stream=stream,
            emit_positions=emit_positions,
        )
    return harvest_scores, harvest_positions, harvest_lengths


def flydsl_fp8_paged_mqa_position_map(
    final_ordinals,
    selected_group_ids,
    selected_group_counts,
    context_lens,
    *,
    output_positions=None,
):
    """Materialize observable logical positions from harvest ordinals."""
    rows, output_width = final_ordinals.shape
    if output_positions is None:
        output_positions = torch.empty_like(final_ordinals)
    if (
        final_ordinals.dtype != torch.int32
        or selected_group_ids.dtype != torch.int32
        or selected_group_counts.dtype != torch.int32
        or context_lens.dtype != torch.int32
        or output_positions.dtype != torch.int32
    ):
        raise ValueError("position-map inputs and output must be int32")
    if (
        selected_group_ids.ndim != 2
        or selected_group_ids.shape[0] != rows
        or selected_group_counts.shape != (rows,)
        or context_lens.shape != (rows,)
        or output_positions.shape != (rows, output_width)
    ):
        raise ValueError("position-map tensor shapes are inconsistent")
    tensors = (
        final_ordinals,
        selected_group_ids,
        selected_group_counts,
        context_lens,
        output_positions,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("position-map tensors must be contiguous")
    if any(tensor.device != final_ordinals.device for tensor in tensors):
        raise ValueError("position-map tensors must share a device")
    stream = torch.cuda.current_stream(final_ordinals.device)
    with torch.cuda.device(final_ordinals.device):
        launch_fp8_paged_mqa_position_map(
            final_ordinals,
            selected_group_ids,
            selected_group_counts,
            context_lens,
            output_positions,
            arch=_arch_name(final_ordinals.device),
            stream=stream,
        )
    return output_positions


def flydsl_fp8_paged_mqa_group_max_topk(
    q_fp8,
    packed_preshuffled_kv,
    weights,
    context_lens,
    block_tables,
    indices,
    query_start_loc,
    decode_lens,
    *,
    max_decode_width,
    k=2048,
    group_size=GROUP_SIZE,
    num_splits=64,
    harvest_splits=HARVEST_SPLITS,
    rows_per_cta=4,
    group_max=None,
    group_ends=None,
    selected_group_scores=None,
    selected_group_ids=None,
    selected_group_counts=None,
) -> GroupMaxTopKResult:
    """Run the safe dense-harvest bring-up without serving dispatch changes."""
    if k != 2048:
        raise ValueError("the initial dense-harvest pipeline requires k=2048")
    rows = int(q_fp8.shape[0])
    if group_max is None or group_ends is None:
        if group_max is not None or group_ends is not None:
            raise ValueError("group_max and group_ends must be supplied together")
        group_max, group_ends = flydsl_fp8_paged_mqa_group_max(
            q_fp8,
            packed_preshuffled_kv,
            weights,
            context_lens,
            block_tables,
            indices,
            query_start_loc,
            decode_lens,
            max_decode_width=max_decode_width,
            num_splits=num_splits,
            group_size=group_size,
            rows_per_cta=rows_per_cta,
        )

    if selected_group_ids is None or selected_group_counts is None:
        if selected_group_ids is not None or selected_group_counts is not None:
            raise ValueError(
                "selected_group_ids and selected_group_counts must be supplied together"
            )
        selected_group_ids = torch.empty(
            (rows, k), dtype=torch.int32, device=q_fp8.device
        )
        selected_group_counts = torch.minimum(
            group_ends,
            torch.full_like(group_ends, k),
        )
        top_k_per_row_decode(
            logits=group_max,
            next_n=1,
            seqLens=group_ends,
            indices=selected_group_ids,
            numRows=rows,
            stride0=group_max.stride(0),
            stride1=1,
            k=k,
            stable=False,
            values=selected_group_scores,
        )
    else:
        if selected_group_ids.shape != (rows, k):
            raise ValueError(f"selected_group_ids must have shape {(rows, k)}")
        if selected_group_counts.shape != (rows,):
            raise ValueError(f"selected_group_counts must have shape {(rows,)}")
        if selected_group_scores is not None and (
            selected_group_scores.shape != (rows, k)
            or selected_group_scores.dtype != torch.float32
        ):
            raise ValueError(
                f"selected_group_scores must be float32 with shape {(rows, k)}"
            )
    harvest_scores, harvest_positions, harvest_lengths = (
        flydsl_fp8_paged_mqa_group_harvest(
            q_fp8,
            packed_preshuffled_kv,
            weights,
            context_lens,
            block_tables,
            indices,
            query_start_loc,
            decode_lens,
            selected_group_ids,
            selected_group_counts,
            group_size=group_size,
            harvest_splits=harvest_splits,
            emit_positions=False,
        )
    )
    harvest_ordinals = torch.empty((rows, k), dtype=torch.int32, device=q_fp8.device)
    scores = torch.empty((rows, k), dtype=torch.float32, device=q_fp8.device)
    top_k_per_row_decode(
        logits=harvest_scores,
        next_n=1,
        seqLens=harvest_lengths,
        indices=harvest_ordinals,
        numRows=rows,
        stride0=harvest_scores.stride(0),
        stride1=1,
        k=k,
        stable=False,
        values=scores,
    )
    positions = flydsl_fp8_paged_mqa_position_map(
        harvest_ordinals,
        selected_group_ids,
        selected_group_counts,
        context_lens,
    )
    return GroupMaxTopKResult(
        scores=scores,
        positions=positions,
        group_max=group_max,
        group_ends=group_ends,
        selected_group_scores=selected_group_scores,
        selected_group_ids=selected_group_ids,
        selected_group_counts=selected_group_counts,
        harvest_scores=harvest_scores,
        harvest_positions=harvest_positions,
        harvest_lengths=harvest_lengths,
        harvest_ordinals=harvest_ordinals,
    )


def group_space_split_bounds(
    length: int,
    split: int,
    num_splits: int,
    *,
    group_size: int = GROUP_SIZE,
) -> tuple[int, int]:
    """Return an aligned token interval whose groups cannot be bisected."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    if num_splits <= 0 or not 0 <= split < num_splits:
        raise ValueError("split must be in [0, num_splits)")
    groups = (length + group_size - 1) // group_size
    begin_group = groups * split // num_splits
    end_group = groups * (split + 1) // num_splits
    return group_size * begin_group, group_size * end_group
