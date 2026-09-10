# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-aware exact TopK merge for fixed-width local candidate bags."""

from functools import cache

import torch

from .kernels.sorted_split_topk_merge import (
    build_coordinated_rank_sorted_split_topk_merge,
    build_multisequence_sorted_split_topk_merge,
    build_parallel_multisequence_sorted_split_topk_merge,
)
from .kernels.tensor_shim import _run_compiled
from .kernels.topk_per_row_decode import (
    _STATE_SIZE,
    build_topk_per_row_decode_module,
)

_RADIX_BINS = 1 << 11
_BLOCK_THREADS = 256
_WAVE_SIZE = 64


@cache
def _full_widths(device_index: int, rows: int, width: int) -> torch.Tensor:
    return torch.full(
        (rows,),
        width,
        dtype=torch.int32,
        device=torch.device("cuda", device_index),
    )


@cache
def _split_workspace(
    device_index: int,
    stream_id: int,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del stream_id
    device = torch.device("cuda", device_index)
    return (
        torch.zeros((rows, 1, _RADIX_BINS), dtype=torch.int32, device=device),
        torch.empty((rows, _STATE_SIZE), dtype=torch.int32, device=device),
    )


def clear_split_topk_merge_workspace_cache() -> None:
    _split_workspace.cache_clear()
    _full_widths.cache_clear()


@cache
def _build_split_topk_merge(k: int, splits: int, precomputed_first_pass: bool):
    return build_topk_per_row_decode_module(
        k,
        stable=False,
        wave_size=_WAVE_SIZE,
        write_values=True,
        chunks_per_row=splits,
        block_threads=_BLOCK_THREADS,
        split_width=k,
        payload_indices=True,
        nan_to_bottom=True,
        combine_histograms=True,
        use_split_counts=True,
        precomputed_first_pass=precomputed_first_pass,
    )


def split_topk_merge_workspace(
    device: torch.device,
    rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the stream-local histogram and state used by split TopK."""
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.zeros((rows, 1, _RADIX_BINS), dtype=torch.int32, device=device),
            torch.empty((rows, _STATE_SIZE), dtype=torch.int32, device=device),
        )
    stream = torch.cuda.current_stream(device)
    return _split_workspace(device.index, stream.cuda_stream, rows)


def _row_ends(device: torch.device, rows: int, width: int) -> torch.Tensor:
    if torch.cuda.is_current_stream_capturing():
        return torch.full(
            (rows,),
            width,
            dtype=torch.int32,
            device=device,
        )
    return _full_widths(device.index, rows, width)


def split_topk_merge(
    candidate_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    k: int,
    precomputed_first_pass: bool = False,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge unordered split-local TopK pairs into an unordered row TopK."""
    if candidate_scores.ndim != 3:
        raise ValueError("candidate_scores must have shape [rows,splits,local_k]")
    if candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate_positions must match candidate_scores")
    rows, splits, local_k = candidate_scores.shape
    if candidate_counts.shape != (rows, splits):
        raise ValueError("candidate_counts must have shape [rows,splits]")
    if local_k != k:
        raise ValueError(f"local_k must equal k, got {local_k} and {k}")
    if splits <= 1:
        raise ValueError("split-aware merge requires at least two splits")
    if candidate_scores.dtype != torch.float32:
        raise TypeError("candidate_scores must be float32")
    if candidate_positions.dtype != torch.int32:
        raise TypeError("candidate_positions must be int32")
    if not candidate_scores.is_cuda:
        raise ValueError("candidate tensors must be on a CUDA/HIP device")
    if (
        candidate_positions.device != candidate_scores.device
        or candidate_counts.device != candidate_scores.device
    ):
        raise ValueError("candidate tensors must share one device")
    if not candidate_scores.is_contiguous() or not candidate_positions.is_contiguous():
        raise ValueError("candidate scores and positions must be contiguous")

    device = candidate_scores.device
    width = splits * k
    scores = candidate_scores.view(rows, width)
    positions = candidate_positions.view(rows, width)
    if precomputed_first_pass:
        selected_scores = torch.full(
            (rows, k),
            -float("inf"),
            dtype=torch.float32,
            device=device,
        )
        selected_positions = torch.full(
            (rows, k),
            -1,
            dtype=torch.int32,
            device=device,
        )
    else:
        selected_scores = torch.empty((rows, k), dtype=torch.float32, device=device)
        selected_positions = torch.empty((rows, k), dtype=torch.int32, device=device)
    row_ends = _row_ends(device, rows, width)
    stream = torch.cuda.current_stream(device)
    partial_hist, state = (
        split_topk_merge_workspace(device, rows) if workspace is None else workspace
    )
    launcher = _build_split_topk_merge(k, splits, precomputed_first_pass)
    _run_compiled(
        launcher,
        scores,
        positions,
        candidate_counts,
        row_ends,
        selected_positions,
        selected_scores,
        partial_hist,
        state,
        width,
        1,
        width,
        rows,
        stream,
    )
    return selected_scores, selected_positions


def multisequence_sorted_split_topk_merge(
    candidate_scores: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    k: int,
    coordinated_partition: bool = False,
    parallel_representatives: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge descending runs with an explicitly selected experimental algorithm."""
    if candidate_scores.ndim != 3:
        raise ValueError("candidate_scores must have shape [rows,splits,local_k]")
    if candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate_positions must match candidate_scores")
    rows, splits, local_k = candidate_scores.shape
    if candidate_counts.shape != (rows, splits):
        raise ValueError("candidate_counts must have shape [rows,splits]")
    if local_k != k:
        raise ValueError(f"local_k must equal k, got {local_k} and {k}")
    if not 2 <= splits <= 2 * _WAVE_SIZE:
        raise ValueError(f"splits must be in [2, {2 * _WAVE_SIZE}], got {splits}")
    if coordinated_partition and parallel_representatives:
        raise ValueError(
            "coordinated_partition and parallel_representatives are mutually exclusive"
        )
    if coordinated_partition and (k <= 0 or k & (k - 1)):
        raise ValueError(
            f"coordinated partition requires a positive power-of-two k, got {k}"
        )
    if parallel_representatives:
        if splits > _WAVE_SIZE:
            raise ValueError(
                f"parallel representative selection supports at most {_WAVE_SIZE} "
                f"splits, got {splits}"
            )
        remaining = k
        while remaining > 2 * splits:
            remaining -= (remaining // (2 * splits)) * splits
        capacity = 1 << (max(splits * splits, splits * remaining) - 1).bit_length()
        capacity_bytes = (capacity * 4 + 15) // 16 * 16
        split_bytes = (splits * 4 + 15) // 16 * 16
        lds_bytes = (
            3 * capacity_bytes
            + 2 * split_bytes
            + 4
        )
        arch = torch.cuda.get_device_properties(
            candidate_scores.device
        ).gcnArchName.split(":", 1)[0]
        if arch not in ("gfx942", "gfx950"):
            raise ValueError(
                f"parallel representative selection is unsupported on {arch}"
            )
        lds_limit = 64 * 1024 if arch == "gfx942" else 160 * 1024
        if lds_bytes > lds_limit:
            raise ValueError(
                "parallel representative selection requires "
                f"{lds_bytes} LDS bytes on {arch}, exceeding {lds_limit}"
            )
    if candidate_scores.dtype != torch.float32:
        raise TypeError("candidate_scores must be float32")
    if candidate_positions.dtype != torch.int32:
        raise TypeError("candidate_positions must be int32")
    if candidate_counts.dtype != torch.int32:
        raise TypeError("candidate_counts must be int32")
    if not candidate_scores.is_cuda:
        raise ValueError("candidate tensors must be on a CUDA/HIP device")
    if (
        candidate_positions.device != candidate_scores.device
        or candidate_counts.device != candidate_scores.device
    ):
        raise ValueError("candidate tensors must share one device")
    if (
        not candidate_scores.is_contiguous()
        or not candidate_positions.is_contiguous()
        or not candidate_counts.is_contiguous()
    ):
        raise ValueError("candidate tensors must be contiguous")

    selected_scores = torch.empty(
        (rows, k),
        dtype=torch.float32,
        device=candidate_scores.device,
    )
    selected_positions = torch.empty(
        (rows, k),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    if coordinated_partition:
        launcher = build_coordinated_rank_sorted_split_topk_merge(k, splits)
    elif parallel_representatives:
        launcher = build_parallel_multisequence_sorted_split_topk_merge(k, splits)
    else:
        launcher = build_multisequence_sorted_split_topk_merge(k, splits)
    _run_compiled(
        launcher,
        candidate_scores,
        candidate_positions,
        candidate_counts,
        selected_scores,
        selected_positions,
        rows,
        torch.cuda.current_stream(candidate_scores.device),
    )
    return selected_scores, selected_positions
