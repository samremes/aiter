# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-aware exact TopK merge for fixed-width local candidate bags."""

from functools import cache

import torch

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
    stream = torch.cuda.current_stream(device)
    return _split_workspace(device.index, stream.cuda_stream, rows)


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
    row_ends = _full_widths(device.index, rows, width)
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
