# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL fused blockscale-fp8 g1u1 silu MoE (gfx950)."""

from __future__ import annotations

import math

import flydsl.compiler as flyc
import torch
from torch import Tensor

from .kernels.fmoe_blockscale_gfx950 import (
    LDS_MAX,
    MAX_SLICES,
    check_config,
    launch_fmoe_blockscale,
    lds_bytes,
)
from .kernels.kernels_common import set_slp_vectorize_once_used
from .kernels.tensor_shim import _compile_and_run, ptr_arg

set_slp_vectorize_once_used(launch_fmoe_blockscale, False)

# Compiled launchers by their Constexprs (D, I, cols, slices, grid, rows).
_COMPILED: dict[tuple, flyc.CompiledFunction] = {}
_SEM: dict[torch.device, Tensor] = {}


def _semaphore(device: torch.device) -> Tensor:
    """The down-phase admission counter; every launch leaves it at zero."""
    sem = _SEM.get(device)
    if sem is None:
        sem = _SEM[device] = torch.zeros(16, dtype=torch.int32, device=device)
    return sem


def fmoe_tile_n(inter_dim: int, rows: int = 64) -> int:
    """Slice width: 128 at 128 rows; at 64, 384 where it divides inter_dim
    (fewer token reads per tile, measured faster), else 256."""
    if rows == 128:
        return 128
    return 384 if inter_dim % 384 == 0 else 256


def fmoe_slices(inter_dim: int, tile_n: int, rows: int = 64) -> int:
    """Column slices one tile runs: all of them where they fit, so a token row
    gets one routed partial per expert to merge; else the most that divide
    them and fit the LDS."""
    n = inter_dim // tile_n
    for s in range(min(n, MAX_SLICES[rows]), 0, -1):
        if n % s == 0 and lds_bytes(rows, tile_n, s) <= LDS_MAX:
            if rows == 64 and s < n and s % 2:
                continue  # 64 rows: two where they pair up, else one
            return s
    return 1


def fmoe_grid(
    device: torch.device, inter_dim: int, tile_n: int, slices: int = 1
) -> int:
    """One workgroup per CU, rounded down so the grid splits evenly into XCDs
    and column tiles."""
    ps_deno = inter_dim // (tile_n * slices)
    step = math.lcm(ps_deno, 16)
    return torch.cuda.get_device_properties(device).multi_processor_count // step * step


def fmoe_blockscale_g1u1_flydsl(
    out: Tensor,
    a_q: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    a_scale: Tensor,
    w1_scale: Tensor,
    w2_scale: Tensor,
    tile_n: int | None = None,
    slices: int | None = None,
    rows: int = 64,
) -> Tensor:
    """``out[t] += sum_k w_k * down(silu(gate(x_t)) * up(x_t))`` for sorted
    ``rows``-row blocks; ``out`` must be zeroed by the caller.

    ``a_q`` ``[T, D]`` fp8 with ``a_scale`` ``[D/128, T]``; ``w1`` ``[E, 2I, D]``
    and ``w2`` ``[E, D, I]`` fp8 preshuffled (16, 16), with 128x128 block scales
    ``w1_scale`` ``[E, 2I/128, D/128]`` and ``w2_scale`` ``[E, D/128, I/128]``.
    The sort is ``moe_sorting`` with block ``rows`` (64; 128 is experimental).
    """
    tokens, model_dim = a_q.shape
    experts, _, inter_dim = w2.shape
    if tile_n is None:
        tile_n = fmoe_tile_n(inter_dim, rows)
    if slices is None:
        slices = fmoe_slices(inter_dim, tile_n, rows)
    check_config(model_dim, inter_dim, tile_n, slices, rows)
    grid = fmoe_grid(a_q.device, inter_dim, tile_n, slices)
    constexprs = (model_dim, inter_dim, tile_n, slices, grid, rows)
    tensors = (
        out,
        a_q,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        a_scale,
        w1_scale,
        w2_scale,
        _semaphore(a_q.device),
    )
    stream = torch.cuda.current_stream(a_q.device)
    args = (*map(ptr_arg, tensors), tokens, experts, stream, *constexprs)
    compiled = _COMPILED.get(constexprs)
    if compiled is not None:
        compiled(*args)
        return out
    compiled = _compile_and_run(launch_fmoe_blockscale, *args)
    if compiled is not None:
        _COMPILED[constexprs] = compiled
    return out


__all__ = [
    "fmoe_blockscale_g1u1_flydsl",
    "fmoe_grid",
    "fmoe_slices",
    "fmoe_tile_n",
]
