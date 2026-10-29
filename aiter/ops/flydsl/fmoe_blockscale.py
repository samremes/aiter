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
from .kernels.moe_reduce import compile_moe_reduction
from .kernels.tensor_shim import _compile_and_run, _run_compiled, ptr_arg

set_slp_vectorize_once_used(launch_fmoe_blockscale, False)

# Compiled launchers by their Constexprs
# (D, I, cols, slices, grid, rows, ws, stored, admit).
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


def fmoe_ws(tokens: int, model_dim: int, inter_dim: int, rows: int = 64) -> int:
    """Default wave split: ping-pong (2) at 7168 x 256 from 2048 tokens, where
    hiding the output under the next tile's gate/up beats its 256-VGPR squeeze
    (4.4% at 16384 tokens, 5% at 2048; it loses 9% at 1024); else 0."""
    if rows == 64 and (model_dim, inter_dim) == (7168, 256) and tokens >= 2048:
        return 2
    return 0


def fmoe_slices(inter_dim: int, tile_n: int, rows: int = 64, ws: int = 0) -> int:
    """Column slices one tile runs: all of them where they fit, so a token row
    gets one routed partial per expert to merge; else the most that divide
    them and fit the LDS."""
    n = inter_dim // tile_n
    for s in range(min(n, MAX_SLICES[rows]), 0, -1):
        if n % s == 0 and lds_bytes(rows, tile_n, s, ws) <= LDS_MAX:
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
    ws: int | None = None,
    stored: int = 0,
    topk: int | None = None,
    admit: int = 0,
    reduce: bool = True,
) -> Tensor:
    """``out[t] += sum_k w_k * down(silu(gate(x_t)) * up(x_t))`` for sorted
    ``rows``-row blocks; ``out`` must be zeroed by the caller (unless
    ``stored == topk``).

    ``a_q`` ``[T, D]`` fp8 with ``a_scale`` ``[D/128, T]``; ``w1`` ``[E, 2I, D]``
    and ``w2`` ``[E, D, I]`` fp8 preshuffled (16, 16), with 128x128 block scales
    ``w1_scale`` ``[E, 2I/128, D/128]`` and ``w2_scale`` ``[E, D/128, I/128]``.
    The sort is ``moe_sorting`` with block ``rows`` (64; 128 is experimental).

    ``ws`` 2: ping-pong over eight waves (256 VGPRs a wave, 64 x 256 tiles):
    waves 0-3 run gate/up of tile i while waves 4-7 run down and the output
    of tile i - 1, so the output spreads over the whole tile. ``None``:
    per ``fmoe_ws``.
    ``stored``: routing slots (of ``topk``) whose rows the kernel stores into a
    ``[T, stored, D]`` bf16 partial buffer instead of adding them atomically;
    a reduce pass then adds them into ``out`` (``reduce=False`` skips it, for
    timing the kernel alone). With ``stored == topk`` the reduce writes
    ``out``, which then needs no zeroing.
    ``admit``: down-phase workgroups at once in grid / 16 (0: default).
    """
    tokens, model_dim = a_q.shape
    experts, _, inter_dim = w2.shape
    if ws is None:
        ws = fmoe_ws(tokens, model_dim, inter_dim, rows)
    if tile_n is None:
        tile_n = 256 if ws else fmoe_tile_n(inter_dim, rows)
    if slices is None:
        slices = (
            inter_dim // tile_n if stored else fmoe_slices(inter_dim, tile_n, rows, ws)
        )
    check_config(model_dim, inter_dim, tile_n, slices, rows, ws, stored)
    if stored:
        assert topk is not None and 0 < stored <= topk, "stored needs topk"
        partial = _partial(
            out.device, max(tokens * stored, sorted_ids.numel()) * model_dim
        )
    else:
        partial = out
    grid = fmoe_grid(a_q.device, inter_dim, tile_n, slices)
    constexprs = (model_dim, inter_dim, tile_n, slices, grid, rows, ws, stored, admit)
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
        partial,
    )
    stream = torch.cuda.current_stream(a_q.device)
    args = (*map(ptr_arg, tensors), tokens, experts, stream, *constexprs)
    compiled = _COMPILED.get(constexprs)
    if compiled is not None:
        compiled(*args)
    else:
        compiled = _compile_and_run(launch_fmoe_blockscale, *args)
        if compiled is not None:
            _COMPILED[constexprs] = compiled
    if stored and reduce:
        fmoe_reduce(partial, out, tokens, stored, accumulate=stored < topk)
    return out


def fmoe_reduce(
    partial: Tensor, out: Tensor, tokens: int, stored: int, accumulate: bool
) -> None:
    """``out (+)= partial[T, stored, D].sum(1)`` with aiter's FlyDSL topk reduce."""
    model_dim = out.shape[-1]
    exe = compile_moe_reduction(
        topk=stored, model_dim=model_dim, dtype_str="bf16", accumulate=accumulate
    )
    empty = _EMPTY_I32.get(out.device)
    if empty is None:
        empty = _EMPTY_I32[out.device] = torch.empty(
            0, dtype=torch.int32, device=out.device
        )
    stream = torch.cuda.current_stream(out.device)
    _run_compiled(
        exe,
        ptr_arg(partial),
        ptr_arg(out),
        ptr_arg(empty),
        ptr_arg(empty),
        ptr_arg(empty),
        tokens,
        stream,
    )


_EMPTY_I32: dict[torch.device, Tensor] = {}
_PARTIAL: dict[torch.device, Tensor] = {}


def _partial(device: torch.device, numel: int) -> Tensor:
    """The stored-partials scratch (bf16), grown on demand; every element the
    reduce reads is written by the kernel, so it is never zeroed."""
    buf = _PARTIAL.get(device)
    if buf is None or buf.numel() < numel:
        buf = _PARTIAL[device] = torch.empty(numel, dtype=torch.bfloat16, device=device)
    return buf


__all__ = [
    "fmoe_blockscale_g1u1_flydsl",
    "fmoe_grid",
    "fmoe_reduce",
    "fmoe_slices",
    "fmoe_tile_n",
    "fmoe_ws",
]
