from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from aiter import dtypes


@triton.jit
def _mla_decode_varlen_stage1(
    q,
    kv,
    query_start_loc,
    kv_indptr,
    kv_indices,
    q_scale,
    kv_scale,
    output,
    partial_output,
    partial_lse,
    sm_scale,
    num_requests: tl.constexpr,
    num_heads: tl.constexpr,
    num_splits: tl.constexpr,
    block_h: tl.constexpr,
    block_n: tl.constexpr,
    direct_output: tl.constexpr,
):
    row = tl.program_id(0)
    head_block = tl.program_id(1)
    split = tl.program_id(2)
    live_rows = tl.load(query_start_loc + num_requests)
    if row < live_rows:
        row_begin = tl.load(kv_indptr + row)
        row_end = tl.load(kv_indptr + row + 1)
        row_length = row_end - row_begin
        split_length = tl.cdiv(row_length, num_splits)
        split_begin = row_begin + split * split_length
        split_end = tl.minimum(split_begin + split_length, row_end)
        heads = head_block * block_h + tl.arange(0, block_h)
        latent = tl.arange(0, 512)
        head_mask = heads < num_heads
        if row_length == 0:
            if direct_output:
                tl.store(
                    output
                    + (row * num_heads + heads[:, None]) * 512
                    + latent[None, :],
                    0.0,
                    mask=head_mask[:, None],
                )
        if split_begin < split_end:
            rope = tl.arange(0, 64)
            q_base = (row * num_heads + heads[:, None]) * 576
            q_latent = tl.load(
                q + q_base + latent[None, :], mask=head_mask[:, None], other=0.0
            )
            q_rope = tl.load(
                q + q_base + 512 + rope[None, :],
                mask=head_mask[:, None],
                other=0.0,
            )
            scale_qk = tl.load(q_scale) * tl.load(kv_scale) * sm_scale
            scale_v = tl.load(kv_scale)
            maximum = tl.full([block_h], -float("inf"), tl.float32)
            denominator = tl.zeros([block_h], tl.float32)
            accumulator = tl.zeros([block_h, 512], tl.float32)

            for tile_begin in range(split_begin, split_end, block_n):
                positions = tile_begin + tl.arange(0, block_n)
                token_mask = positions < split_end
                physical = tl.load(
                    kv_indices + positions, mask=token_mask, other=0
                ).to(tl.int64)
                k_latent = tl.load(
                    kv + physical[None, :] * 576 + latent[:, None],
                    mask=token_mask[None, :],
                    other=0.0,
                )
                k_rope = tl.load(
                    kv + physical[None, :] * 576 + 512 + rope[:, None],
                    mask=token_mask[None, :],
                    other=0.0,
                )
                scores = tl.dot(q_latent, k_latent)
                scores += tl.dot(q_rope, k_rope)
                scores *= scale_qk
                scores = tl.where(
                    head_mask[:, None] & token_mask[None, :],
                    scores,
                    -float("inf"),
                )
                next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
                old_weight = tl.exp(maximum - next_maximum)
                probabilities = tl.exp(scores - next_maximum[:, None])
                values = tl.load(
                    kv + physical[:, None] * 576 + latent[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                )
                accumulator *= old_weight[:, None]
                accumulator += tl.dot(
                    probabilities.to(tl.bfloat16), values.to(tl.bfloat16)
                )
                denominator = denominator * old_weight + tl.sum(
                    probabilities, axis=1
                )
                maximum = next_maximum

            result = accumulator / denominator[:, None] * scale_v
            if direct_output:
                tl.store(
                    output
                    + (row * num_heads + heads[:, None]) * 512
                    + latent[None, :],
                    result,
                    mask=head_mask[:, None],
                )
            else:
                partial_base = (
                    ((row * num_heads + heads[:, None]) * num_splits + split)
                    * 512
                )
                tl.store(
                    partial_output + partial_base + latent[None, :],
                    result,
                    mask=head_mask[:, None],
                )
                tl.store(
                    partial_lse + (row * num_heads + heads) * num_splits + split,
                    maximum + tl.log(denominator),
                    mask=head_mask,
                )


@triton.jit
def _mla_decode_varlen_stage2(
    output,
    query_start_loc,
    kv_indptr,
    partial_output,
    partial_lse,
    num_requests: tl.constexpr,
    num_heads: tl.constexpr,
    num_splits: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    live_rows = tl.load(query_start_loc + num_requests)
    if row < live_rows:
        row_begin = tl.load(kv_indptr + row)
        row_end = tl.load(kv_indptr + row + 1)
        latent = tl.arange(0, 512)
        output_base = (row * num_heads + head) * 512
        if row_begin == row_end:
            tl.store(output + output_base + latent, 0.0)
        else:
            row_length = row_end - row_begin
            split_length = tl.cdiv(row_length, num_splits)
            maximum = tl.full([], -float("inf"), tl.float32)
            denominator = tl.zeros([], tl.float32)
            accumulator = tl.zeros([512], tl.float32)
            for split in range(num_splits):
                split_begin = row_begin + split * split_length
                if split_begin < row_end:
                    offset = (row * num_heads + head) * num_splits + split
                    split_lse = tl.load(partial_lse + offset)
                    next_maximum = tl.maximum(maximum, split_lse)
                    old_weight = tl.exp(maximum - next_maximum)
                    new_weight = tl.exp(split_lse - next_maximum)
                    partial = tl.load(
                        partial_output + offset * 512 + latent
                    )
                    accumulator = accumulator * old_weight + partial * new_weight
                    denominator = denominator * old_weight + new_weight
                    maximum = next_maximum
            tl.store(output + output_base + latent, accumulator / denominator)


@dataclass
class MlaDecodeVarlenWorkspace:
    max_rows: int
    num_heads: int
    num_kv_splits: int
    partial_output: torch.Tensor
    partial_lse: torch.Tensor


def create_mla_decode_varlen_workspace(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    *,
    num_kv_splits: int | None = None,
) -> MlaDecodeVarlenWorkspace:
    del kv_buffer
    max_rows, num_heads, _ = q.shape
    if num_kv_splits is None:
        num_kv_splits = 8
    num_kv_splits = int(num_kv_splits)
    if num_kv_splits < 1:
        raise ValueError("`num_kv_splits` must be positive")
    if tuple(o.shape) != (max_rows, num_heads, 512):
        raise ValueError("`o` must have shape [max_rows, num_heads, 512]")
    partial_shape = (
        (max_rows, num_heads, num_kv_splits, 512)
        if num_kv_splits > 1
        else (0,)
    )
    lse_shape = (
        (max_rows, num_heads, num_kv_splits) if num_kv_splits > 1 else (0,)
    )
    return MlaDecodeVarlenWorkspace(
        max_rows=max_rows,
        num_heads=num_heads,
        num_kv_splits=num_kv_splits,
        partial_output=torch.empty(partial_shape, dtype=torch.float32, device=q.device),
        partial_lse=torch.empty(lse_shape, dtype=torch.float32, device=q.device),
    )


def _validate_inputs(
    q,
    kv_buffer,
    o,
    query_start_loc,
    kv_indptr,
    kv_indices,
    q_scale,
    kv_scale,
    workspace,
    num_kv_splits,
):
    if q.dtype != dtypes.fp8 or kv_buffer.dtype != dtypes.fp8:
        raise TypeError("`q` and `kv_buffer` must use the AITER FP8 dtype")
    if o.dtype != torch.bfloat16:
        raise TypeError("`o` must be bfloat16")
    if q.ndim != 3 or q.shape[-1] != 576:
        raise ValueError("`q` must have shape [max_rows, num_heads, 576]")
    if kv_buffer.ndim not in (3, 4) or kv_buffer.shape[-1] != 576:
        raise ValueError("`kv_buffer` must have packed width 576")
    if query_start_loc.dtype != torch.int32 or query_start_loc.ndim != 1:
        raise TypeError("`query_start_loc` must be one-dimensional int32")
    if kv_indptr.dtype != torch.int32 or kv_indptr.shape != (q.shape[0] + 1,):
        raise ValueError("`kv_indptr` must be int32 with max_rows + 1 entries")
    if kv_indices.dtype != torch.int32 or kv_indices.ndim != 1:
        raise TypeError("`kv_indices` must be one-dimensional int32")
    if q_scale is None or kv_scale is None:
        raise ValueError("FP8 inputs require `q_scale` and `kv_scale`")
    if q_scale.dtype != torch.float32 or q_scale.numel() != 1:
        raise ValueError("`q_scale` must be a scalar float32 tensor")
    if kv_scale.dtype != torch.float32 or kv_scale.numel() != 1:
        raise ValueError("`kv_scale` must be a scalar float32 tensor")
    if workspace.max_rows != q.shape[0] or workspace.num_heads != q.shape[1]:
        raise ValueError("workspace shape does not match `q`")
    if workspace.num_kv_splits != num_kv_splits:
        raise ValueError("workspace split count does not match `num_kv_splits`")


def mla_decode_varlen(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    workspace: MlaDecodeVarlenWorkspace | None = None,
    num_kv_splits: int | None = None,
):
    if num_kv_splits is None:
        num_kv_splits = workspace.num_kv_splits if workspace is not None else 8
    num_kv_splits = int(num_kv_splits)
    if workspace is None:
        workspace = create_mla_decode_varlen_workspace(
            q, kv_buffer, o, num_kv_splits=num_kv_splits
        )
    _validate_inputs(
        q,
        kv_buffer,
        o,
        query_start_loc,
        kv_indptr,
        kv_indices,
        q_scale,
        kv_scale,
        workspace,
        num_kv_splits,
    )
    if sm_scale is None:
        sm_scale = 1.0 / (576**0.5)
    block_h = 16
    block_n = 32
    num_requests = query_start_loc.numel() - 1
    _mla_decode_varlen_stage1[
        (q.shape[0], triton.cdiv(q.shape[1], block_h), num_kv_splits)
    ](
        q,
        kv_buffer,
        query_start_loc,
        kv_indptr,
        kv_indices,
        q_scale,
        kv_scale,
        o,
        workspace.partial_output,
        workspace.partial_lse,
        float(sm_scale),
        num_requests=num_requests,
        num_heads=q.shape[1],
        num_splits=num_kv_splits,
        block_h=block_h,
        block_n=block_n,
        direct_output=num_kv_splits == 1,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    if num_kv_splits > 1:
        _mla_decode_varlen_stage2[(q.shape[0], q.shape[1])](
            o,
            query_start_loc,
            kv_indptr,
            workspace.partial_output,
            workspace.partial_lse,
            num_requests=num_requests,
            num_heads=q.shape[1],
            num_splits=num_kv_splits,
            num_warps=8,
            num_stages=1,
            waves_per_eu=2,
        )
    return o


def mla_decode_varlen_scratch_bytes(workspace: MlaDecodeVarlenWorkspace) -> int:
    return workspace.partial_output.nbytes + workspace.partial_lse.nbytes


__all__ = [
    "MlaDecodeVarlenWorkspace",
    "create_mla_decode_varlen_workspace",
    "mla_decode_varlen",
    "mla_decode_varlen_scratch_bytes",
]
