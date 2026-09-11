# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import importlib
import itertools
import random
from dataclasses import dataclass

import pandas as pd
import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in ps decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16
# qdtype fp8, kdtype fp8: nhead16, nhead128
# qdtype fp8, kdtype bf16: nhead16


def check_support(dtype, kv_dtype, nhead):
    return not (dtype == dtypes.fp8 and kv_dtype == dtypes.bf16)


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    # RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    # amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8_q=False,
    is_fp8_kvc=False,
    q_scale=None,
    kv_scale=None,
):

    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))

    l = attn_weights_exp.sum(-1)

    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtype)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0, 1).unsqueeze(-1)

    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


def generate_topk_kv(
    kv_indptr: torch.Tensor,
    qo_len: int = 1,
    NUM_TOPK_TOKENS: int = 2048,
):
    batch_size = kv_indptr.shape[0] - 1
    batch_size = batch_size * qo_len
    token_indices = torch.empty([batch_size, NUM_TOPK_TOKENS], dtype=torch.int32)
    for i in range(batch_size):
        i_ori = i // qo_len
        kv_end = kv_indptr[i_ori + 1]
        kv_start = kv_indptr[i_ori]
        kv_len = kv_end - kv_start

        if kv_len < NUM_TOPK_TOKENS:
            token_indices[i, :kv_len] = torch.arange(0, kv_len, dtype=torch.int32)
        else:
            token_indices[i] = torch.randint(
                0, kv_len, (NUM_TOPK_TOKENS,), dtype=torch.int32
            )

    return token_indices


def sparse_kv_indptr_to_dense(
    kv_indptr: torch.Tensor,
    converted_indices: torch.Tensor,
    qo_len: int = 1,
    NUM_TOPK_TOKENS: int = 2048,
):
    new_kv_indptr = [0]
    indices_list = []
    batch_size = kv_indptr.shape[0] - 1
    batch_size = qo_len * batch_size
    for i in range(batch_size):
        i_ori = i // qo_len
        kv_len = kv_indptr[i_ori + 1] - kv_indptr[i_ori]
        kv_len = min(kv_len, NUM_TOPK_TOKENS)
        indices_list.append(converted_indices[i, :kv_len])
        new_kv_indptr.append(kv_len + new_kv_indptr[i])
    return (
        torch.arange(0, batch_size + 1, dtype=torch.int32),
        torch.tensor(new_kv_indptr, dtype=torch.int32),
        torch.concat(indices_list),
    )


@triton.jit
def _convert_req_index_to_global_index_kernel(
    kv_indptr,  # int32 [num_requests]
    kv_indices,  # int32 [num_requests * max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0: tl.constexpr,
    ti_stride0: tl.constexpr,
    ti_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    qo_len: tl.constexpr,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    batch_id = token_id // qo_len

    # Load request id for this token (no mask: grid is exact)
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_len = kv_end - kv_start

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = indice_id < kv_len
    # tl.device_print("offset", valid_block)
    base = tl.load(
        kv_indices + kv_start + block_id * bt_stride0, mask=valid_block, other=0
    )

    # base = 0

    # If token == -1 OR block_id OOB, output -1; else base * BLOCK_SIZE + offset
    out_val = tl.where(
        is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off
    )

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indices: torch.Tensor,  # int32 [total_kv_seqlen]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    qo_len: int = 1,
    BLOCK_SIZE: int = 1,  # page_block_size = 1 for now
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert kv_indices.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    # num_batches = kv_indptr.shape[0] - 1
    num_tokens = token_indices.shape[0]

    # num_requests, max_num_blocks_per_req = block_table.shape
    # max_num_blocks_per_req = 65536 * 32
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    kv_indptr_c = kv_indptr.contiguous()
    kv_indices_c = kv_indices.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0 = kv_indices_c.stride()[0]
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens x column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        kv_indptr_c,
        kv_indices_c,
        token_indices_c,
        out,
        # shapes / constexprs
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        qo_len,
    )
    return out


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    max_split_per_batch,
):
    ret = {}

    out_dtype = torch.bfloat16
    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(6, ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)

    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    kv_indices = torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    # max_seqlen_kv = seq_lens_kv.max().item()
    # total_qo = qo_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    # for none absorb (mha)
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # us_asm = None
    # if batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    torch.cuda.empty_cache()
    nhead_kv = 1

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    # troch implementation
    out_ref, _lse_ref = torch_mla_extend(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=True,
        dtype=dtype,
    )

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        dtype,
        kvtype,
        is_sparse=True,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
    )

    # aiter implementation
    # the tensor's meaning please refer aiter/ops/attention.py
    work_meta_data = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device="cuda"
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(
        work_info_set_size,
        dtype=work_info_set_type,
        device="cuda",
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
    )

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=page_size,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        topk=2048,
        dtype_q_nope=dtype,
        dtype_kv_nope=kvtype,
    )

    # generate kv topk per token & convert indices into per token
    token_indices = generate_topk_kv(kv_indptr, decode_qlen)
    converted_indices = triton_convert_req_index_to_global_index(
        kv_indptr,
        kv_indices,
        token_indices,
        decode_qlen,
    )

    # convert kv indptr perbatch into pertoken and calc ref
    new_qo_indptr, new_kv_indptr, new_indices = sparse_kv_indptr_to_dense(
        kv_indptr,
        converted_indices,
        decode_qlen,
    )
    total_kv = new_kv_indptr[-1].item()  # change into pertoken total_kv
    out_ref, _lse_ref = torch_mla_extend(
        q,
        kv_buffer,
        new_qo_indptr,
        new_kv_indptr,
        new_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=False,
        dtype=out_dtype,
    )

    def test_sparse_mla_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        (_attn_logits, _attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            # new_kv_indptr,
            converted_indices.view(-1),
            kv_last_page_lens,
            1,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    def test_sparse_mla_fp8():
        # if dtype != dtypes.fp8 and nhead == 128:
        #     aiter.logger.info("don't support this case:\n")
        #     return None, 1e12

        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtypes.fp8)
        q_scale = torch.ones([1], dtype=torch.float, device="cuda")

        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, _lse_ref_fp8 = torch_mla_extend(
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8,
            new_qo_indptr,
            new_kv_indptr,
            new_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

        (_attn_logits, _attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            converted_indices.view(-1),
            kv_last_page_lens,
            1,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            return_lse=False,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    err = None
    us_asm_decode = 1e12
    if dtype == torch.bfloat16 and kvtype == dtypes.bf16:
        err, us_asm_decode = test_sparse_mla_bf16()
    elif kvtype == dtypes.fp8:
        err, us_asm_decode = test_sparse_mla_fp8()
    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )

    ret["decode:flops"] = flops
    ret["decode:bytes"] = bytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    return ret


ADAPTIVE_SCENARIOS = {
    "mixed": ((1, 2, 4, 8), (1, 19, 2051, 4099)),
    "zero": ((0, 1, 4, 2), (0, 1, 2, 65)),
    "ragged": ((8, 4, 2, 1), (3, 17, 2053, 4097)),
    "grow": ((8, 8, 4, 2), (11, 257, 3073, 6145)),
}


def _quantize_fp8(values, scale):
    limit = torch.finfo(dtypes.fp8).max
    return torch.clamp(values / scale, -limit, limit).to(dtypes.fp8)


@dataclass
class AdaptiveMLADecodeFixture:
    max_rows: int
    num_heads: int
    topk: int
    max_context_lens: tuple[int, ...]
    q: torch.Tensor
    kv_buffer: torch.Tensor
    output: torch.Tensor
    query_start_loc: torch.Tensor
    decode_lens: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_lens: torch.Tensor
    q_scale: torch.Tensor
    kv_scale: torch.Tensor
    sentinel: float
    live_rows: int = 0
    selected_physical: tuple[torch.Tensor, ...] = ()

    @classmethod
    def create(cls, max_rows, num_heads, topk, max_context_lens, sentinel=-123.0):
        num_requests = len(max_context_lens)
        physical_tokens = sum(max_context_lens) + 17 * num_requests
        return cls(
            max_rows=max_rows,
            num_heads=num_heads,
            topk=topk,
            max_context_lens=tuple(max_context_lens),
            q=torch.empty((max_rows, num_heads, 576), dtype=dtypes.fp8),
            kv_buffer=torch.empty((physical_tokens, 1, 576), dtype=dtypes.fp8),
            output=torch.full(
                (max_rows, num_heads, 512), sentinel, dtype=torch.bfloat16
            ),
            query_start_loc=torch.zeros(num_requests + 1, dtype=torch.int32),
            decode_lens=torch.zeros(num_requests, dtype=torch.int32),
            qo_indptr=torch.arange(max_rows + 1, dtype=torch.int32),
            kv_indptr=torch.zeros(max_rows + 1, dtype=torch.int32),
            kv_indices=torch.full((max_rows * topk,), -1, dtype=torch.int32),
            kv_last_page_lens=torch.ones(max_rows, dtype=torch.int32),
            q_scale=torch.ones(1, dtype=torch.float32),
            kv_scale=torch.ones(1, dtype=torch.float32),
            sentinel=sentinel,
        )

    def load(self, widths, context_lens, seed):
        if len(widths) != self.decode_lens.numel():
            raise ValueError("width count must match request capacity")
        if len(context_lens) != len(widths):
            raise ValueError("context length count must match width count")
        if any(width < 0 for width in widths):
            raise ValueError("decode widths must be non-negative")
        if any(length < 0 for length in context_lens):
            raise ValueError("context lengths must be non-negative")
        if any(
            length > maximum
            for length, maximum in zip(context_lens, self.max_context_lens)
        ):
            raise ValueError("context length exceeds fixture capacity")
        live_rows = sum(widths)
        if live_rows > self.max_rows:
            raise ValueError("live rows exceed graph capacity")

        query_start_loc = [0]
        for width in widths:
            query_start_loc.append(query_start_loc[-1] + width)
        self.query_start_loc.copy_(
            torch.tensor(query_start_loc, dtype=torch.int32, device=self.q.device)
        )
        self.decode_lens.copy_(
            torch.tensor(widths, dtype=torch.int32, device=self.q.device)
        )

        q_scale = 0.03125 + (seed % 5) * 0.00390625
        kv_scale = 0.0234375 + (seed % 7) * 0.001953125
        self.q_scale.fill_(q_scale)
        self.kv_scale.fill_(kv_scale)

        generator = torch.Generator(device=self.q.device).manual_seed(seed)
        self.q.fill_(float("nan"))
        if live_rows:
            q_values = torch.randn(
                (live_rows, self.num_heads, 576),
                generator=generator,
                dtype=torch.float32,
                device=self.q.device,
            )
            self.q[:live_rows].copy_(_quantize_fp8(q_values, q_scale))

        self.kv_buffer.fill_(float("nan"))
        kv_values = torch.randn(
            self.kv_buffer.shape,
            generator=generator,
            dtype=torch.float32,
            device=self.kv_buffer.device,
        )
        self.kv_buffer.copy_(_quantize_fp8(kv_values, kv_scale))

        cpu_generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
        permutation = torch.randperm(
            self.kv_buffer.shape[0], generator=cpu_generator, device="cpu"
        ).tolist()
        request_pages = []
        offset = 0
        for context_len, maximum in zip(context_lens, self.max_context_lens):
            request_pages.append(permutation[offset : offset + context_len])
            offset += maximum + 17

        selected_physical = []
        row_indptr = [0]
        row_indices = []
        row = 0
        for width, context_len, pages in zip(widths, context_lens, request_pages):
            for token_index in range(width):
                causal_bound = max(context_len - width + token_index + 1, 0)
                selected_count = min(causal_bound, self.topk)
                if selected_count:
                    logical = torch.randperm(
                        causal_bound, generator=cpu_generator, device="cpu"
                    )[:selected_count]
                    physical = torch.tensor(
                        [pages[index] for index in logical.tolist()],
                        dtype=torch.int32,
                        device=self.q.device,
                    )
                else:
                    physical = torch.empty(0, dtype=torch.int32, device=self.q.device)
                selected_physical.append(physical)
                row_indices.extend(physical.cpu().tolist())
                row_indptr.append(len(row_indices))
                row += 1
        while row < self.max_rows:
            selected_physical.append(
                torch.empty(0, dtype=torch.int32, device=self.q.device)
            )
            row_indptr.append(len(row_indices))
            row += 1

        self.kv_indptr.copy_(
            torch.tensor(row_indptr, dtype=torch.int32, device=self.q.device)
        )
        self.kv_indices.fill_(-1)
        if row_indices:
            self.kv_indices[: len(row_indices)].copy_(
                torch.tensor(row_indices, dtype=torch.int32, device=self.q.device)
            )
        self.output.fill_(self.sentinel)
        self.live_rows = live_rows
        self.selected_physical = tuple(selected_physical)
        return self

    @property
    def selected_count(self):
        return int(self.kv_indptr[self.max_rows].item())


def build_adaptive_mla_fixture(
    widths,
    context_lens,
    num_heads,
    topk,
    max_rows=32,
    seed=1,
    max_context_lens=None,
):
    if max_context_lens is None:
        max_context_lens = context_lens
    fixture = AdaptiveMLADecodeFixture.create(
        max_rows=max_rows,
        num_heads=num_heads,
        topk=topk,
        max_context_lens=max_context_lens,
    )
    return fixture.load(widths, context_lens, seed)


def torch_mla_varlen_sparse_reference(fixture):
    output = torch.full_like(fixture.output, fixture.sentinel)
    q = fixture.q.float() * fixture.q_scale.float()
    kv = fixture.kv_buffer[:, 0].float() * fixture.kv_scale.float()
    sm_scale = 1.0 / (576**0.5)
    for row in range(fixture.live_rows):
        physical = fixture.selected_physical[row]
        if physical.numel() == 0:
            output[row].zero_()
            continue
        row_kv = kv.index_select(0, physical.to(torch.int64))
        scores = torch.matmul(q[row], row_kv.transpose(0, 1)) * sm_scale
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output[row].copy_(torch.matmul(probabilities, row_kv[:, :512]).to(output.dtype))
    return output


@dataclass
class _AdaptiveCandidate:
    name: str
    invoke: object
    scratch_bytes: object
    checks_empty_rows: bool
    graph_native: bool


def _aiter_flattened_candidate(fixture, num_kv_splits):
    selected_count = fixture.selected_count
    split_indptr = torch.arange(
        0,
        (fixture.live_rows + 1) * num_kv_splits,
        num_kv_splits,
        dtype=torch.int32,
        device=fixture.q.device,
    )

    def invoke():
        live_qo_indptr = fixture.qo_indptr[: fixture.live_rows + 1]
        live_kv_indptr = fixture.kv_indptr[: fixture.live_rows + 1]
        aiter.mla.mla_decode_fwd(
            fixture.q[: fixture.live_rows],
            fixture.kv_buffer.view(-1, 1, 1, 576),
            fixture.output[: fixture.live_rows],
            live_qo_indptr,
            live_kv_indptr,
            fixture.kv_indices[:selected_count],
            fixture.kv_last_page_lens[: fixture.live_rows],
            1,
            1,
            1,
            1.0 / (576**0.5),
            num_kv_splits=num_kv_splits,
            num_kv_splits_indptr=split_indptr,
            q_scale=fixture.q_scale,
            kv_scale=fixture.kv_scale,
            causal=False,
        )
        return fixture.output

    def scratch_bytes():
        return fixture.live_rows * num_kv_splits * fixture.num_heads * 513 * 4 * 2

    return _AdaptiveCandidate(
        name="aiter_flattened_asm_decode_merge",
        invoke=invoke,
        scratch_bytes=scratch_bytes,
        checks_empty_rows=False,
        graph_native=False,
    )


def _production_adaptive_candidate(fixture, num_kv_splits):
    try:
        module = importlib.import_module("aiter.ops.flydsl.mla_decode_varlen")
    except ModuleNotFoundError as error:
        if error.name == "aiter.ops.flydsl.mla_decode_varlen":
            return None
        raise
    function = module.mla_decode_varlen
    factory = getattr(module, "create_mla_decode_varlen_workspace", None)
    workspace = (
        factory(
            fixture.q,
            fixture.kv_buffer,
            fixture.output,
            num_kv_splits=num_kv_splits,
        )
        if factory is not None
        else None
    )

    def invoke():
        function(
            fixture.q,
            fixture.kv_buffer,
            fixture.output,
            fixture.query_start_loc,
            fixture.kv_indptr,
            fixture.kv_indices,
            sm_scale=1.0 / (576**0.5),
            q_scale=fixture.q_scale,
            kv_scale=fixture.kv_scale,
            workspace=workspace,
            num_kv_splits=num_kv_splits,
        )
        return fixture.output

    def scratch_bytes():
        estimator = getattr(module, "mla_decode_varlen_scratch_bytes", None)
        if estimator is None:
            return 0
        return int(estimator(workspace))

    return _AdaptiveCandidate(
        name="adaptive_varlen",
        invoke=invoke,
        scratch_bytes=scratch_bytes,
        checks_empty_rows=True,
        graph_native=True,
    )


def _validate_adaptive_output(fixture, reference, candidate, expected):
    if candidate.checks_empty_rows:
        checked_rows = torch.arange(fixture.live_rows, device=fixture.q.device)
    else:
        checked_rows = torch.tensor(
            [
                row
                for row in range(fixture.live_rows)
                if fixture.selected_physical[row].numel()
            ],
            dtype=torch.int64,
            device=fixture.q.device,
        )
    err = checkAllclose(
        reference.index_select(0, checked_rows).float(),
        candidate.index_select(0, checked_rows).float(),
        msg=expected,
    )
    if not torch.equal(candidate[fixture.live_rows :], reference[fixture.live_rows :]):
        raise AssertionError(f"{expected}: padded output rows were modified")
    return err


def _adaptive_work(fixture):
    flops = fixture.selected_count * fixture.num_heads * 2 * (576 + 512)
    io_bytes = (
        fixture.live_rows * fixture.num_heads * 576
        + fixture.selected_count * 576
        + fixture.live_rows * fixture.num_heads * 512 * 2
        + fixture.selected_count * 4
        + fixture.kv_indptr.nbytes
        + fixture.query_start_loc.nbytes
        + fixture.q_scale.nbytes
        + fixture.kv_scale.nbytes
    )
    return flops, io_bytes


def _run_graph_replay(candidate, fixture):
    replay_names = ("zero", "mixed", "grow", "ragged", "mixed")
    widths, context_lens = ADAPTIVE_SCENARIOS["zero"]
    fixture.load(widths, context_lens, 701)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        candidate.invoke()
    side_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    fixture.output.fill_(fixture.sentinel)
    with torch.cuda.graph(graph):
        candidate.invoke()
    previous = None
    for replay_index, name in enumerate(replay_names):
        widths, context_lens = ADAPTIVE_SCENARIOS[name]
        fixture.load(widths, context_lens, 809 + replay_index)
        reference = torch_mla_varlen_sparse_reference(fixture)
        graph.replay()
        torch.cuda.synchronize()
        _validate_adaptive_output(
            fixture,
            reference,
            fixture.output,
            f"{candidate.name}: graph replay {name}",
        )
        current = fixture.output.clone()
        if previous is not None and torch.equal(previous, current):
            raise AssertionError(
                "graph replay output did not change after input update"
            )
        previous = current
    return 0


@benchmark()
def test_mla_varlen_adaptive(
    scenario,
    nhead,
    topk,
    max_rows,
    num_kv_splits,
):
    widths, context_lens = ADAPTIVE_SCENARIOS[scenario]
    max_context_lens = tuple(
        max(values)
        for values in zip(*(item[1] for item in ADAPTIVE_SCENARIOS.values()))
    )
    fixture = build_adaptive_mla_fixture(
        widths,
        context_lens,
        nhead,
        topk,
        max_rows=max_rows,
        seed=101,
        max_context_lens=max_context_lens,
    )
    reference = torch_mla_varlen_sparse_reference(fixture)
    candidates = {}
    if all(fixture.selected_physical[row].numel() for row in range(fixture.live_rows)):
        flattened = _aiter_flattened_candidate(fixture, num_kv_splits)
        candidates[flattened.name] = flattened
    adaptive = _production_adaptive_candidate(fixture, num_kv_splits)
    if adaptive is not None:
        candidates[adaptive.name] = adaptive

    flops, io_bytes = _adaptive_work(fixture)
    ret = {
        "gfx": get_gfx(),
        "rows": fixture.live_rows,
        "selected": fixture.selected_count,
        "adaptive available": adaptive is not None,
    }
    for name, candidate in candidates.items():
        fixture.output.fill_(fixture.sentinel)
        output, microseconds = run_perftest(
            candidate.invoke,
            num_warmup=10,
            num_iters=51,
            use_cuda_event=True,
        )
        err = _validate_adaptive_output(
            fixture, reference, output, f"{name}: independent fp32 oracle"
        )
        first = output.clone()
        fixture.output.fill_(fixture.sentinel)
        candidate.invoke()
        torch.cuda.synchronize()
        if not torch.equal(first, fixture.output):
            raise AssertionError(f"{name}: repeated invocation is not bitwise stable")
        traffic_bytes = io_bytes + candidate.scratch_bytes()
        ret[f"{name} us"] = microseconds
        ret[f"{name} TFLOPS"] = flops / microseconds / 1e6
        ret[f"{name} TB/s"] = traffic_bytes / microseconds / 1e6
        ret[f"{name} err"] = err

    if adaptive is not None:
        ret["adaptive_varlen graph err"] = _run_graph_replay(adaptive, fixture)
    return ret


def main():
    if get_gfx() not in ("gfx942", "gfx950"):
        aiter.logger.warning(
            "adaptive sparse MLA unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="adaptive ragged sparse MLA decode correctness and performance",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        choices=tuple(ADAPTIVE_SCENARIOS),
        default=["mixed", "zero", "ragged"],
    )
    parser.add_argument("-n", "--nhead", type=int, nargs="*", default=[16, 32, 64, 128])
    parser.add_argument("--topk", type=int, nargs="*", default=[128, 2048])
    parser.add_argument("--max-rows", type=int, nargs="*", default=[32])
    parser.add_argument("--num-kv-splits", type=int, nargs="*", default=[8])
    args = parser.parse_args()

    rows = []
    for scenario, nhead, topk, max_rows, num_kv_splits in itertools.product(
        args.scenarios,
        args.nhead,
        args.topk,
        args.max_rows,
        args.num_kv_splits,
    ):
        rows.append(
            test_mla_varlen_adaptive(
                scenario,
                nhead,
                topk,
                max_rows,
                num_kv_splits,
            )
        )
    dataframe = pd.DataFrame(rows)
    aiter.logger.info(
        "adaptive sparse MLA summary (markdown):\n%s",
        dataframe.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
