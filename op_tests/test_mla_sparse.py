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
    "mixed": ((1, 2, 4, 5, 8), (1, 19, 2051, 3073, 4099)),
    "zero": ((0, 1, 4, 2, 0), (0, 1, 2, 65, 0)),
    "ragged": ((8, 4, 2, 1, 5), (3, 17, 2053, 4097, 129)),
    "grow": ((8, 8, 4, 2, 5), (11, 257, 3073, 6145, 2048)),
    "empty_batch": ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
}


def _quantize_fp8(values, scale):
    limit = torch.finfo(dtypes.fp8).max
    return torch.clamp(values / scale, -limit, limit).to(dtypes.fp8)


@dataclass
class AdaptiveMLADecodeFixture:
    max_rows: int
    num_heads: int
    topk: int
    mapping: str
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
    selected_logical: tuple[torch.Tensor, ...] = ()

    @classmethod
    def create(
        cls,
        max_rows,
        num_heads,
        topk,
        max_context_lens,
        mapping="page64",
        sentinel=-123.0,
    ):
        num_requests = len(max_context_lens)
        if mapping == "page64":
            physical_pages = (
                sum((length + 63) // 64 + 17 for length in max_context_lens) + 1
            )
            physical_tokens = physical_pages * 64
        elif mapping == "token_shuffle":
            physical_tokens = sum(max_context_lens) + 17 * num_requests + 1
        else:
            raise ValueError(f"unsupported mapping: {mapping}")
        return cls(
            max_rows=max_rows,
            num_heads=num_heads,
            topk=topk,
            mapping=mapping,
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
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
        request_pages = []
        offset = 0
        if self.mapping == "page64":
            permutation = torch.randperm(
                self.kv_buffer.shape[0] // 64 - 1,
                generator=cpu_generator,
                device="cpu",
            ).tolist()
            for context_len, maximum in zip(context_lens, self.max_context_lens):
                maximum_blocks = (maximum + 63) // 64
                block_table = permutation[offset : offset + maximum_blocks]
                request_pages.append(
                    [
                        block_table[position // 64] * 64 + position % 64
                        for position in range(context_len)
                    ]
                )
                offset += maximum_blocks + 17
        else:
            permutation = torch.randperm(
                self.kv_buffer.shape[0] - 1,
                generator=cpu_generator,
                device="cpu",
            ).tolist()
            for context_len, maximum in zip(context_lens, self.max_context_lens):
                request_pages.append(permutation[offset : offset + context_len])
                offset += maximum + 17
        active_pages = [page for pages in request_pages for page in pages]
        if active_pages:
            active_page_tensor = torch.tensor(
                active_pages, dtype=torch.int64, device=self.kv_buffer.device
            )
            kv_values = torch.full(
                self.kv_buffer.shape,
                float("nan"),
                dtype=torch.float32,
                device=self.kv_buffer.device,
            )
            active_values = torch.randn(
                (len(active_pages), 1, 576),
                generator=generator,
                dtype=torch.float32,
                device=self.kv_buffer.device,
            )
            kv_values.index_copy_(0, active_page_tensor, active_values)
            self.kv_buffer.copy_(_quantize_fp8(kv_values, kv_scale))

        selected_physical = []
        selected_logical = []
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
                    logical = torch.empty(0, dtype=torch.int64, device="cpu")
                    physical = torch.empty(0, dtype=torch.int32, device=self.q.device)
                selected_logical.append(logical)
                selected_physical.append(physical)
                row_indices.extend(physical.cpu().tolist())
                row_indptr.append(len(row_indices))
                row += 1
        while row < self.max_rows:
            selected_physical.append(
                torch.empty(0, dtype=torch.int32, device=self.q.device)
            )
            selected_logical.append(torch.empty(0, dtype=torch.int64, device="cpu"))
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
        self.selected_logical = tuple(selected_logical)
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
    mapping="page64",
):
    if max_context_lens is None:
        max_context_lens = context_lens
    fixture = AdaptiveMLADecodeFixture.create(
        max_rows=max_rows,
        num_heads=num_heads,
        topk=topk,
        max_context_lens=max_context_lens,
        mapping=mapping,
    )
    return fixture.load(widths, context_lens, seed)


def torch_mla_varlen_sparse_reference(fixture, sm_scale=None):
    output = torch.full_like(fixture.output, fixture.sentinel)
    q = fixture.q.float() * fixture.q_scale.float()
    kv = fixture.kv_buffer[:, 0].float() * fixture.kv_scale.float()
    if sm_scale is None:
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


def _adaptive_error_metrics(reference, output):
    reference_fp32 = reference.float()
    output_fp32 = output.float()
    difference = output_fp32 - reference_fp32
    finite = torch.isfinite(output_fp32)
    tolerance = 0.01 + 0.01 * reference_fp32.abs()
    return {
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "mismatch": int((difference.abs() > tolerance).sum().item()),
        "nonfinite": int((~finite).sum().item()),
    }


def _build_adaptive_cancellation_fixture(num_heads):
    fixture = build_adaptive_mla_fixture(
        (1,),
        (2,),
        num_heads,
        2,
        max_rows=1,
        seed=211,
        max_context_lens=(2,),
        mapping="page64",
    )
    fixture.q_scale.fill_(1.0)
    fixture.kv_scale.fill_(1.0)
    fixture.q.zero_()
    fixture.q[:, :, 0].fill_(1.0)
    selected = fixture.selected_physical[0].to(torch.int64)
    positive = torch.ones((1, 1, 576), dtype=torch.float32, device=fixture.q.device)
    negative = -positive
    positive[:, :, 0].fill_(0.375)
    negative[:, :, 0].fill_(-0.375)
    values = torch.cat((positive, negative), dim=0).to(dtypes.fp8)
    for source_row, physical_row in enumerate(selected.cpu().tolist()):
        fixture.kv_buffer[physical_row].copy_(values[source_row])
    fixture.output.fill_(fixture.sentinel)
    return fixture


def _build_adaptive_dominant_tail_fixture(num_heads):
    fixture = build_adaptive_mla_fixture(
        (1,),
        (2048,),
        num_heads,
        2048,
        max_rows=1,
        seed=223,
        max_context_lens=(2048,),
        mapping="page64",
    )
    fixture.q_scale.fill_(1.0)
    fixture.kv_scale.fill_(1.0)
    fixture.q.zero_()
    fixture.q[:, :, -1].fill_(24.0)
    selected = fixture.selected_physical[0].cpu().tolist()
    tail = torch.full((1, 576), 4.0, dtype=torch.float32, device=fixture.q.device)
    tail[:, -1].fill_(-0.5)
    tail = tail.to(dtypes.fp8)
    fixture.kv_buffer[selected[0]].zero_()
    for physical_row in selected[1:]:
        fixture.kv_buffer[physical_row].copy_(tail)
    fixture.output.fill_(fixture.sentinel)
    return fixture


@dataclass
class _AdaptiveCandidate:
    name: str
    invoke: object
    scratch_bytes: object
    checks_empty_rows: bool
    graph_native: bool
    executed_heads: int
    executed_qk_heads: int
    executed_pv_heads: int
    executed_splits: object


def _adaptive_candidate(fixture, num_kv_splits, module_name, candidate_name):
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError):
        return None
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

    def executed_splits():
        if workspace is not None and hasattr(workspace, "row_splits"):
            return workspace.row_splits[: fixture.live_rows].cpu().tolist()
        if workspace is not None and hasattr(workspace, "stage_splits"):
            return workspace.stage_splits
        return num_kv_splits

    if candidate_name == "flydsl_adaptive" and fixture.num_heads == 8:
        executed_qk_heads = fixture.num_heads
        executed_pv_heads = fixture.num_heads
    elif candidate_name == "flydsl_adaptive":
        value_groups = 8 // (min(fixture.num_heads, 64) // 16)
        executed_qk_heads = fixture.num_heads * value_groups
        executed_pv_heads = 2 * fixture.num_heads
    else:
        executed_qk_heads = fixture.num_heads
        executed_pv_heads = fixture.num_heads

    return _AdaptiveCandidate(
        name=candidate_name,
        invoke=invoke,
        scratch_bytes=scratch_bytes,
        checks_empty_rows=True,
        graph_native=True,
        executed_heads=fixture.num_heads,
        executed_qk_heads=executed_qk_heads,
        executed_pv_heads=executed_pv_heads,
        executed_splits=executed_splits,
    )


def _validate_adaptive_output(fixture, reference, output, candidate, expected):
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
        output.index_select(0, checked_rows).float(),
        msg=expected,
    )
    if err != 0:
        raise AssertionError(f"{expected}: error ratio {err:.6f}")
    request_offsets = fixture.query_start_loc.cpu().tolist()
    for request_index, (begin, end) in enumerate(itertools.pairwise(request_offsets)):
        request_err = checkAllclose(
            reference[begin:end].float(),
            output[begin:end].float(),
            msg=f"{expected}: request {request_index}",
        )
        if request_err != 0:
            raise AssertionError(
                f"{expected}: request {request_index} error ratio " f"{request_err:.6f}"
            )
    if not torch.equal(output[fixture.live_rows :], reference[fixture.live_rows :]):
        raise AssertionError(f"{expected}: padded output rows were modified")
    return err


def _validate_adaptive_fixture(
    fixture, widths, context_lens, allow_nonempty_padding=False
):
    expected_lens = torch.tensor(widths, dtype=torch.int32, device=fixture.q.device)
    if not torch.equal(torch.diff(fixture.query_start_loc), expected_lens):
        raise AssertionError("query_start_loc does not encode decode widths")
    if fixture.live_rows != sum(widths):
        raise AssertionError("packed live-row count is inconsistent")
    if not allow_nonempty_padding and not torch.equal(
        fixture.kv_indptr[fixture.live_rows :],
        fixture.kv_indptr[fixture.live_rows].expand(
            fixture.max_rows + 1 - fixture.live_rows
        ),
    ):
        raise AssertionError("padded CSR rows are not empty")
    if (
        fixture.live_rows < fixture.max_rows
        and not torch.isnan(fixture.q[fixture.live_rows :].float()).all()
    ):
        raise AssertionError("padded query rows are not poisoned")
    if (
        fixture.live_rows
        and not torch.isfinite(fixture.q[: fixture.live_rows].float()).all()
    ):
        raise AssertionError("live query rows contain poison values")
    if not torch.isnan(fixture.kv_buffer.float()).any():
        raise AssertionError("unused KV pages are not poisoned")
    selected_count = int(fixture.kv_indptr[fixture.live_rows].item())
    if selected_count:
        selected = fixture.kv_indices[:selected_count].to(torch.int64)
        if not torch.isfinite(
            fixture.kv_buffer.index_select(0, selected).float()
        ).all():
            raise AssertionError("selected KV pages contain poison values")

    expected_counts = []
    request_sets = []
    row = 0
    for width, context_len in zip(widths, context_lens):
        request_set = set()
        for token_index in range(width):
            causal_bound = max(context_len - width + token_index + 1, 0)
            expected_count = min(causal_bound, fixture.topk)
            expected_counts.append(expected_count)
            logical = fixture.selected_logical[row]
            if logical.numel() and int(logical.max()) >= causal_bound:
                raise AssertionError("selected logical position exceeds causal bound")
            request_set.update(fixture.selected_physical[row].cpu().tolist())
            row += 1
        request_sets.append(request_set)
    actual_counts = torch.diff(fixture.kv_indptr[: fixture.live_rows + 1]).cpu()
    if actual_counts.tolist() != expected_counts:
        raise AssertionError("CSR row lengths do not match causal top-k bounds")
    for left in range(len(request_sets)):
        for right in range(left + 1, len(request_sets)):
            if request_sets[left].intersection(request_sets[right]):
                raise AssertionError("requests share physical cache pages")
    nonempty_rows = [
        tuple(sorted(indices.cpu().tolist()))
        for indices in fixture.selected_physical[: fixture.live_rows]
        if indices.numel()
    ]
    if len(nonempty_rows) != len(set(nonempty_rows)):
        raise AssertionError("query rows do not have unique selected sets")


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
    replay_names = ("empty_batch", "mixed", "grow", "zero", "ragged", "mixed")
    widths, context_lens = ADAPTIVE_SCENARIOS["zero"]
    groups = fixture.decode_lens.numel() // len(widths)
    widths = widths * groups
    context_lens = context_lens * groups
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
        widths = widths * groups
        context_lens = context_lens * groups
        fixture.load(widths, context_lens, 809 + replay_index)
        poison_padding = replay_index % 2 == 1 and fixture.live_rows < fixture.max_rows
        if poison_padding:
            base = int(fixture.kv_indptr[fixture.live_rows].item())
            padding_rows = fixture.max_rows - fixture.live_rows
            fixture.kv_indptr[fixture.live_rows + 1 :].copy_(
                torch.arange(
                    base + 1,
                    base + padding_rows + 1,
                    dtype=torch.int32,
                    device=fixture.q.device,
                )
            )
            fixture.kv_indices[base : base + padding_rows].fill_(
                fixture.kv_buffer.shape[0] - 1
            )
        _validate_adaptive_fixture(
            fixture,
            widths,
            context_lens,
            allow_nonempty_padding=poison_padding,
        )
        reference = torch_mla_varlen_sparse_reference(fixture)
        graph.replay()
        torch.cuda.synchronize()
        _validate_adaptive_output(
            fixture,
            reference,
            fixture.output,
            candidate,
            f"{candidate.name}: graph replay {name}",
        )
        current = fixture.output.clone()
        if previous is not None and torch.equal(previous, current):
            raise AssertionError(
                "graph replay output did not change after input update"
            )
        previous = current
    return 0


def _run_packed_graph_perftest(candidate, packed_launches):
    candidate.invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(packed_launches):
            candidate.invoke()

    def replay():
        graph.replay()

    _, packed_microseconds = run_perftest(
        replay,
        num_warmup=5,
        num_iters=31,
        num_rotate_args=1,
        use_cuda_event=True,
    )
    return packed_microseconds / packed_launches


@benchmark()
def test_mla_varlen_adaptive(
    scenario,
    nhead,
    topk,
    max_rows,
    num_kv_splits,
    groups,
    mapping,
):
    widths, context_lens = ADAPTIVE_SCENARIOS[scenario]
    widths = widths * groups
    context_lens = context_lens * groups
    base_max_context_lens = tuple(
        max(values)
        for values in zip(*(item[1] for item in ADAPTIVE_SCENARIOS.values()))
    )
    max_context_lens = base_max_context_lens * groups
    fixture = build_adaptive_mla_fixture(
        widths,
        context_lens,
        nhead,
        topk,
        max_rows=max_rows * groups,
        seed=101,
        max_context_lens=max_context_lens,
        mapping=mapping,
    )
    _validate_adaptive_fixture(fixture, widths, context_lens)
    reference = torch_mla_varlen_sparse_reference(fixture)
    candidates = {}
    adaptive = _adaptive_candidate(
        fixture,
        num_kv_splits,
        "aiter.ops.flydsl.mla_decode_varlen",
        "flydsl_adaptive",
    )
    if adaptive is not None:
        candidates[adaptive.name] = adaptive
    triton_comparison = _adaptive_candidate(
        fixture,
        num_kv_splits,
        "aiter.ops.triton.mla_decode_varlen",
        "triton_comparison",
    )
    if triton_comparison is not None:
        candidates[triton_comparison.name] = triton_comparison

    flops, io_bytes = _adaptive_work(fixture)
    ret = {
        "gfx": get_gfx(),
        "rows": fixture.live_rows,
        "selected": fixture.selected_count,
        "working set": "hot-fixed",
        "adaptive available": adaptive is not None,
    }
    if not candidates:
        ret["status"] = "SKIP: no adaptive backend available"
        aiter.logger.warning(ret["status"])
        return ret
    for name, candidate in candidates.items():
        fixture.output.fill_(fixture.sentinel)
        output = candidate.invoke()
        torch.cuda.synchronize()
        err = _validate_adaptive_output(
            fixture,
            reference,
            output,
            candidate,
            f"{name}: independent fp32 oracle",
        )
        first = output.clone()
        fixture.output.fill_(fixture.sentinel)
        candidate.invoke()
        torch.cuda.synchronize()
        repeated_metrics = _adaptive_error_metrics(reference, fixture.output)
        if repeated_metrics["mismatch"] or repeated_metrics["nonfinite"]:
            raise AssertionError(
                f"{name}: repeated invocation failed oracle: {repeated_metrics}"
            )
        ret[f"{name} repeat equal"] = torch.equal(first, fixture.output)
        graph_times = {
            packed_launches: _run_packed_graph_perftest(candidate, packed_launches)
            for packed_launches in (16, 32, 64)
        }
        microseconds = graph_times[64]
        traffic_bytes = io_bytes
        ret[f"{name} us"] = microseconds
        for packed_launches, graph_microseconds in graph_times.items():
            ret[f"{name} graph{packed_launches} us"] = graph_microseconds
        ret[f"{name} TFLOPS"] = flops / microseconds / 1e6
        ret[f"{name} TB/s"] = traffic_bytes / microseconds / 1e6
        ret[f"{name} bytes"] = traffic_bytes
        ret[f"{name} scratch allocation bytes"] = candidate.scratch_bytes()
        ret[f"{name} executed heads"] = candidate.executed_heads
        ret[f"{name} executed QK heads"] = candidate.executed_qk_heads
        ret[f"{name} executed PV heads"] = candidate.executed_pv_heads
        ret[f"{name} executed splits"] = (
            candidate.executed_splits()
            if callable(candidate.executed_splits)
            else candidate.executed_splits
        )
        ret[f"{name} err"] = err

    if adaptive is not None:
        ret["flydsl_adaptive graph err"] = _run_graph_replay(adaptive, fixture)
    return ret


def _load_flydsl_adaptive_module():
    if get_gfx() != "gfx950":
        return None, f"FlyDSL adaptive MLA requires gfx950, got {get_gfx()}"
    try:
        module = importlib.import_module("aiter.ops.flydsl.mla_decode_varlen")
    except (ImportError, OSError) as error:
        return None, f"FlyDSL adaptive MLA unavailable: {error}"
    return module, None


def _run_adaptive_multiwave_auto_checks(module):
    fixture = AdaptiveMLADecodeFixture.create(
        max_rows=272,
        num_heads=8,
        topk=2048,
        max_context_lens=(2056,) * 33,
        mapping="page64",
    )
    workspace = module.create_mla_decode_varlen_workspace(
        fixture.q, fixture.kv_buffer, fixture.output, num_kv_splits=None
    )

    def invoke():
        module.mla_decode_varlen(
            fixture.q,
            fixture.kv_buffer,
            fixture.output,
            fixture.query_start_loc,
            fixture.kv_indptr,
            fixture.kv_indices,
            q_scale=fixture.q_scale,
            kv_scale=fixture.kv_scale,
            workspace=workspace,
            num_kv_splits=None,
        )

    fixture.load((8,) * 32 + (0,), (2056,) * 33, seed=701)
    invoke()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()

    for case_index, (live_rows, empty_context) in enumerate(
        (
            (256, False),
            (65, False),
            (0, False),
            (128, False),
            (64, False),
            (257, False),
            (256, True),
            (128, False),
        )
    ):
        widths = tuple(min(max(live_rows - request * 8, 0), 8) for request in range(33))
        fixture.load(widths, ((0 if empty_context else 2056),) * 33, 709 + case_index)
        reference = torch_mla_varlen_sparse_reference(fixture)
        for replay_index in range(4):
            fixture.output.fill_(fixture.sentinel)
            workspace.row_splits.fill_(-1)
            workspace.work_indptr.fill_(-1)
            workspace.finalize_count.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            metrics = _adaptive_error_metrics(reference, fixture.output)
            if metrics["mismatch"] or metrics["nonfinite"]:
                raise AssertionError(
                    f"multiwave auto R{live_rows} empty={empty_context} "
                    f"replay={replay_index} failed: {metrics}"
                )
            if not torch.equal(fixture.output[live_rows:], reference[live_rows:]):
                raise AssertionError("multiwave auto graph modified output padding")
            splits = workspace.row_splits[:live_rows].cpu()
            if live_rows:
                if empty_context:
                    if torch.count_nonzero(splits).item():
                        raise AssertionError("empty rows received attention tasks")
                elif (
                    splits.unique().numel() != 1
                    or splits.min().item() < 1
                    or splits.max().item() > workspace.stage_splits
                ):
                    raise AssertionError(
                        f"identical CSR lengths received inconsistent auto splits: {splits}"
                    )
            if workspace.work_indptr[0].item() != splits.sum().item():
                raise AssertionError("compact attention task count does not match rows")
            expected_finalize = int((splits != 1).sum().item()) * fixture.num_heads
            if workspace.finalize_count.item() != expected_finalize:
                raise AssertionError("compact finalize count does not match rows")
        aiter.logger.info(
            "adaptive multiwave auto R%d empty=%s: four graph replays passed",
            live_rows,
            empty_context,
        )


def _run_adaptive_validation_checks(module):
    _run_adaptive_multiwave_auto_checks(module)
    fixture = build_adaptive_mla_fixture(
        (1,),
        (64,),
        16,
        64,
        max_rows=4,
        seed=97,
        max_context_lens=(128,),
    )
    workspace = module.create_mla_decode_varlen_workspace(
        fixture.q, fixture.kv_buffer, fixture.output, num_kv_splits=1
    )

    def expect(error_type, **overrides):
        arguments = {
            "q": fixture.q,
            "kv_buffer": fixture.kv_buffer,
            "o": fixture.output,
            "query_start_loc": fixture.query_start_loc,
            "kv_indptr": fixture.kv_indptr,
            "kv_indices": fixture.kv_indices,
            "q_scale": fixture.q_scale,
            "kv_scale": fixture.kv_scale,
            "workspace": workspace,
            "num_kv_splits": 1,
        }
        arguments.update(overrides)
        try:
            module.mla_decode_varlen(**arguments)
        except error_type:
            return
        raise AssertionError(f"expected {error_type.__name__} for {tuple(overrides)}")

    expect(TypeError, q=fixture.q.to(torch.bfloat16))
    expect(ValueError, kv_buffer=fixture.kv_buffer[:, 0])
    expect(ValueError, q_scale=fixture.q_scale.cpu())
    expect(
        ValueError,
        query_start_loc=torch.empty(0, dtype=torch.int32, device=fixture.q.device),
    )
    expect(ValueError, num_kv_splits=0)
    bad_workspace = module.create_mla_decode_varlen_workspace(
        fixture.q, fixture.kv_buffer, fixture.output, num_kv_splits=2
    )
    expect(ValueError, workspace=bad_workspace)
    bad_stage_workspace = module.create_mla_decode_varlen_workspace(
        fixture.q, fixture.kv_buffer, fixture.output, num_kv_splits=1
    )
    bad_stage_workspace.stage_splits = 2
    expect(ValueError, workspace=bad_stage_workspace)
    large_q = torch.empty((960, 128, 576), dtype=dtypes.fp8, device="meta")
    small_kv = torch.empty((1, 1, 576), dtype=dtypes.fp8, device="meta")
    large_output = torch.empty((960, 128, 512), dtype=torch.bfloat16, device="meta")
    try:
        module.create_mla_decode_varlen_workspace(
            large_q, small_kv, large_output, num_kv_splits=16
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected signed-i32 workspace span rejection")

    for cancellation_heads in (8, 16):
        cancellation_fixture = _build_adaptive_cancellation_fixture(cancellation_heads)
        cancellation_reference = torch_mla_varlen_sparse_reference(cancellation_fixture)
        for splits in (1, 4, 16):
            cancellation_workspace = module.create_mla_decode_varlen_workspace(
                cancellation_fixture.q,
                cancellation_fixture.kv_buffer,
                cancellation_fixture.output,
                num_kv_splits=splits,
            )
            cancellation_fixture.output.fill_(cancellation_fixture.sentinel)
            module.mla_decode_varlen(
                cancellation_fixture.q,
                cancellation_fixture.kv_buffer,
                cancellation_fixture.output,
                cancellation_fixture.query_start_loc,
                cancellation_fixture.kv_indptr,
                cancellation_fixture.kv_indices,
                q_scale=cancellation_fixture.q_scale,
                kv_scale=cancellation_fixture.kv_scale,
                workspace=cancellation_workspace,
                num_kv_splits=splits,
            )
            torch.cuda.synchronize()
            metrics = _adaptive_error_metrics(
                cancellation_reference, cancellation_fixture.output
            )
            aiter.logger.info(
                "adaptive cancellation H%d S%d: max_abs=%g rms=%g mismatch=%d nonfinite=%d",
                cancellation_heads,
                splits,
                metrics["max_abs"],
                metrics["rms"],
                metrics["mismatch"],
                metrics["nonfinite"],
            )
            if metrics["mismatch"] or metrics["nonfinite"]:
                raise AssertionError(
                    f"adaptive cancellation H{cancellation_heads} S{splits} failed: {metrics}"
                )

    for dominant_heads in (8, 16):
        dominant_fixture = _build_adaptive_dominant_tail_fixture(dominant_heads)
        dominant_reference = torch_mla_varlen_sparse_reference(
            dominant_fixture, sm_scale=1.0
        )
        dominant_workspace = module.create_mla_decode_varlen_workspace(
            dominant_fixture.q,
            dominant_fixture.kv_buffer,
            dominant_fixture.output,
            num_kv_splits=1,
        )
        module.mla_decode_varlen(
            dominant_fixture.q,
            dominant_fixture.kv_buffer,
            dominant_fixture.output,
            dominant_fixture.query_start_loc,
            dominant_fixture.kv_indptr,
            dominant_fixture.kv_indices,
            sm_scale=1.0,
            q_scale=dominant_fixture.q_scale,
            kv_scale=dominant_fixture.kv_scale,
            workspace=dominant_workspace,
            num_kv_splits=1,
        )
        torch.cuda.synchronize()
        metrics = _adaptive_error_metrics(dominant_reference, dominant_fixture.output)
        aiter.logger.info(
            "adaptive dominant tail H%d S1: max_abs=%g rms=%g mismatch=%d nonfinite=%d",
            dominant_heads,
            metrics["max_abs"],
            metrics["rms"],
            metrics["mismatch"],
            metrics["nonfinite"],
        )
        if metrics["mismatch"] or metrics["nonfinite"]:
            raise AssertionError(
                f"adaptive dominant tail H{dominant_heads} S1 failed: {metrics}"
            )

    canonical_lengths = (1, 31, 32, 63, 64, 65, 128, 256, 384, 512, 1024, 2048)
    for mapping, num_heads, splits, seed in itertools.product(
        ("page64", "token_shuffle"),
        (8, 16, 32, 64, 128),
        (1, 4, 16),
        (307, 401),
    ):
        canonical_fixture = build_adaptive_mla_fixture(
            (1,) * len(canonical_lengths),
            canonical_lengths,
            num_heads,
            2048,
            max_rows=len(canonical_lengths),
            seed=seed,
            max_context_lens=canonical_lengths,
            mapping=mapping,
        )
        canonical_reference = torch_mla_varlen_sparse_reference(canonical_fixture)
        canonical_workspace = module.create_mla_decode_varlen_workspace(
            canonical_fixture.q,
            canonical_fixture.kv_buffer,
            canonical_fixture.output,
            num_kv_splits=splits,
        )
        module.mla_decode_varlen(
            canonical_fixture.q,
            canonical_fixture.kv_buffer,
            canonical_fixture.output,
            canonical_fixture.query_start_loc,
            canonical_fixture.kv_indptr,
            canonical_fixture.kv_indices,
            q_scale=canonical_fixture.q_scale,
            kv_scale=canonical_fixture.kv_scale,
            workspace=canonical_workspace,
            num_kv_splits=splits,
        )
        torch.cuda.synchronize()
        metrics = _adaptive_error_metrics(canonical_reference, canonical_fixture.output)
        aiter.logger.info(
            "adaptive canonical mapping=%s H%d S%d seed=%d: "
            "max_abs=%g rms=%g mismatch=%d nonfinite=%d",
            mapping,
            num_heads,
            splits,
            seed,
            metrics["max_abs"],
            metrics["rms"],
            metrics["mismatch"],
            metrics["nonfinite"],
        )
        if metrics["mismatch"] or metrics["nonfinite"]:
            raise AssertionError(
                "adaptive canonical gate failed: "
                f"mapping={mapping} H{num_heads} S{splits} seed={seed} {metrics}"
            )

    prepared_fixture = build_adaptive_mla_fixture(
        (1, 2, 4, 5, 8),
        (1, 129, 384, 1024, 2048),
        8,
        2048,
        max_rows=24,
        seed=467,
        max_context_lens=(1, 129, 384, 1024, 2048),
        mapping="token_shuffle",
    )
    prepared_reference = torch_mla_varlen_sparse_reference(prepared_fixture)
    prepared = module.prepare_mla_decode_varlen(
        prepared_fixture.q,
        prepared_fixture.kv_buffer,
        prepared_fixture.output,
        prepared_fixture.query_start_loc,
        prepared_fixture.kv_indptr,
        prepared_fixture.kv_indices,
        q_scale=prepared_fixture.q_scale,
        kv_scale=prepared_fixture.kv_scale,
        num_kv_splits=None,
    )
    prepared_q = prepared_fixture.q.clone()
    prepared_kv = prepared_fixture.kv_buffer.clone()
    prepared_indices = prepared_fixture.kv_indices.clone()
    prepared_output = torch.empty_like(prepared_fixture.output)
    prepared_output.fill_(prepared_fixture.sentinel)

    def prepared_invoke():
        module.execute_mla_decode_varlen_prepared(
            prepared,
            prepared_q,
            prepared_kv,
            prepared_output,
            prepared_indices,
            q_scale=prepared_fixture.q_scale,
            kv_scale=prepared_fixture.kv_scale,
        )

    prepared_invoke()
    torch.cuda.synchronize()
    prepared_metrics = _adaptive_error_metrics(prepared_reference, prepared_output)
    if prepared_metrics["mismatch"] or prepared_metrics["nonfinite"]:
        raise AssertionError(
            f"prepared independent-buffer route failed: {prepared_metrics}"
        )
    prepared_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(prepared_graph):
        prepared_invoke()
    for replay_index in range(3):
        prepared_q.copy_(prepared_fixture.q)
        prepared_kv.copy_(prepared_fixture.kv_buffer)
        prepared_indices.copy_(prepared_fixture.kv_indices)
        prepared_output.fill_(prepared_fixture.sentinel)
        prepared_graph.replay()
        torch.cuda.synchronize()
        replay_metrics = _adaptive_error_metrics(prepared_reference, prepared_output)
        if replay_metrics["mismatch"] or replay_metrics["nonfinite"]:
            raise AssertionError(
                f"prepared graph replay {replay_index} failed: {replay_metrics}"
            )

    for graph_heads in (8, 128):
        graph_fixture = AdaptiveMLADecodeFixture.create(
            max_rows=32,
            num_heads=graph_heads,
            topk=2048,
            max_context_lens=(129, 2048, 3073, 6145, 4099),
            mapping="page64",
        )
        for requested_splits, expected_splits in ((None, 16), (1, 1)):
            graph_fixture.load(*ADAPTIVE_SCENARIOS["mixed"], seed=503)
            graph_workspace = module.create_mla_decode_varlen_workspace(
                graph_fixture.q,
                graph_fixture.kv_buffer,
                graph_fixture.output,
                num_kv_splits=requested_splits,
            )
            if graph_workspace.stage_splits != expected_splits:
                raise AssertionError(
                    f"unexpected default split policy: {graph_workspace.stage_splits}"
                )

            def graph_invoke(
                fixture=graph_fixture,
                workspace=graph_workspace,
                splits=requested_splits,
            ):
                module.mla_decode_varlen(
                    fixture.q,
                    fixture.kv_buffer,
                    fixture.output,
                    fixture.query_start_loc,
                    fixture.kv_indptr,
                    fixture.kv_indices,
                    q_scale=fixture.q_scale,
                    kv_scale=fixture.kv_scale,
                    workspace=workspace,
                    num_kv_splits=splits,
                )

            graph_invoke()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_invoke()
            previous = None
            for replay_index, scenario in enumerate(
                ("empty_batch", "mixed", "grow", "zero", "ragged", "mixed")
            ):
                widths, context_lens = ADAPTIVE_SCENARIOS[scenario]
                graph_fixture.load(widths, context_lens, 601 + replay_index)
                reference = torch_mla_varlen_sparse_reference(graph_fixture)
                graph.replay()
                torch.cuda.synchronize()
                metrics = _adaptive_error_metrics(reference, graph_fixture.output)
                if metrics["mismatch"] or metrics["nonfinite"]:
                    raise AssertionError(
                        f"public graph H{graph_heads} {scenario} S{expected_splits} failed: {metrics}"
                    )
                if not torch.equal(
                    graph_fixture.output[graph_fixture.live_rows :],
                    reference[graph_fixture.live_rows :],
                ):
                    raise AssertionError(
                        f"public graph H{graph_heads} {scenario} S{expected_splits} modified padding"
                    )
                current = graph_fixture.output.clone()
                if previous is not None and scenario == "mixed" and replay_index == 5:
                    graph_fixture.load(widths, context_lens, 601 + replay_index)
                    repeated_reference = torch_mla_varlen_sparse_reference(
                        graph_fixture
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    repeated_metrics = _adaptive_error_metrics(
                        repeated_reference, graph_fixture.output
                    )
                    if repeated_metrics["mismatch"] or repeated_metrics["nonfinite"]:
                        raise AssertionError(
                            f"public graph H{graph_heads} S{expected_splits} repeated replay failed: {repeated_metrics}"
                        )
                previous = current


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="sparse MLA decode correctness and performance",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=("adaptive", "legacy"),
        default=["legacy"],
        help="test suite selection; adaptive FlyDSL tests require --modes adaptive",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        choices=tuple(ADAPTIVE_SCENARIOS),
        default=["mixed", "zero", "ragged", "empty_batch"],
    )
    parser.add_argument("-n", "--nhead", type=str, nargs="*", default=None)
    parser.add_argument("--topk", type=int, nargs="*", default=[128, 2048])
    parser.add_argument("--max-rows", type=int, nargs="*", default=[32])
    parser.add_argument("--num-kv-splits", type=int, nargs="*", default=[16])
    parser.add_argument("--groups", type=int, nargs="*", default=[1])
    parser.add_argument(
        "--mapping",
        type=str,
        nargs="*",
        choices=("page64", "token_shuffle"),
        default=["page64"],
    )
    parser.add_argument("-k", "--kv_lora_rank", type=int, default=512)
    parser.add_argument("-qn", "--qk_nope_head_dim", type=int, default=128)
    parser.add_argument("-qr", "--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("-vh", "--v_head_dim", type=int, default=512)
    parser.add_argument("-blk", "--block_size", type=int, default=1)
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp8],
    )
    parser.add_argument(
        "-kvd",
        "--kv_dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp8],
    )
    parser.add_argument(
        "-c",
        "--ctxLen",
        type=int,
        nargs="*",
        default=[21, 64, 256, 512, 1200, 3200, 5200, 8192],
    )
    parser.add_argument(
        "-b",
        "--batchSize",
        type=int,
        nargs="*",
        default=[1, 3, 5, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "-ms", "--max_split_per_batch", type=int, nargs="*", default=[32]
    )
    parser.add_argument("--varlen", action="store_true")
    args = parser.parse_args()

    if "adaptive" in args.modes:
        adaptive_module, unavailable_reason = _load_flydsl_adaptive_module()
        if adaptive_module is None:
            aiter.logger.warning("%s; skipping adaptive suite", unavailable_reason)
        else:
            _run_adaptive_validation_checks(adaptive_module)
            adaptive_heads = (
                [8, 16, 32, 64, 128]
                if args.nhead is None
                else [int(value) for value in args.nhead]
            )
            rows = []
            for (
                scenario,
                nhead,
                topk,
                max_rows,
                num_kv_splits,
                groups,
                mapping,
            ) in itertools.product(
                args.scenarios,
                adaptive_heads,
                args.topk,
                args.max_rows,
                args.num_kv_splits,
                args.groups,
                args.mapping,
            ):
                rows.append(
                    test_mla_varlen_adaptive(
                        scenario,
                        nhead,
                        topk,
                        max_rows,
                        num_kv_splits,
                        groups,
                        mapping,
                    )
                )
            dataframe = pd.DataFrame(rows)
            aiter.logger.info(
                "adaptive sparse MLA summary (markdown):\n%s",
                dataframe.to_markdown(index=False),
            )

    if "legacy" in args.modes:
        legacy_heads = (
            [(16, 2), (48, 1), (128, 2)]
            if args.nhead is None
            else [dtypes.str2tuple(value) for value in args.nhead]
        )
        for nhead, decode_qlen in legacy_heads:
            rows = []
            for (
                dtype,
                kvtype,
                ctx_len,
                batch_size,
                max_split_per_batch,
            ) in itertools.product(
                args.dtype,
                args.kv_dtype,
                args.ctxLen,
                args.batchSize,
                args.max_split_per_batch,
            ):
                if check_support(dtype, kvtype, nhead):
                    rows.append(
                        test_mla(
                            ctx_len,
                            batch_size,
                            nhead,
                            args.kv_lora_rank,
                            args.qk_nope_head_dim,
                            args.qk_rope_head_dim,
                            args.v_head_dim,
                            dtype,
                            kvtype,
                            args.block_size,
                            varlen=args.varlen,
                            decode_qlen=decode_qlen,
                            max_split_per_batch=max_split_per_batch,
                        )
                    )
            dataframe = pd.DataFrame(rows)
            aiter.logger.info(
                "mla_sparse summary (markdown):\n%s",
                dataframe.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
