# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark suite for experimental paged-MQA Stage A."""

import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass

import pandas as pd
import pytest
import torch

pytest.importorskip("flydsl")

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_fp8_paged_mqa_local_topk,
    flydsl_fp8_paged_mqa_topk,
)
from aiter.ops.flydsl.fp8_paged_mqa_local_topk import (
    _plan_num_splits,
    merge_local_topk_candidates,
)
from aiter.ops.flydsl.split_topk_merge import (
    clear_split_topk_merge_workspace_cache,
    multisequence_sorted_split_topk_merge,
    persistent_split_topk_merge_workspace,
    restore_persistent_b_private,
    split_topk_merge,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx950",)
HEADS = 32
HEAD_DIM = 128


@dataclass
class Case:
    q: torch.Tensor
    kv: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    block_tables: torch.Tensor
    indices: torch.Tensor
    query_start_loc: torch.Tensor
    decode_lens: torch.Tensor


@dataclass
class PackedCase:
    q: torch.Tensor
    kv: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    block_tables: torch.Tensor
    indices: torch.Tensor
    query_start_loc: torch.Tensor
    decode_lens: torch.Tensor
    live_rows: int


def _make_packed_case(
    decode_lens,
    length,
    page_size,
    *,
    seed=17,
    independent_kv=False,
    context_len=None,
    r_max=None,
):
    decode_lens_cpu = torch.as_tensor(decode_lens, dtype=torch.int32, device="cpu")
    if decode_lens_cpu.ndim != 1 or decode_lens_cpu.numel() == 0:
        raise ValueError("decode_lens must be a non-empty 1D sequence")
    if torch.any(decode_lens_cpu < 1):
        raise ValueError("decode_lens entries must be positive")

    batch = decode_lens_cpu.numel()
    live_rows = int(decode_lens_cpu.sum().item())
    if r_max is None:
        r_max = live_rows
    if r_max < live_rows:
        raise ValueError("r_max must be at least sum(decode_lens)")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    if context_len is None:
        context_len = length
    if context_len < int(decode_lens_cpu.max()) or context_len > length:
        raise ValueError("context_len must be in [max(decode_lens), length]")

    context_pages = max(1, (context_len + page_size - 1) // page_size)
    table_pages = max(1, (length + page_size - 1) // page_size)
    q = (
        torch.randn(
            r_max,
            1,
            HEADS,
            HEAD_DIM,
            device=device,
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    pages = batch * context_pages if independent_kv else context_pages
    kv = (
        torch.randn(pages, page_size, HEAD_DIM, device=device, generator=generator)
        * 0.25
    ).to(torch.float8_e4m3fn)
    scales = torch.rand(
        pages, page_size, device=device, generator=generator, dtype=torch.float32
    )
    weights = torch.randn(
        r_max, HEADS, device=device, generator=generator, dtype=torch.float32
    )

    decode_lens = decode_lens_cpu.to(device)
    query_start_loc = torch.empty(batch + 1, device=device, dtype=torch.int32)
    query_start_loc[0] = 0
    torch.cumsum(decode_lens, dim=0, out=query_start_loc[1:])
    live_indices = torch.repeat_interleave(
        torch.arange(batch, device=device, dtype=torch.int32),
        decode_lens,
        output_size=live_rows,
    )
    indices = torch.zeros(r_max, device=device, dtype=torch.int32)
    indices[:live_rows] = live_indices
    live_lengths = torch.cat(
        [
            torch.arange(
                context_len - int(n) + 1,
                context_len + 1,
                device=device,
                dtype=torch.int32,
            )
            for n in decode_lens_cpu.tolist()
        ]
    )
    lengths = torch.zeros(r_max, device=device, dtype=torch.int32)
    lengths[:live_rows] = live_lengths

    block_tables = torch.zeros(
        (batch, table_pages), device=device, dtype=torch.int32
    )
    if independent_kv:
        for req in range(batch):
            origin = req * context_pages
            perm = torch.randperm(
                context_pages, device=device, generator=generator
            )
            block_tables[req, :context_pages] = (perm + origin).to(torch.int32)
    else:
        physical_order = torch.randperm(
            context_pages, device=device, generator=generator
        )
        block_tables[:, :context_pages] = physical_order.to(torch.int32)

    return PackedCase(
        q,
        kv,
        scales,
        weights,
        lengths,
        block_tables,
        indices,
        query_start_loc,
        decode_lens,
        live_rows,
    )


def _require_supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/HIP GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(":", 1)[0]
    if arch not in SUPPORTED_GFX:
        pytest.skip(f"Stage A is not yet supported on {arch}")
    return arch


def _make_case(
    rows,
    length,
    page_size,
    *,
    seed=17,
    ragged=False,
    next_n=1,
    independent_kv=False,
    context_len=None,
):
    if rows % next_n:
        raise ValueError("rows must be divisible by next_n")
    batch = rows // next_n
    packed = _make_packed_case(
        [next_n] * batch,
        length,
        page_size,
        seed=seed,
        independent_kv=independent_kv,
        context_len=context_len,
    )
    q = packed.q.view(batch, next_n, HEADS, HEAD_DIM)
    lengths = packed.lengths.view(batch, next_n) if next_n > 1 else packed.lengths
    if ragged and rows > 1:
        flat_lengths = lengths.reshape(-1)
        flat_lengths[0] = 0
        flat_lengths[-1] = max(0, int(packed.lengths[-1]) - 3)
    return Case(
        q,
        packed.kv,
        packed.scales,
        packed.weights,
        lengths,
        packed.block_tables,
        packed.indices,
        packed.query_start_loc,
        packed.decode_lens,
    )


def _preshuffle_kv(kv):
    return shuffle_weight(kv, layout=(16, 16)).contiguous()


def _pack_kv(kv, scales):
    pages, page_size, dim = kv.shape
    raw = torch.empty(
        (pages, page_size * (dim + 4)),
        dtype=torch.uint8,
        device=kv.device,
    )
    raw[:, : page_size * dim] = kv.view(torch.uint8).reshape(pages, -1)
    raw[:, page_size * dim :] = scales.view(torch.uint8).reshape(pages, -1)
    return raw.view(pages, page_size, 1, dim + 4)


def run_torch(case):
    """Independent FP32 oracle with explicit page mapping and epilogue order."""
    rows = case.weights.shape[0]
    q = case.q.reshape(rows, HEADS, HEAD_DIM)
    lengths = case.lengths.reshape(-1)
    page_size = case.kv.shape[1]
    flat_kv = case.kv.reshape(-1, HEAD_DIM).float()
    flat_scales = case.scales.reshape(-1)
    outputs = []
    for row in range(rows):
        length = int(lengths[row].item())
        logical = torch.arange(length, device=case.q.device)
        physical = (
            case.block_tables[int(case.indices[row]), logical // page_size] * page_size
            + logical % page_size
        )
        keys = flat_kv[physical]
        dots = torch.sum(q[row].float()[:, None, :] * keys[None, :, :], dim=-1)
        scaled = dots * flat_scales[physical][None, :]
        activated = torch.relu(scaled)
        outputs.append(torch.sum(case.weights[row, :, None] * activated, dim=0))
    return outputs


def _assert_candidates(case, scores, positions, counts, *, k, splits):
    reference = run_torch(case)
    all_global = []
    for row, row_scores in enumerate(reference):
        length = row_scores.numel()
        union = set()
        for split in range(splits):
            begin = length * split // splits
            end = length * (split + 1) // splits
            count = min(k, end - begin)
            assert int(counts[row, split]) == count
            got_positions = positions[row, split, :count].long()
            assert got_positions.unique().numel() == count
            assert torch.all((got_positions >= begin) & (got_positions < end))
            if count:
                local_scores = row_scores[begin:end]
                if local_scores.numel() > k:
                    ordered = torch.sort(local_scores, descending=True).values
                    assert ordered[k - 1] > ordered[k]
                expected = torch.topk(local_scores, count, sorted=False).indices + begin
                assert set(got_positions.cpu().tolist()) == set(expected.cpu().tolist())
                got_scores = scores[row, split, :count].float()
                checkAllclose(
                    row_scores[got_positions].float(),
                    got_scores,
                    rtol=2e-4,
                    atol=2e-4,
                    msg=f"row={row} split={split} selected scores",
                )
                if count > 1:
                    assert torch.all(got_scores[:-1] >= got_scores[1:])
                union.update(got_positions.cpu().tolist())
            assert torch.all(positions[row, split, count:] == -1)
            assert torch.all(torch.isneginf(scores[row, split, count:]))

        global_count = min(k, length)
        if global_count:
            global_topk = torch.topk(
                row_scores, global_count, sorted=False
            ).indices.cpu()
            assert set(global_topk.tolist()).issubset(union)
        all_global.append(global_count)
    return reference, all_global


@pytest.mark.parametrize(
    "rows,length,k,splits,page_size",
    [
        (1, 0, 128, 1, 16),
        (2, 1, 128, 4, 1),
        (2, 127, 128, 4, 16),
        (2, 128, 128, 1, 64),
        (2, 129, 128, 4, 16),
    ],
)
def test_bringup_empty_and_tails(rows, length, k, splits, page_size):
    _require_supported_gpu()
    case = _make_case(rows, length, page_size, ragged=rows > 1)
    outputs = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        case.kv,
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    _assert_candidates(case, *outputs, k=k, splits=splits)


def test_unique_kth_local_sets_and_union_recall():
    _require_supported_gpu()
    rows, length, k, splits = 2, 8193, 128, 4
    case = _make_case(rows, length, 16, seed=31)
    outputs = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        case.kv,
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    _assert_candidates(case, *outputs, k=k, splits=splits)


def test_preshuffled_page64_local_sets():
    _require_supported_gpu()
    rows, length, k, splits = 2, 8193, 128, 4
    case = _make_case(rows, length, 64, seed=41)
    outputs = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        _preshuffle_kv(case.kv),
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    _assert_candidates(case, *outputs, k=k, splits=splits)


def test_preshuffled_page64_packed_matches_split():
    _require_supported_gpu()
    rows, length, k, splits = 2, 8193, 128, 4
    case = _make_case(rows, length, 64, seed=41)
    shuffled = _preshuffle_kv(case.kv)
    split_out = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        shuffled,
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    packed_out = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        _pack_kv(shuffled, case.scales),
        None,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    _assert_candidates(case, *split_out, k=k, splits=splits)
    _assert_candidates(case, *packed_out, k=k, splits=splits)
    torch.testing.assert_close(packed_out[0], split_out[0], rtol=0, atol=0)
    torch.testing.assert_close(packed_out[1], split_out[1], rtol=0, atol=0)
    torch.testing.assert_close(packed_out[2], split_out[2], rtol=0, atol=0)


def _assert_compact_topk(case, scores, positions, k):
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = min(k, row_scores.numel())
        assert torch.all(positions[row, count:] == -1)
        assert torch.all(torch.isneginf(scores[row, count:]))
        if count:
            expected = torch.topk(row_scores, count, sorted=False).indices
            got = positions[row, :count].long()
            assert set(got.cpu().tolist()) == set(expected.cpu().tolist())
            checkAllclose(
                row_scores[got].float(),
                scores[row, :count].float(),
                rtol=2e-4,
                atol=2e-4,
                msg=f"row={row} compact TopK",
            )


@pytest.mark.parametrize(
    "length,k,splits,next_n",
    [
        (4096, 128, 1, 1),
        (8193, 128, 4, 2),
        (8193, 512, 4, 1),
        (8193, 1024, 4, 1),
        (8193, 2048, 4, 1),
    ],
)
def test_packed_page64_compact_topk(length, k, splits, next_n):
    _require_supported_gpu()
    rows = 2 * next_n
    case = _make_case(rows, length, 64, seed=43, next_n=next_n)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    lengths = (
        case.lengths[:, -1].contiguous() if case.lengths.ndim == 2 else case.lengths
    )
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        packed,
        None,
        case.weights,
        lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    _assert_compact_topk(case, scores, positions, k)


def test_oversized_length_clamps_to_table_span():
    _require_supported_gpu()
    rows, length, k, splits = 2, 128, 128, 1
    case = _make_case(rows, length, 64, seed=71)
    inflated = torch.full_like(case.lengths, 10_000)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        packed,
        None,
        case.weights,
        inflated,
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    _assert_compact_topk(case, scores, positions, k)


def test_invalid_physical_page_is_not_scored():
    _require_supported_gpu()
    rows, length, k, splits = 1, 128, 128, 1
    case = _make_case(rows, length, 64, seed=73)
    case.block_tables[0, 0] = -1
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        packed,
        None,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    chosen = positions[0]
    live = chosen[scores[0] > float("-inf")]
    assert live.numel() == 64
    assert torch.all(live >= 64)
    assert torch.all(live < 128)


def test_short_split_invalid_page_keeps_live_positions():
    """Pad ``-inf`` must not steal slots from live ``-inf`` (retained < k)."""
    _require_supported_gpu()
    rows, length, k, splits = 1, 64, 128, 1
    case = _make_case(rows, length, 64, seed=73)
    case.block_tables[0, 0] = -1
    scores, positions, counts = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        _preshuffle_kv(case.kv),
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    assert int(counts[0, 0]) == length
    assert torch.all(torch.isneginf(scores[0, 0, :length]))
    assert torch.all(torch.isneginf(scores[0, 0, length:]))
    got = positions[0, 0, :length]
    assert got.unique().numel() == length
    assert set(got.cpu().tolist()) == set(range(length))
    assert torch.all(positions[0, 0, length:] == -1)


@pytest.mark.parametrize(
    "next_n,rows",
    [(2, 8), (3, 6), (4, 8), (8, 8)],
)
def test_preshuffled_page64_mtp_causal_local_sets(next_n, rows):
    _require_supported_gpu()
    length, k, splits = 8193, 128, 4
    case = _make_case(
        rows,
        length,
        64,
        seed=42,
        next_n=next_n,
    )
    outputs = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        _preshuffle_kv(case.kv),
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    _assert_candidates(case, *outputs, k=k, splits=splits)


def test_preshuffled_page64_mtp_threshold_ties_and_nan_bottom():
    _require_supported_gpu()
    rows, next_n, length, k, splits = 2, 2, 4096, 128, 1

    tie_case = _make_case(
        rows,
        length,
        64,
        seed=48,
        next_n=next_n,
    )
    tie_case.weights.zero_()
    tie_scores, tie_positions, tie_counts = flydsl_fp8_paged_mqa_local_topk(
        tie_case.q,
        _preshuffle_kv(tie_case.kv),
        tie_case.scales,
        tie_case.weights,
        tie_case.lengths,
        tie_case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    assert torch.equal(tie_counts, torch.full_like(tie_counts, k))
    assert torch.equal(tie_scores, torch.zeros_like(tie_scores))
    for row in range(rows):
        assert tie_positions[row, 0].unique().numel() == k

    nan_case = _make_case(
        rows,
        length,
        64,
        seed=49,
        next_n=next_n,
    )
    nan_case.scales.reshape(-1)[1::2] = float("nan")
    nan_scores, nan_positions, _ = flydsl_fp8_paged_mqa_local_topk(
        nan_case.q,
        _preshuffle_kv(nan_case.kv),
        nan_case.scales,
        nan_case.weights,
        nan_case.lengths,
        nan_case.block_tables,
        k=k,
        num_splits=splits,
        preshuffled=True,
    )
    reference = run_torch(nan_case)
    for row in range(rows):
        expected = torch.topk(
            torch.nan_to_num(reference[row], nan=-float("inf")),
            k,
            sorted=False,
        ).indices
        got = nan_positions[row, 0].long()
        assert set(got.cpu().tolist()) == set(expected.cpu().tolist())
        assert not torch.isnan(nan_scores[row, 0]).any()


def test_auto_split_plan():
    length, k, cu_count = 1_048_576, 2048, 256
    assert _plan_num_splits(8, length, k, cu_count) == 64
    assert _plan_num_splits(16, length, k, cu_count) == 32
    assert _plan_num_splits(32, length, k, cu_count) == 32
    assert _plan_num_splits(1, 4096, k, cu_count) == 2


def test_packed_adaptive_harness_layout():
    _require_supported_gpu()
    case = _make_packed_case(
        [2, 2, 2, 2, 1, 1, 1, 1],
        128,
        64,
        seed=53,
        independent_kv=True,
        r_max=16,
    )
    assert case.live_rows == 12
    assert case.q.shape == (16, 1, HEADS, HEAD_DIM)
    assert case.weights.shape == (16, HEADS)
    assert case.query_start_loc.tolist() == [0, 2, 4, 6, 8, 9, 10, 11, 12]
    assert case.indices.tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        5,
        6,
        7,
        0,
        0,
        0,
        0,
    ]
    assert case.lengths.tolist() == [
        127,
        128,
        127,
        128,
        127,
        128,
        127,
        128,
        128,
        128,
        128,
        128,
        0,
        0,
        0,
        0,
    ]
    page_sets = [set(row.tolist()) for row in case.block_tables]
    assert all(page_sets[i].isdisjoint(page_sets[j]) for i in range(8) for j in range(i))


def test_preshuffled_page64_single_split_topk():
    _require_supported_gpu()
    rows, next_n, length, k = 2, 2, 4096, 128
    case = _make_case(rows, length, 64, seed=51, next_n=next_n)
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        _preshuffle_kv(case.kv),
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=1,
    )
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        expected = torch.topk(row_scores, k, sorted=False).indices
        got = positions[row].long()
        assert set(got.cpu().tolist()) == set(expected.cpu().tolist())
        checkAllclose(
            row_scores[got].float(),
            scores[row].float(),
            rtol=2e-4,
            atol=2e-4,
            msg=f"row={row} single-split TopK",
        )


@pytest.mark.parametrize("next_n", [2, 3, 4, 8])
def test_preshuffled_page64_mtp_compact_topk(next_n):
    _require_supported_gpu()
    rows, length, k, splits = next_n * 2, 8193, 128, 4
    case = _make_case(
        rows,
        length,
        64,
        seed=44,
        next_n=next_n,
    )
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        _preshuffle_kv(case.kv),
        case.scales,
        case.weights,
        case.lengths[:, -1].contiguous(),
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = min(k, row_scores.numel())
        expected = torch.topk(row_scores, count, sorted=False).indices
        got = positions[row, :count].long()
        assert set(got.cpu().tolist()) == set(expected.cpu().tolist())
        checkAllclose(
            row_scores[got].float(),
            scores[row, :count].float(),
            rtol=2e-4,
            atol=2e-4,
            msg=f"row={row} next_n={next_n} compact TopK",
        )


@pytest.mark.parametrize(
    "length,k",
    [(0, 128), (129, 128), (8193, 512), (8193, 1024), (8193, 2048)],
)
def test_preshuffled_page64_compact_topk(length, k):
    _require_supported_gpu()
    rows = 2
    case = _make_case(rows, length, 64, seed=43)
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        _preshuffle_kv(case.kv),
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=4,
    )
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = min(k, length)
        assert torch.all(positions[row, count:] == -1)
        assert torch.all(torch.isneginf(scores[row, count:]))
        if count:
            expected = torch.topk(row_scores, count, sorted=False).indices
            got = positions[row, :count].long()
            assert set(got.cpu().tolist()) == set(expected.cpu().tolist())
            checkAllclose(
                row_scores[got].float(),
                scores[row, :count].float(),
                rtol=2e-4,
                atol=2e-4,
                msg=f"row={row} compact TopK",
            )


def test_packed_page64_run_only_cache_hit():
    _require_supported_gpu()
    from aiter.aot.flydsl.common import run_only_env

    case = _make_case(2, 8193, 64, seed=61)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    args = (
        case.q,
        packed,
        None,
        case.weights,
        case.lengths,
        case.block_tables,
    )
    scores, positions = flydsl_fp8_paged_mqa_topk(*args, k=128, num_splits=4)
    _assert_compact_topk(case, scores, positions, 128)
    with run_only_env():
        scores, positions = flydsl_fp8_paged_mqa_topk(*args, k=128, num_splits=4)
    _assert_compact_topk(case, scores, positions, 128)


def test_packed_page64_e2e_graph_replay():
    _require_supported_gpu()
    case = _make_case(2, 8193, 64, seed=67)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    args = (
        case.q,
        packed,
        None,
        case.weights,
        case.lengths,
        case.block_tables,
    )
    flydsl_fp8_paged_mqa_topk(*args, k=128, num_splits=4)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scores, positions = flydsl_fp8_paged_mqa_topk(
            *args,
            k=128,
            num_splits=4,
        )
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    _assert_compact_topk(case, scores, positions, 128)


def test_k2048_reservoir_and_existing_stage_b():
    _require_supported_gpu()
    rows, length, k, splits = 1, 8193, 2048, 2
    case = _make_case(rows, length, 64, seed=47)
    outputs = flydsl_fp8_paged_mqa_local_topk(
        case.q,
        case.kv,
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
    )
    reference, _ = _assert_candidates(case, *outputs, k=k, splits=splits)
    _, final_positions = merge_local_topk_candidates(*outputs, k=k)
    _, split_positions = split_topk_merge(*outputs, k=k)
    _, replay_positions = split_topk_merge(*outputs, k=k)
    expected = torch.topk(reference[0], k, sorted=False).indices
    assert set(final_positions[0].cpu().tolist()) == set(expected.cpu().tolist())
    assert set(split_positions[0].cpu().tolist()) == set(expected.cpu().tolist())
    assert set(replay_positions[0].cpu().tolist()) == set(expected.cpu().tolist())


def test_split_topk_merge_counts_nan_and_graph_replay():
    _require_supported_gpu()
    rows, splits, k = 2, 4, 2048
    counts = torch.tensor(
        [[2048, 1537, 2048, 777], [1301, 2048, 911, 2048]],
        dtype=torch.int32,
        device="cuda",
    )
    scores = torch.full(
        (rows, splits, k), float("inf"), dtype=torch.float32, device="cuda"
    )
    positions = torch.full((rows, splits, k), -1, dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(59)
    for row in range(rows):
        position = 0
        for split in range(splits):
            count = int(counts[row, split])
            scores[row, split, :count] = torch.randn(
                count, generator=generator, device="cuda"
            )
            positions[row, split, :count] = torch.arange(
                position, position + count, dtype=torch.int32, device="cuda"
            )
            position += count
        scores[row, 0, :32] = float("nan")

    def expected_positions(row):
        valid_scores = torch.cat(
            [scores[row, split, : int(counts[row, split])] for split in range(splits)]
        )
        valid_positions = torch.cat(
            [
                positions[row, split, : int(counts[row, split])]
                for split in range(splits)
            ]
        )
        valid_scores = torch.nan_to_num(valid_scores, nan=-float("inf"))
        return set(valid_positions[torch.topk(valid_scores, k).indices].cpu().tolist())

    _, selected = split_topk_merge(scores, positions, counts, k=k)
    for row in range(rows):
        assert set(selected[row].cpu().tolist()) == expected_positions(row)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, replayed = split_topk_merge(scores, positions, counts, k=k)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    for row in range(rows):
        assert set(replayed[row].cpu().tolist()) == expected_positions(row)

    clear_split_topk_merge_workspace_cache()
    cold_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold_graph):
        _, cold_replayed = split_topk_merge(scores, positions, counts, k=k)
    for _ in range(5):
        cold_graph.replay()
    torch.cuda.synchronize()
    for row in range(rows):
        assert set(cold_replayed[row].cpu().tolist()) == expected_positions(row)

    tie_scores = torch.full_like(scores, float("inf"))
    for row in range(rows):
        for split in range(splits):
            tie_scores[row, split, : int(counts[row, split])] = 0
    tie_values, tie_positions = split_topk_merge(tie_scores, positions, counts, k=k)
    assert torch.equal(tie_values, torch.zeros_like(tie_values))
    for row in range(rows):
        chosen = tie_positions[row].cpu().tolist()
        assert len(set(chosen)) == k
        assert min(chosen) >= 0


def test_ordered_emit_stress_is_canonical_descending():
    _require_supported_gpu()
    rows, splits, k = 8, 64, 2048
    case = _make_case(rows, 131072, 64, seed=107)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    for _ in range(32):
        scores, _, _ = flydsl_fp8_paged_mqa_local_topk(
            case.q,
            packed,
            None,
            case.weights,
            case.lengths,
            case.block_tables,
            k=k,
            num_splits=splits,
            preshuffled=True,
        )
        bits = scores.view(torch.int32)
        ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        ordered[torch.isnan(scores)] = -(1 << 31)
        assert torch.all(ordered[:, :, :-1] >= ordered[:, :, 1:])


@pytest.mark.parametrize(
    "splits,k,parallel_representatives,coordinated_partition",
    [
        (4, 128, False, False),
        (4, 128, True, False),
        (4, 128, False, True),
        (2, 128, False, True),
        (16, 2048, False, False),
        (16, 2048, True, False),
        (16, 2048, False, True),
        (64, 2048, False, True),
        (88, 2048, False, False),
        (88, 2048, False, True),
    ],
)
def test_multisequence_sorted_split_topk_merge(
    splits,
    k,
    parallel_representatives,
    coordinated_partition,
):
    _require_supported_gpu()
    rows = 2
    generator = torch.Generator(device="cuda").manual_seed(101 + splits)
    scores = torch.randn(
        rows,
        splits,
        k,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    counts = torch.full((rows, splits), k, dtype=torch.int32, device="cuda")
    counts[0, ::3] = max(1, k // 3)
    scores[:, :, :16] = 0
    scores[:, :, 16:24] = float("nan")
    positions = torch.arange(
        rows * splits * k,
        dtype=torch.int32,
        device="cuda",
    ).reshape(rows, splits, k)

    bits = scores.view(torch.int32)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    ordered[torch.isnan(scores)] = -(1 << 31)
    order = torch.argsort(ordered, dim=-1, descending=True, stable=True)
    scores = torch.gather(scores, -1, order)
    positions = torch.gather(positions, -1, order)

    selected_scores, selected_positions = multisequence_sorted_split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
        coordinated_partition=coordinated_partition,
        parallel_representatives=parallel_representatives,
    )
    for row in range(rows):
        live_scores = torch.cat(
            [scores[row, split, : int(counts[row, split])] for split in range(splits)]
        )
        live_bits = live_scores.view(torch.int32)
        live_ordered = live_bits ^ ((live_bits >> 31) & 0x7FFFFFFF)
        live_ordered[torch.isnan(live_scores)] = -(1 << 31)
        expected_keys = torch.topk(live_ordered, k, sorted=True).values

        got_bits = selected_scores[row].view(torch.int32)
        got_keys = got_bits ^ ((got_bits >> 31) & 0x7FFFFFFF)
        got_keys[torch.isnan(selected_scores[row])] = -(1 << 31)
        torch.testing.assert_close(
            torch.sort(got_keys, descending=True).values,
            expected_keys,
            rtol=0,
            atol=0,
        )
        assert len(set(selected_positions[row].cpu().tolist())) == k

    if splits == 4 and not parallel_representatives:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_scores, graph_positions = (
                multisequence_sorted_split_topk_merge(
                    scores,
                    positions,
                    counts,
                    k=k,
                    coordinated_partition=coordinated_partition,
                )
            )
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_scores, selected_scores, rtol=0, atol=0)
        assert torch.equal(graph_positions, selected_positions)


def test_multisequence_sorted_split_topk_merge_short_row():
    _require_supported_gpu()
    rows, splits, k = 1, 4, 128
    counts = torch.tensor([[7, 0, 11, 5]], dtype=torch.int32, device="cuda")
    scores = torch.full(
        (rows, splits, k),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    positions = torch.full((rows, splits, k), -1, dtype=torch.int32, device="cuda")
    position = 0
    for split in range(splits):
        count = int(counts[0, split])
        scores[0, split, :count] = torch.arange(
            count,
            0,
            -1,
            dtype=torch.float32,
            device="cuda",
        )
        positions[0, split, :count] = torch.arange(
            position,
            position + count,
            dtype=torch.int32,
            device="cuda",
        )
        position += count

    selected_scores, selected_positions = multisequence_sorted_split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
    )
    assert set(selected_positions[0, :position].cpu().tolist()) == set(range(position))
    assert torch.equal(
        selected_positions[0, position:],
        torch.full_like(selected_positions[0, position:], -1),
    )
    assert torch.all(torch.isneginf(selected_scores[0, position:]))
    coordinated_scores, coordinated_positions = (
        multisequence_sorted_split_topk_merge(
            scores,
            positions,
            counts,
            k=k,
            coordinated_partition=True,
        )
    )
    assert torch.equal(coordinated_positions, selected_positions)
    torch.testing.assert_close(
        coordinated_scores,
        selected_scores,
        rtol=0,
        atol=0,
    )

    multisequence_sorted_split_topk_merge(scores, positions, counts, k=k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_scores, graph_positions = multisequence_sorted_split_topk_merge(
            scores,
            positions,
            counts,
            k=k,
        )
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_positions, selected_positions)
    torch.testing.assert_close(graph_scores, selected_scores, rtol=0, atol=0)

    bounded_counts = torch.tensor(
        [[k + 7, -3, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    bounded_scores = torch.arange(
        k,
        0,
        -1,
        dtype=torch.float32,
        device="cuda",
    ).reshape(1, 1, k).expand(1, splits, k).contiguous()
    bounded_positions = torch.arange(
        splits * k,
        dtype=torch.int32,
        device="cuda",
    ).reshape(1, splits, k)
    _, bounded_output = multisequence_sorted_split_topk_merge(
        bounded_scores,
        bounded_positions,
        bounded_counts,
        k=k,
    )
    assert torch.equal(bounded_output[0], bounded_positions[0, 0])
    _, coordinated_bounded_output = multisequence_sorted_split_topk_merge(
        bounded_scores,
        bounded_positions,
        bounded_counts,
        k=k,
        coordinated_partition=True,
    )
    assert torch.equal(coordinated_bounded_output, bounded_output)


def test_coordinated_partition_all_short_and_stable_ties():
    _require_supported_gpu()
    k, splits = 128, 16
    counts = torch.full((1, splits), 9, dtype=torch.int32, device="cuda")
    scores = torch.full(
        (1, splits, k),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    positions = torch.arange(
        splits * k,
        dtype=torch.int32,
        device="cuda",
    ).reshape(1, splits, k)
    for split in range(splits):
        scores[0, split, :9] = torch.arange(
            9,
            0,
            -1,
            dtype=torch.float32,
            device="cuda",
        )
    output_scores, output_positions = multisequence_sorted_split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
        coordinated_partition=True,
    )
    live = torch.cat([scores[0, split, :9] for split in range(splits)])
    expected = torch.topk(live, k).values
    torch.testing.assert_close(
        torch.sort(output_scores[0], descending=True).values,
        torch.sort(expected, descending=True).values,
        rtol=0,
        atol=0,
    )
    assert output_positions.unique().numel() == k

    tie_k, tie_splits = 4, 4
    tie_scores = torch.ones(
        (1, tie_splits, tie_k),
        dtype=torch.float32,
        device="cuda",
    )
    tie_positions = torch.arange(
        tie_splits * tie_k,
        dtype=torch.int32,
        device="cuda",
    ).reshape(1, tie_splits, tie_k)
    tie_counts = torch.full(
        (1, tie_splits),
        tie_k,
        dtype=torch.int32,
        device="cuda",
    )
    _, tie_output = multisequence_sorted_split_topk_merge(
        tie_scores,
        tie_positions,
        tie_counts,
        k=tie_k,
        coordinated_partition=True,
    )
    assert torch.equal(tie_output[0], tie_positions[0, 0])


@benchmark()
def benchmark_local_topk(rows, length, k, splits, page_size):
    case = _make_case(rows, length, page_size, seed=71)
    reference = run_torch(case)
    kv_cache = _preshuffle_kv(case.kv) if page_size % 16 == 0 else case.kv
    preshuffled = page_size % 16 == 0
    candidates = {
        "flydsl_stage_a": lambda: flydsl_fp8_paged_mqa_local_topk(
            case.q,
            kv_cache,
            case.scales,
            case.weights,
            case.lengths,
            case.block_tables,
            k=k,
            num_splits=splits,
            preshuffled=preshuffled,
        )
    }
    flops = 2 * rows * length * HEADS * HEAD_DIM
    nbytes = (
        case.q.numel() * case.q.element_size()
        + case.kv.numel() * case.kv.element_size()
        + case.scales.numel() * case.scales.element_size()
        + case.weights.numel() * case.weights.element_size()
        + rows * splits * k * 8
    )
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        output, us = run_perftest(candidate)
        _assert_candidates(case, *output, k=k, splits=splits)
        # Representative score error for the first emitted candidate.
        position = output[1][0, 0, 0].long()
        err = checkAllclose(
            reference[0][position].reshape(1).float(),
            output[0][0, 0, 0].reshape(1).float(),
            rtol=2e-4,
            atol=2e-4,
            msg=name,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def benchmark_auto_topk(rows, next_n, length, k, page_size):
    case = _make_case(rows, length, page_size, seed=73, next_n=next_n)
    reference = run_torch(case)
    kv_cache = _preshuffle_kv(case.kv)

    def candidate():
        return flydsl_fp8_paged_mqa_topk(
            case.q,
            kv_cache,
            case.scales,
            case.weights,
            case.lengths,
            case.block_tables,
            k=k,
            num_splits=None,
        )

    (scores, positions), us = run_perftest(candidate)
    for row, row_scores in enumerate(reference):
        expected = torch.topk(row_scores, k, sorted=False).indices
        got = positions[row].long()
        assert set(got.cpu().tolist()) == set(expected.cpu().tolist())
    err = checkAllclose(
        reference[0][positions[0].long()].float(),
        scores[0].float(),
        rtol=2e-4,
        atol=2e-4,
        msg="flydsl_auto_topk",
    )
    flops = 2 * rows * length * HEADS * HEAD_DIM
    nbytes = (
        case.q.numel() * case.q.element_size()
        + case.kv.numel() * case.kv.element_size()
        + case.scales.numel() * case.scales.element_size()
        + case.weights.numel() * case.weights.element_size()
        + rows * k * 8
    )
    return {
        "gfx": get_gfx(),
        "flydsl_auto_topk us": us,
        "flydsl_auto_topk TFLOPS": flops / us / 1e6,
        "flydsl_auto_topk TB/s": nbytes / us / 1e6,
        "flydsl_auto_topk err": err,
    }


def _ordered_i32(values):
    bits = values.view(torch.int32)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    abs_bits = bits & 0x7FFFFFFF
    nan = abs_bits > 0x7F800000
    return torch.where(nan, torch.full_like(bits, -(1 << 31)), ordered)


def _prepare_pass0_workspace(scores, counts, workspace):
    from aiter.ops.flydsl.kernels.split_topk_merge_layout import (
        HIST0_OFF,
        NUM_HIST_BINS,
        ROW_STRIDE,
    )

    rows, splits, local_k = scores.shape
    restore_persistent_b_private(workspace, rows)
    rows_view = workspace.view(rows, ROW_STRIDE)
    rows_view[:, HIST0_OFF : HIST0_OFF + NUM_HIST_BINS].zero_()
    sign = 1 << 10
    for row in range(rows):
        for split in range(splits):
            count = int(counts[row, split].clamp(min=0).item())
            count = min(count, local_k)
            if count <= 0:
                continue
            keys = _ordered_i32(scores[row, split, :count])
            buckets = ((keys >> 21) & 2047) ^ sign
            hist = workspace[
                row * ROW_STRIDE
                + HIST0_OFF : row * ROW_STRIDE
                + HIST0_OFF
                + NUM_HIST_BINS
            ]
            hist.scatter_add_(0, buckets.long(), torch.ones_like(buckets))


def _assert_unique_kth_set(scores, positions, counts, selected, k):
    rows, splits, _ = scores.shape
    for row in range(rows):
        live_scores = []
        live_positions = []
        for split in range(splits):
            count = int(max(0, min(k, int(counts[row, split]))))
            if count == 0:
                continue
            live_scores.append(scores[row, split, :count])
            live_positions.append(positions[row, split, :count])
        if not live_scores:
            continue
        cat_scores = torch.nan_to_num(torch.cat(live_scores), nan=float("-inf"))
        cat_positions = torch.cat(live_positions)
        take = min(k, int(cat_scores.numel()))
        ordered = torch.sort(cat_scores, descending=True).values
        unique_kth = take == k and (
            int(cat_scores.numel()) == k or ordered[k - 1] > ordered[k]
        )
        got = [int(x) for x in selected[row].tolist() if x >= 0]
        assert len(set(got)) == take
        if unique_kth:
            expected = set(
                cat_positions[torch.topk(cat_scores, take).indices].cpu().tolist()
            )
            assert set(got) == expected


def _synthetic_bags(rows, splits, k, seed=3):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn(rows, splits, k, device=device, generator=generator)
    positions = (
        torch.arange(splits * k, device=device, dtype=torch.int32)
        .view(1, splits, k)
        .expand(rows, -1, -1)
        .contiguous()
    )
    counts = torch.full((rows, splits), k, dtype=torch.int32, device=device)
    return scores, positions, counts


def _run_direct_persistent(rows, splits, k, seed=3):
    scores, positions, counts = _synthetic_bags(rows, splits, k, seed=seed)
    workspace = persistent_split_topk_merge_workspace(scores.device, rows)
    _prepare_pass0_workspace(scores, counts, workspace)
    _, selected = split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
        precomputed_first_pass=True,
        persistent=True,
        persistent_workspace=workspace,
    )
    _assert_unique_kth_set(scores, positions, counts, selected, k)
    _, radix_pos = split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
        precomputed_first_pass=False,
    )
    for row in range(rows):
        got = set(int(x) for x in selected[row].tolist() if x >= 0)
        ref = set(int(x) for x in radix_pos[row].tolist() if x >= 0)
        assert got == ref
    return selected


def _run_persistent_from_stage_a(rows, splits, k, length, seed=47):
    case = _make_case(rows, length, 64, seed=seed)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    args = (
        case.q,
        packed,
        None,
        case.weights,
        case.lengths,
        case.block_tables,
    )
    scores, positions = flydsl_fp8_paged_mqa_topk(
        *args, k=k, num_splits=splits, persistent_merge=True
    )
    radix_scores, radix_positions = flydsl_fp8_paged_mqa_topk(
        *args, k=k, num_splits=splits, persistent_merge=False
    )
    reference = run_torch(case)
    lengths = case.lengths.reshape(-1)
    for row in range(rows):
        take = min(k, int(lengths[row]))
        got = [int(x) for x in positions[row].cpu().tolist() if x >= 0]
        radix_got = [int(x) for x in radix_positions[row].cpu().tolist() if x >= 0]
        assert len(set(got)) == take
        assert set(got) == set(radix_got), f"row {row} persistent vs radix set mismatch"
        row_ref = reference[row]
        checkAllclose(
            torch.sort(row_ref[positions[row].long()], descending=True).values.float(),
            torch.topk(row_ref, take, sorted=True).values.float(),
            rtol=2e-4,
            atol=2e-4,
            msg=f"persistent_from_stage_a row {row} scores",
        )
    return scores, positions


def test_persistent_split_topk_merge_small_and_nan():
    _require_supported_gpu()
    rows, splits, k = 2, 4, 128
    selected = _run_direct_persistent(rows, splits, k)
    assert selected.shape == (rows, k)

    counts = torch.tensor(
        [[128, 90, 128, 40], [80, 128, 70, 128]],
        dtype=torch.int32,
        device="cuda",
    )
    scores = torch.full((rows, splits, k), float("-inf"), device="cuda")
    positions = torch.full((rows, splits, k), -1, dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(59)
    for row in range(rows):
        base = 0
        for split in range(splits):
            count = int(counts[row, split])
            scores[row, split, :count] = torch.randn(
                count, generator=generator, device="cuda"
            )
            positions[row, split, :count] = torch.arange(
                base, base + count, dtype=torch.int32, device="cuda"
            )
            base += count
        scores[row, 0, :16] = float("nan")
    workspace = persistent_split_topk_merge_workspace(torch.device("cuda"), rows)
    _prepare_pass0_workspace(scores, counts, workspace)
    _, selected = split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
        precomputed_first_pass=True,
        persistent=True,
        persistent_workspace=workspace,
    )
    _assert_unique_kth_set(scores, positions, counts, selected, k)

    tie_scores = torch.full_like(scores, float("-inf"))
    for row in range(rows):
        for split in range(splits):
            tie_scores[row, split, : int(counts[row, split])] = 0
    _prepare_pass0_workspace(tie_scores, counts, workspace)
    values, tie_positions = split_topk_merge(
        tie_scores,
        positions,
        counts,
        k=k,
        precomputed_first_pass=True,
        persistent=True,
        persistent_workspace=workspace,
    )
    assert torch.equal(values, torch.zeros_like(values))
    for row in range(rows):
        chosen = [int(x) for x in tie_positions[row].tolist() if x >= 0]
        assert len(set(chosen)) == k
        assert min(chosen) >= 0


def test_persistent_production_shapes_subprocess():
    _require_supported_gpu()
    cases = (
        (8, 32, 2048, 8193, 90),
        (16, 32, 2048, 8193, 90),
        (32, 16, 2048, 8193, 90),
        (8, 8, 2048, 8193, 60),
    )
    env = os.environ.copy()
    root = "/home/samremes/dev/aiter-dsa-indexer-stage-a"
    env["PYTHONPATH"] = root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    for rows, splits, k, length, timeout in cases:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from op_tests.test_flydsl_fp8_paged_mqa_local_topk import "
                    "_run_persistent_from_stage_a; "
                    f"_run_persistent_from_stage_a({rows}, {splits}, {k}, {length})"
                ),
            ],
            cwd=root,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr[-4000:]


def test_persistent_refuses_missing_and_dirty_workspace():
    _require_supported_gpu()
    rows, splits, k = 2, 4, 128
    scores, positions, counts = _synthetic_bags(rows, splits, k)
    with pytest.raises(ValueError, match="persistent_workspace"):
        split_topk_merge(
            scores,
            positions,
            counts,
            k=k,
            persistent=True,
        )
    workspace = persistent_split_topk_merge_workspace(scores.device, rows)
    _prepare_pass0_workspace(scores, counts, workspace)
    split_topk_merge(
        scores,
        positions,
        counts,
        k=k,
        precomputed_first_pass=True,
        persistent=True,
        persistent_workspace=workspace,
    )
    with pytest.raises(RuntimeError, match="dirty"):
        split_topk_merge(
            scores,
            positions,
            counts,
            k=k,
            precomputed_first_pass=True,
            persistent=True,
            persistent_workspace=workspace,
        )


def test_persistent_b_only_second_launch_subprocess():
    _require_supported_gpu()
    env = os.environ.copy()
    root = "/home/samremes/dev/aiter-dsa-indexer-stage-a"
    env["PYTHONPATH"] = root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from op_tests.test_flydsl_fp8_paged_mqa_local_topk import "
                "test_persistent_refuses_missing_and_dirty_workspace; "
                "test_persistent_refuses_missing_and_dirty_workspace()"
            ),
        ],
        cwd=root,
        env=env,
        timeout=30,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_persistent_e2e_eager_and_graph_replay():
    _require_supported_gpu()
    rows, length, k, splits = 2, 8193, 2048, 4
    case = _make_case(rows, length, 64, seed=47)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    args = (
        case.q,
        packed,
        None,
        case.weights,
        case.lengths,
        case.block_tables,
    )
    reference = run_torch(case)
    scores, positions = flydsl_fp8_paged_mqa_topk(
        *args, k=k, num_splits=splits, persistent_merge=True
    )
    for _ in range(32):
        scores, positions = flydsl_fp8_paged_mqa_topk(
            *args, k=k, num_splits=splits, persistent_merge=True
        )
    torch.cuda.synchronize()
    for row in range(rows):
        expected = set(
            torch.topk(reference[row], k, sorted=False).indices.cpu().tolist()
        )
        got = set(int(x) for x in positions[row].cpu().tolist() if x >= 0)
        assert got == expected

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        flydsl_fp8_paged_mqa_topk(
            *args, k=k, num_splits=splits, persistent_merge=True
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_scores, replay_positions = flydsl_fp8_paged_mqa_topk(
                *args, k=k, num_splits=splits, persistent_merge=True
            )
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    for row in range(rows):
        expected = set(
            torch.topk(reference[row], k, sorted=False).indices.cpu().tolist()
        )
        got = set(int(x) for x in replay_positions[row].cpu().tolist() if x >= 0)
        assert got == expected


def test_persistent_packed_page64_e2e_k2048():
    _require_supported_gpu()
    rows, length, k, splits = 1, 8193, 2048, 2
    case = _make_case(rows, length, 64, seed=47)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    scores, positions = flydsl_fp8_paged_mqa_topk(
        case.q,
        packed,
        None,
        case.weights,
        case.lengths,
        case.block_tables,
        k=k,
        num_splits=splits,
        persistent_merge=True,
    )
    reference = run_torch(case)
    expected = torch.topk(reference[0], k, sorted=False).indices
    assert set(positions[0].cpu().tolist()) == set(expected.cpu().tolist())
    checkAllclose(
        reference[0][positions[0].long()].float(),
        scores[0].float(),
        rtol=2e-4,
        atol=2e-4,
        msg="persistent_e2e",
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_local_topk unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Experimental paged-MQA Stage A sweep",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    parser.add_argument("-l", "--length", type=int, nargs="*", default=[8193])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[128, 2048])
    parser.add_argument("--splits", type=int, nargs="*", default=[4])
    parser.add_argument("--page-size", type=int, nargs="*", default=[16, 64])
    args = parser.parse_args()

    rows = [
        benchmark_local_topk(batch, length, topk, splits, page_size)
        for batch, length, topk, splits, page_size in itertools.product(
            args.batch,
            args.length,
            args.topk,
            args.splits,
            args.page_size,
        )
    ]
    summary = pd.DataFrame(rows)
    aiter.logger.info(
        "fp8_paged_mqa_local_topk summary (markdown):\n%s",
        summary.to_markdown(index=False),
    )
    auto_rows = [
        benchmark_auto_topk(rows, next_n, 1_048_576, 2048, 64)
        for rows, next_n in ((8, 2), (16, 1), (32, 1))
    ]
    auto_summary = pd.DataFrame(auto_rows)
    aiter.logger.info(
        "fp8_paged_mqa_topk auto-dispatch summary (markdown):\n%s",
        auto_summary.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
