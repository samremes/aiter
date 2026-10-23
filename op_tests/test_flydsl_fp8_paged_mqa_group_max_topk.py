# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness proof for packed-varlen group-max pruning and harvest."""

import argparse
import itertools

import aiter
import pandas as pd
import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fp8_paged_mqa_group_max_topk import (
    GROUP_SIZE,
    flydsl_fp8_paged_mqa_group_harvest,
    flydsl_fp8_paged_mqa_group_max,
    flydsl_fp8_paged_mqa_group_max_topk,
    flydsl_fp8_paged_mqa_logits_k_stationary,
    flydsl_fp8_paged_mqa_position_map,
    group_space_split_bounds,
    plan_group_max_num_splits,
    torch_group_max_topk_prototype,
)
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_fused_k2 import (
    fused_k2_parts,
    fused_k2_workspace_elems,
)
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_group_harvest import (
    launch_fp8_paged_mqa_group_harvest,
    launch_fp8_paged_mqa_position_map,
)
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_group_max import (
    launch_fp8_paged_mqa_group_max,
)
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_local_topk import (
    launch_fp8_paged_mqa_local_topk,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.kernels.topk_per_row_decode_persistent import (
    build_topk_per_row_decode_one_workgroup_module,
)
from aiter.ops.flydsl.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.topk import _hip_top_k_per_row_decode, top_k_per_row_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.test_flydsl_fp8_paged_mqa_local_topk import (
    HEAD_DIM,
    HEADS,
    _make_packed_case,
    _pack_kv,
    _preshuffle_kv,
    _require_supported_gpu,
    run_torch,
)

SUPPORTED_GFX = ["gfx950"]
PAGE_SIZE = 64
# One packed page token is an fp8 key row plus its fp32 scale.
TOKEN_BYTES = HEAD_DIM + 4


def _run_prototype(case, *, k):
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    return torch_group_max_topk_prototype(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        k=k,
    )


def _assert_valid_unstable_topk(case, result, *, k):
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = min(k, row_scores.numel())
        got = result.positions[row, :count].long()
        assert got.unique().numel() == count
        assert torch.all((got >= 0) & (got < row_scores.numel()))
        assert torch.all(result.positions[row, count:] == -1)
        assert torch.all(torch.isneginf(result.scores[row, count:]))
        if count == 0:
            continue
        threshold = torch.topk(row_scores, count, sorted=False).values.min()
        assert torch.all(row_scores[got] >= threshold)
        torch.testing.assert_close(
            result.scores[row, :count],
            row_scores[got],
            rtol=0,
            atol=0,
        )


def test_mixed_packed_widths_lengths_dead_slots_and_group_maxima():
    _require_supported_gpu()
    case = _make_packed_case(
        [2, 2, 2, 2, 1, 1, 1, 1],
        128,
        64,
        seed=211,
        independent_kv=True,
        r_max=16,
    )
    lengths = [0, 1, 15, 16, 17, 64, 65, 1, 15, 16, 17, 64]
    case.lengths[:] = 0
    case.lengths[: case.live_rows] = torch.tensor(
        lengths,
        dtype=torch.int32,
        device=case.lengths.device,
    )
    result = _run_prototype(case, k=4)
    reference = run_torch(case)

    assert (
        result.group_ends.tolist()
        == [(length + GROUP_SIZE - 1) // GROUP_SIZE for length in lengths] + [0] * 4
    )
    for row, scores in enumerate(reference):
        groups = (scores.numel() + GROUP_SIZE - 1) // GROUP_SIZE
        if groups:
            padded = torch.full(
                (groups * GROUP_SIZE,),
                float("-inf"),
                dtype=torch.float32,
                device=scores.device,
            )
            padded[: scores.numel()] = scores
            torch.testing.assert_close(
                result.group_max[row, :groups],
                padded.view(groups, GROUP_SIZE).amax(dim=1),
                rtol=0,
                atol=0,
            )
        assert torch.all(torch.isneginf(result.group_max[row, groups:]))
    _assert_valid_unstable_topk(case, result, k=4)


@pytest.mark.parametrize("width", [2, 4, 5, 8])
def test_packed_uniform_width_unique_kth_matches_oracle_set(width):
    _require_supported_gpu()
    case = _make_packed_case(
        [width, width],
        128,
        64,
        seed=223 + width,
        independent_kv=True,
        r_max=2 * width + 3,
    )
    case.lengths[: case.live_rows] = 65
    result = _run_prototype(case, k=4)
    reference = run_torch(case)
    for row in range(case.live_rows):
        ordered = torch.sort(reference[row], descending=True).values
        assert ordered[3] > ordered[4]
        expected = torch.topk(reference[row], 4, sorted=False).indices
        assert set(result.positions[row, :4].cpu().tolist()) == set(
            expected.cpu().tolist()
        )
    _assert_valid_unstable_topk(case, result, k=4)


def test_invalid_pages_remain_candidates_with_negative_infinity():
    _require_supported_gpu()
    case = _make_packed_case(
        [2],
        128,
        64,
        seed=239,
        independent_kv=True,
    )
    case.lengths[:] = torch.tensor([64, 65], dtype=torch.int32, device="cuda")
    case.block_tables[0, 0] = -1
    result = _run_prototype(case, k=65)
    assert torch.all(torch.isneginf(result.canonical_scores[0]))
    assert result.positions[0, :64].unique().numel() == 64
    assert set(result.positions[0, :64].cpu().tolist()) == set(range(64))
    assert result.positions[1, :65].unique().numel() == 65
    assert set(result.positions[1, :65].cpu().tolist()) == set(range(65))
    assert torch.isfinite(result.scores[1, :65]).any()


def test_tie_at_group_cutoff_and_mass_zero_accepts_any_valid_set():
    _require_supported_gpu()
    case = _make_packed_case(
        [2],
        128,
        64,
        seed=241,
        independent_kv=True,
    )
    case.lengths[:] = 128
    case.weights.zero_()
    result = _run_prototype(case, k=4)
    for row in range(case.live_rows):
        assert result.selected_group_ids[row, :4].unique().numel() == 4
        assert result.positions[row, :4].unique().numel() == 4
        assert torch.equal(
            result.scores[row, :4], torch.zeros_like(result.scores[row, :4])
        )
    _assert_valid_unstable_topk(case, result, k=4)


def test_scale_clamp_and_nan_canonicalization():
    _require_supported_gpu()
    case = _make_packed_case([2], 64, 64, seed=251, independent_kv=True)
    case.lengths[:] = 16
    case.scales[:, :16:3] = -1
    case.scales[:, 1:16:3] = float("nan")
    result = _run_prototype(case, k=16)
    reference = run_torch(case)
    for row in range(case.live_rows):
        torch.testing.assert_close(
            result.canonical_scores[row],
            reference[row],
            rtol=0,
            atol=0,
        )
        assert not torch.isnan(result.canonical_scores[row]).any()
        assert torch.all(result.canonical_scores[row][0:16:3] == 0)
        assert torch.all(result.canonical_scores[row][1:16:3] == 0)


@pytest.mark.parametrize("splits", [1, 8, 64])
def test_r2_pass1_group_max_matches_prototype(splits):
    _require_supported_gpu()
    case = _make_packed_case(
        [2, 2, 2, 2, 1, 1, 1, 1],
        128,
        64,
        seed=263,
        independent_kv=True,
        r_max=16,
    )
    lengths = [0, 1, 15, 16, 17, 64, 65, 1, 15, 16, 17, 64]
    case.lengths[:] = 0
    case.lengths[: case.live_rows] = torch.tensor(
        lengths,
        dtype=torch.int32,
        device=case.lengths.device,
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    expected = torch_group_max_topk_prototype(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        k=4,
    )
    actual, group_ends = flydsl_fp8_paged_mqa_group_max(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=2,
        num_splits=splits,
    )
    assert torch.equal(group_ends, expected.group_ends)
    for row in range(case.q.shape[0]):
        groups = int(group_ends[row])
        torch.testing.assert_close(
            actual[row, :groups],
            expected.group_max[row, :groups],
            rtol=2e-4,
            atol=2e-4,
        )


def test_r2_pass1_is_bit_identical_to_shared_one_row_scorer():
    _require_supported_gpu()
    case = _make_packed_case([2, 2], 128, 64, seed=269, independent_kv=True)
    case.lengths[:] = torch.tensor([15, 16, 17, 65], dtype=torch.int32, device="cuda")
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    actual, group_ends = flydsl_fp8_paged_mqa_group_max(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=2,
        num_splits=1,
    )
    reference = torch.empty((case.q.shape[0], 1, 128), device="cuda")
    positions = torch.empty_like(reference, dtype=torch.int32)
    counts = torch.empty((case.q.shape[0], 1), device="cuda", dtype=torch.int32)
    launch_fp8_paged_mqa_local_topk(
        case.q,
        packed,
        packed.view(torch.float32).reshape(-1),
        case.weights,
        case.lengths,
        case.indices,
        case.block_tables,
        reference,
        positions,
        counts,
        topk=128,
        num_splits=1,
        preshuffled=True,
        arch="gfx950",
        stream=torch.cuda.current_stream(),
        packed=True,
        xcd_row_fast=False,
        emit_group_max=True,
    )
    torch.cuda.synchronize()
    for row in range(case.q.shape[0]):
        groups = int(group_ends[row])
        torch.testing.assert_close(
            actual[row, :groups],
            reference[row, 0, :groups],
            rtol=0,
            atol=0,
        )


def _packed_pass1(case, packed, *, max_decode_width, num_splits, rows_per_cta):
    return flydsl_fp8_paged_mqa_group_max(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=max_decode_width,
        num_splits=num_splits,
        rows_per_cta=rows_per_cta,
    )


def _assert_group_max_matches_oracle(case, packed, actual, group_ends):
    expected = torch_group_max_topk_prototype(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        k=4,
    )
    assert torch.equal(group_ends, expected.group_ends)
    for row in range(case.q.shape[0]):
        groups = int(group_ends[row])
        torch.testing.assert_close(
            actual[row, :groups],
            expected.group_max[row, :groups],
            rtol=2e-4,
            atol=2e-4,
        )


def test_r4_pass1_mixed_widths_and_dead_slots_match_oracle():
    _require_supported_gpu()
    case = _make_packed_case(
        [2, 2, 2, 2, 1, 1, 1, 1],
        128,
        64,
        seed=401,
        independent_kv=True,
        r_max=16,
    )
    lengths = [0, 1, 15, 16, 17, 64, 65, 1, 15, 16, 17, 64]
    case.lengths[:] = 0
    case.lengths[: case.live_rows] = torch.tensor(
        lengths,
        dtype=torch.int32,
        device=case.lengths.device,
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    actual, group_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=4,
        num_splits=8,
        rows_per_cta=4,
    )
    _assert_group_max_matches_oracle(case, packed, actual, group_ends)


@pytest.mark.parametrize("width", [4, 5, 8])
def test_r4_pass1_uniform_width_matches_oracle(width):
    _require_supported_gpu()
    requests = 2
    case = _make_packed_case(
        [width] * requests,
        128,
        64,
        seed=409 + width,
        independent_kv=True,
        r_max=2 * width + 3,
    )
    case.lengths[: case.live_rows] = 65
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    actual, group_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=width,
        num_splits=8,
        rows_per_cta=4,
    )
    _assert_group_max_matches_oracle(case, packed, actual, group_ends)


def test_r4_pass1_live_zero_row_group_early_exit():
    _require_supported_gpu()
    case = _make_packed_case(
        [4, 4],
        128,
        64,
        seed=419,
        independent_kv=True,
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    actual, group_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=8,
        num_splits=8,
        rows_per_cta=4,
    )
    _assert_group_max_matches_oracle(case, packed, actual, group_ends)


@pytest.mark.parametrize("rows_per_cta", [2, 4])
@pytest.mark.parametrize("kv_load_cache_modifier", [0, 4])
def test_k_stationary_logits_matches_canonical_scores_and_preserves_tail(
    rows_per_cta,
    kv_load_cache_modifier,
):
    _require_supported_gpu()
    case = _make_packed_case([4, 4], 128, 64, seed=427, independent_kv=True)
    case.lengths[:] = torch.tensor(
        [0, 1, 15, 16, 17, 64, 65, 128],
        dtype=torch.int32,
        device="cuda",
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    logits = torch.full(
        (case.q.shape[0], 128),
        -123.0,
        dtype=torch.float32,
        device="cuda",
    )
    actual = flydsl_fp8_paged_mqa_logits_k_stationary(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=4,
        num_splits=8,
        logits=logits,
        rows_per_cta=rows_per_cta,
        kv_load_cache_modifier=kv_load_cache_modifier,
    )
    for row in range(case.q.shape[0]):
        length = int(case.lengths[row])
        positions = torch.arange(length, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(
            actual[row, :length],
            _rescore_positions(case, row, positions),
            rtol=2e-4,
            atol=2e-4,
        )
        assert torch.all(actual[row, length:] == -123.0)


def test_r4_pass1_is_bit_identical_to_r2_and_one_row_scorer():
    _require_supported_gpu()
    case = _make_packed_case([4, 4], 128, 64, seed=431, independent_kv=True)
    case.lengths[:] = torch.tensor(
        [15, 16, 17, 65, 1, 64, 0, 65],
        dtype=torch.int32,
        device="cuda",
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    r4, group_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=4,
        num_splits=1,
        rows_per_cta=4,
    )
    r2, r2_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=4,
        num_splits=1,
        rows_per_cta=2,
    )
    assert torch.equal(group_ends, r2_ends)
    reference = torch.empty((case.q.shape[0], 1, 128), device="cuda")
    positions = torch.empty_like(reference, dtype=torch.int32)
    counts = torch.empty((case.q.shape[0], 1), device="cuda", dtype=torch.int32)
    launch_fp8_paged_mqa_local_topk(
        case.q,
        packed,
        packed.view(torch.float32).reshape(-1),
        case.weights,
        case.lengths,
        case.indices,
        case.block_tables,
        reference,
        positions,
        counts,
        topk=128,
        num_splits=1,
        preshuffled=True,
        arch="gfx950",
        stream=torch.cuda.current_stream(),
        packed=True,
        xcd_row_fast=False,
        emit_group_max=True,
    )
    torch.cuda.synchronize()
    for row in range(case.q.shape[0]):
        groups = int(group_ends[row])
        torch.testing.assert_close(
            r4[row, :groups],
            r2[row, :groups],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            r4[row, :groups],
            reference[row, 0, :groups],
            rtol=0,
            atol=0,
        )


def test_dense_harvest_matches_oracle_and_pass1_group_max_bits():
    _require_supported_gpu()
    case = _make_packed_case(
        [2, 2, 2, 2, 1, 1, 1, 1],
        128,
        64,
        seed=443,
        independent_kv=True,
        r_max=16,
    )
    lengths = [0, 1, 15, 16, 17, 64, 65, 1, 15, 16, 17, 64]
    case.lengths[:] = 0
    case.lengths[: case.live_rows] = torch.tensor(
        lengths, dtype=torch.int32, device=case.lengths.device
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    group_max, group_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=2,
        num_splits=8,
        rows_per_cta=2,
    )
    selected_width = 8
    selected_group_ids = torch.arange(
        selected_width, device="cuda", dtype=torch.int32
    ).repeat(case.q.shape[0], 1)
    selected_group_counts = group_ends.clamp(max=selected_width)
    scores, positions, harvest_lengths = flydsl_fp8_paged_mqa_group_harvest(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        selected_group_ids,
        selected_group_counts,
    )
    torch.cuda.synchronize()
    reference = run_torch(case)
    assert torch.equal(harvest_lengths, selected_group_counts * GROUP_SIZE)
    for row, row_scores in enumerate(reference):
        count = int(selected_group_counts[row])
        for ordinal in range(count):
            group = int(selected_group_ids[row, ordinal])
            begin = group * GROUP_SIZE
            end = min(begin + GROUP_SIZE, row_scores.numel())
            slot = ordinal * GROUP_SIZE
            torch.testing.assert_close(
                scores[row, slot : slot + end - begin],
                row_scores[begin:end],
                rtol=2e-4,
                atol=2e-4,
            )
            assert torch.equal(
                scores[row, slot : slot + GROUP_SIZE].amax().view(torch.int32),
                group_max[row, group].view(torch.int32),
            )
            assert torch.equal(
                positions[row, slot : slot + end - begin],
                torch.arange(begin, end, device="cuda", dtype=torch.int32),
            )
            assert torch.all(positions[row, slot + end - begin : slot + 16] == -1)
        assert torch.all(positions[row, count * GROUP_SIZE :] == -1)
        assert torch.all(torch.isneginf(scores[row, count * GROUP_SIZE :]))


def test_dense_harvest_invalid_pages_and_scale_edges():
    _require_supported_gpu()
    case = _make_packed_case([2], 128, 64, seed=449, independent_kv=True)
    case.lengths[:] = torch.tensor([64, 65], dtype=torch.int32, device="cuda")
    case.block_tables[0, 0] = -1
    case.scales[:, :16:3] = -1
    case.scales[:, 1:16:3] = float("nan")
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    group_max, group_ends = _packed_pass1(
        case,
        packed,
        max_decode_width=2,
        num_splits=1,
        rows_per_cta=2,
    )
    selected_group_ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(
        case.q.shape[0], 1
    )
    selected_group_counts = group_ends.clamp(max=8)
    scores, positions, _ = flydsl_fp8_paged_mqa_group_harvest(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        selected_group_ids,
        selected_group_counts,
    )
    torch.cuda.synchronize()
    assert torch.all(torch.isneginf(scores[0, :64]))
    assert torch.equal(
        scores[0, :16].amax().view(torch.int32),
        group_max[0, 0].view(torch.int32),
    )
    for row in range(case.live_rows):
        count = int(selected_group_counts[row])
        for ordinal in range(count):
            group = int(selected_group_ids[row, ordinal])
            slot = ordinal * GROUP_SIZE
            assert torch.equal(
                scores[row, slot : slot + GROUP_SIZE].amax().view(torch.int32),
                group_max[row, group].view(torch.int32),
            )
        assert not torch.isnan(scores[row]).any()
    assert torch.equal(
        positions[0, :64],
        torch.arange(64, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(
        positions[1, :65],
        torch.arange(65, device="cuda", dtype=torch.int32),
    )


def test_plan_group_max_num_splits_fills_by_row_groups():
    assert plan_group_max_num_splits(8, 4, 65536, rows_per_cta=4, cu_count=256) == 64
    assert plan_group_max_num_splits(4, 8, 65536, rows_per_cta=4, cu_count=256) == 64
    assert plan_group_max_num_splits(2, 8, 65536, rows_per_cta=4, cu_count=256) == 128
    assert plan_group_max_num_splits(2, 4, 65536, rows_per_cta=4, cu_count=256) == 256
    assert plan_group_max_num_splits(1, 8, 65536, rows_per_cta=4, cu_count=256) == 256
    assert plan_group_max_num_splits(16, 2, 65536, rows_per_cta=4, cu_count=256) == 64


def test_fused_public_pipeline_short_rows_and_dead_slot():
    _require_supported_gpu()
    case = _make_packed_case([4], 32_768, 64, seed=457, independent_kv=True)
    case.lengths[:] = torch.tensor([0, 17, 64, 65], dtype=torch.int32, device="cuda")
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=4,
        fused=True,
        persistent=False,
    )
    torch.cuda.synchronize()
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = row_scores.numel()
        got = result.positions[row, :count].long()
        assert got.unique().numel() == count
        assert torch.all((got >= 0) & (got < count))
        torch.testing.assert_close(
            result.scores[row, :count],
            row_scores[got],
            rtol=2e-4,
            atol=2e-4,
        )
        assert torch.all(result.positions[row, count:] == -1)
        assert torch.all(torch.isneginf(result.scores[row, count:]))

    _require_supported_gpu()
    case = _make_packed_case([4], 32_768, 64, seed=457, independent_kv=True)
    case.lengths[:] = torch.tensor([0, 17, 64, 65], dtype=torch.int32, device="cuda")
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=4,
        fused=False,
    )
    torch.cuda.synchronize()
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = row_scores.numel()
        got = result.positions[row, :count].long()
        assert got.unique().numel() == count
        assert torch.all((got >= 0) & (got < count))
        torch.testing.assert_close(
            result.scores[row, :count],
            row_scores[got],
            rtol=2e-4,
            atol=2e-4,
        )
        assert torch.all(result.positions[row, count:] == -1)
        assert torch.all(torch.isneginf(result.scores[row, count:]))


def test_position_map_handles_arbitrary_group_order_and_invalid_metadata():
    _require_supported_gpu()
    lengths = torch.tensor([0, 1, 15, 16, 17, 64, 65], dtype=torch.int32, device="cuda")
    selected_group_ids = torch.tensor(
        [
            [3, 0, 2, 1],
            [0, 3, 2, 1],
            [0, 3, 2, 1],
            [0, 3, 2, 1],
            [1, 0, 3, 2],
            [3, 1, 0, 2],
            [4, 2, 0, 1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    counts = torch.tensor([0, 1, 1, 1, 2, 4, 4], dtype=torch.int32, device="cuda")
    ordinals = torch.tensor(
        [
            [-1, 0, 15, 16, 63, 64, 7, 31],
            [0, 1, 15, 16, -1, 64, 7, 31],
            [0, 1, 14, 15, 16, 64, -1, 31],
            [0, 1, 14, 15, 16, 64, -1, 31],
            [0, 1, 15, 16, 17, 31, 32, -1],
            [0, 15, 16, 31, 32, 47, 48, 63],
            [0, 1, 15, 16, 31, 32, 48, 63],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    actual = flydsl_fp8_paged_mqa_position_map(
        ordinals, selected_group_ids, counts, lengths
    )
    expected = torch.full_like(ordinals, -1)
    for row in range(lengths.numel()):
        for column in range(ordinals.shape[1]):
            ordinal = int(ordinals[row, column])
            group_ordinal = ordinal // GROUP_SIZE if ordinal >= 0 else -1
            if 0 <= group_ordinal < int(counts[row]):
                group = int(selected_group_ids[row, group_ordinal])
                logical = group * GROUP_SIZE + ordinal % GROUP_SIZE
                if group >= 0 and logical < int(lengths[row]):
                    expected[row, column] = logical
    assert torch.equal(actual, expected)


def test_fused_public_pipeline_mass_zero_unstable_set():
    _require_supported_gpu()
    case = _make_packed_case([4], 32_768, 64, seed=461, independent_kv=True)
    case.lengths[:] = 65
    case.weights.zero_()
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=4,
        fused=True,
        persistent=False,
    )
    torch.cuda.synchronize()
    for row in range(case.live_rows):
        positions = result.positions[row, :65]
        assert positions.unique().numel() == 65
        assert torch.all((positions >= 0) & (positions < 65))


def test_dense_harvest_public_pipeline_mass_zero_unstable_set():
    _require_supported_gpu()
    case = _make_packed_case([4], 32_768, 64, seed=461, independent_kv=True)
    case.lengths[:] = 65
    case.weights.zero_()
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=4,
        fused=False,
    )
    torch.cuda.synchronize()
    for row in range(case.live_rows):
        positions = result.positions[row, :65]
        assert positions.unique().numel() == 65
        assert torch.all((positions >= 0) & (positions < 65))
        assert torch.equal(
            result.scores[row, :65], torch.zeros_like(result.scores[row, :65])
        )
        assert torch.all(result.positions[row, 65:] == -1)
        assert torch.all(torch.isneginf(result.scores[row, 65:]))


def test_dense_harvest_public_pipeline_long_partial_tail_is_exact():
    """Pin exactness where the harvest window contains nonexistent lanes.

    `context_len = 40007` gives 2501 groups, so more groups than `k`, and a
    final group holding 7 valid positions and 9 nonexistent ones. Those 9 sit
    inside `harvest_lengths = count * 16` and score `-inf`.

    Ties would make "exact" undecidable, so the scale epilogue is used as a
    switch: `total * max(scale, 0)` pins a position to exactly `0` when its
    scale is zero, and nonnegative weights keep `total` nonnegative. Exactly
    `k` positions are left strictly positive, including the whole valid tail,
    so the correct answer is one known set rather than any of many.
    """
    _require_supported_gpu()
    k = 2048
    context_len = 40_007
    case = _make_packed_case([2], 65_536, 64, seed=977, independent_kv=True)
    case.weights.abs_()
    case.lengths[:] = context_len

    generator = torch.Generator(device="cuda").manual_seed(31337)
    tail_start = (context_len // GROUP_SIZE) * GROUP_SIZE
    tail = torch.arange(tail_start, context_len, device="cuda")
    body = torch.randperm(tail_start, device="cuda", generator=generator)[
        : k - tail.numel()
    ]
    expected = torch.cat([tail, body]).sort().values

    pages = case.block_tables[0][expected // 64].long()
    tokens = expected % 64
    scales = torch.zeros_like(case.scales)
    # A zero scale here would collapse a chosen position back to 0 and close
    # the gap that makes the answer unique.
    scales[pages, tokens] = case.scales[pages, tokens].clamp(min=0.25)
    case.scales.copy_(scales)

    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    reference = run_torch(case)
    for row_scores in reference:
        assert torch.equal((row_scores > 0).nonzero().flatten(), expected)

    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=2,
        fused=False,
    )
    torch.cuda.synchronize()

    for row, row_scores in enumerate(reference):
        got = result.positions[row].long()
        assert torch.equal(got.sort().values, expected)
        assert got[(got >= tail_start) & (got < context_len)].numel() == tail.numel()
        torch.testing.assert_close(
            result.scores[row].sort().values,
            row_scores[got.sort().values].sort().values,
            rtol=2e-4,
            atol=2e-4,
        )


def test_fused_public_pipeline_long_partial_tail_is_exact():
    _require_supported_gpu()
    k = 2048
    context_len = 40_007
    case = _make_packed_case([2], 65_536, 64, seed=977, independent_kv=True)
    case.weights.abs_()
    case.lengths[:] = context_len

    generator = torch.Generator(device="cuda").manual_seed(31337)
    tail_start = (context_len // GROUP_SIZE) * GROUP_SIZE
    tail = torch.arange(tail_start, context_len, device="cuda")
    body = torch.randperm(tail_start, device="cuda", generator=generator)[
        : k - tail.numel()
    ]
    expected = torch.cat([tail, body]).sort().values

    pages = case.block_tables[0][expected // 64].long()
    tokens = expected % 64
    scales = torch.zeros_like(case.scales)
    scales[pages, tokens] = case.scales[pages, tokens].clamp(min=0.25)
    case.scales.copy_(scales)

    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=2,
        fused=True,
        persistent=False,
    )
    torch.cuda.synchronize()
    for row in range(case.live_rows):
        got = result.positions[row].long()
        assert torch.equal(got.sort().values, expected)
        assert got[(got >= tail_start) & (got < context_len)].numel() == tail.numel()


@pytest.mark.xfail(
    reason="persistent fused K2 still drops short-row positions; compact fused is default",
    strict=False,
)
def test_fused_persistent_public_pipeline_short_rows_and_dead_slot():
    _require_supported_gpu()
    case = _make_packed_case([4], 32_768, 64, seed=457, independent_kv=True)
    case.lengths[:] = torch.tensor([0, 17, 64, 65], dtype=torch.int32, device="cuda")
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    rows = int(case.q.shape[0])
    workspace = torch.zeros(
        fused_k2_workspace_elems(rows), dtype=torch.int32, device="cuda"
    )
    result = flydsl_fp8_paged_mqa_group_max_topk(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        case.indices,
        case.query_start_loc,
        case.decode_lens,
        max_decode_width=4,
        fused=True,
        persistent=True,
        fused_workspace=workspace,
    )
    torch.cuda.synchronize()
    reference = run_torch(case)
    for row, row_scores in enumerate(reference):
        count = row_scores.numel()
        got = result.positions[row, :count].long()
        assert got.unique().numel() == count
        assert torch.all((got >= 0) & (got < count))
        torch.testing.assert_close(
            result.scores[row, :count],
            row_scores[got],
            rtol=2e-4,
            atol=2e-4,
        )
        assert torch.all(result.positions[row, count:] == -1)
        assert torch.all(torch.isneginf(result.scores[row, count:]))


def test_fused_workspace_restore_allows_repeated_and_concurrent_replay():
    _require_supported_gpu()
    case = _make_packed_case([4], 32_768, 64, seed=611, independent_kv=True)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    rows = int(case.q.shape[0])
    workspace = torch.zeros(
        fused_k2_workspace_elems(rows), dtype=torch.int32, device="cuda"
    )

    def _launch(*, persistent, fused_workspace=workspace):
        return flydsl_fp8_paged_mqa_group_max_topk(
            case.q,
            packed,
            case.weights,
            case.lengths,
            case.block_tables,
            case.indices,
            case.query_start_loc,
            case.decode_lens,
            max_decode_width=4,
            fused=True,
            persistent=persistent,
            fused_workspace=fused_workspace,
        )

    first = _launch(persistent=False)
    second = _launch(persistent=False)
    torch.cuda.synchronize()
    reference = run_torch(case)
    for result in (first, second):
        for row, row_scores in enumerate(reference):
            count = min(2048, row_scores.numel())
            got = result.positions[row, :count].long()
            assert got.unique().numel() == count
            torch.testing.assert_close(
                result.scores[row, :count],
                row_scores[got],
                rtol=2e-4,
                atol=2e-4,
            )
    other = torch.zeros_like(workspace)
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    with torch.cuda.stream(s0):
        c0 = _launch(persistent=False, fused_workspace=workspace)
    with torch.cuda.stream(s1):
        c1 = _launch(persistent=False, fused_workspace=other)
    torch.cuda.synchronize()
    for result in (c0, c1):
        for row, row_scores in enumerate(reference):
            count = min(2048, row_scores.numel())
            got = result.positions[row, :count].long()
            assert got.unique().numel() == count


@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 64, 65])
@pytest.mark.parametrize("splits", [1, 8, 64])
def test_group_space_split_ownership_never_bisects_groups(length, splits):
    intervals = [
        group_space_split_bounds(length, split, splits) for split in range(splits)
    ]
    assert intervals[0][0] == 0
    assert intervals[-1][1] == ((length + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    for split, (begin, end) in enumerate(intervals):
        assert begin % GROUP_SIZE == 0
        assert end % GROUP_SIZE == 0
        assert begin <= end
        if split:
            assert intervals[split - 1][1] == begin


def test_packed_abi_rejects_missing_indices_and_rectangles():
    _require_supported_gpu()
    case = _make_packed_case([2, 2], 64, 64, seed=257)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    args = (
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
    )
    with pytest.raises(ValueError, match="indices is required"):
        torch_group_max_topk_prototype(
            *args,
            None,
            case.query_start_loc,
            case.decode_lens,
            k=4,
        )
    rectangle = case.q.reshape(2, 2, 32, 128)
    with pytest.raises(ValueError, match="q_fp8 must be packed"):
        torch_group_max_topk_prototype(
            rectangle,
            packed,
            case.weights,
            case.lengths,
            case.block_tables,
            case.indices,
            case.query_start_loc,
            case.decode_lens,
            k=4,
        )


def _rescore_positions(case, row, positions):
    """Independent fp32 scores for just the positions the pipeline returned.

    A full-length oracle needs a `[HEADS, L, HEAD_DIM]` intermediate, which is
    not affordable at the 1M gate inside a sweep. Exact set membership is
    covered by the pytest cases above, so the benchmark checks the other half
    of the contract: that each returned score is the canonical fp32 score of
    the position it is paired with.
    """
    q = case.q.reshape(-1, HEADS, HEAD_DIM)[row].float()
    flat_kv = case.kv.reshape(-1, HEAD_DIM)
    flat_scales = case.scales.reshape(-1)
    logical = positions.long()
    live = logical >= 0
    safe = torch.where(live, logical, torch.zeros_like(logical))
    pages = case.block_tables[int(case.indices[row]), safe // PAGE_SIZE]
    page_ok = live & (pages >= 0) & (pages < case.kv.shape[0])
    physical = torch.where(page_ok, pages, 0) * PAGE_SIZE + safe % PAGE_SIZE
    keys = flat_kv[physical].float()
    activated = torch.relu(torch.sum(q[:, None, :] * keys[None, :, :], dim=-1))
    weighted = torch.sum(case.weights[row, :, None] * activated, dim=0)
    scale = torch.fmax(
        flat_scales[physical],
        torch.zeros((), dtype=torch.float32, device=q.device),
    )
    scores = weighted * scale
    return torch.where(
        page_ok & ~torch.isnan(scores),
        scores,
        torch.full_like(scores, float("-inf")),
    )


def _benchmark_case(rows, next_n, length, k):
    """Build the case plus the caller-owned buffers a serving graph would hold."""
    requests = rows // next_n
    case = _make_packed_case(
        [next_n] * requests,
        length,
        PAGE_SIZE,
        seed=503 + next_n + requests,
        independent_kv=True,
    )
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    group_width = length // GROUP_SIZE
    buffers = {
        "group_max": torch.empty(
            (rows, group_width), dtype=torch.float32, device="cuda"
        ),
        "group_ends": torch.empty((rows,), dtype=torch.int32, device="cuda"),
        "selected_group_ids": torch.empty(
            (rows, k), dtype=torch.int32, device="cuda"
        ),
        "harvest_scores": torch.empty(
            (rows, k * GROUP_SIZE), dtype=torch.float32, device="cuda"
        ),
        "harvest_lengths": torch.empty((rows,), dtype=torch.int32, device="cuda"),
        "final_ordinals": torch.empty((rows, k), dtype=torch.int32, device="cuda"),
        "final_scores": torch.empty((rows, k), dtype=torch.float32, device="cuda"),
        "final_positions": torch.empty((rows, k), dtype=torch.int32, device="cuda"),
    }
    return case, packed, buffers


def _pipeline_stages(case, packed, buffers, *, rows, next_n, k):
    """The five launches, each callable on its own for a stage breakdown.

    Every stage is idempotent given its predecessor has run once, so replaying
    one stage in isolation measures it without disturbing the others.
    """
    stream = torch.cuda.current_stream()
    group_max = buffers["group_max"]
    group_ends = buffers["group_ends"]
    selected_group_ids = buffers["selected_group_ids"]
    harvest_scores = buffers["harvest_scores"]
    harvest_lengths = buffers["harvest_lengths"]
    final_ordinals = buffers["final_ordinals"]
    final_scores = buffers["final_scores"]
    final_positions = buffers["final_positions"]

    def k1_group_max():
        launch_fp8_paged_mqa_group_max(
            case.q,
            packed,
            case.weights,
            case.lengths,
            case.query_start_loc,
            case.decode_lens,
            case.block_tables,
            group_max,
            group_ends,
            max_decode_width=next_n,
            num_splits=64,
            arch=get_gfx(),
            stream=stream,
            rows_per_cta=4,
            q_in_lds=False,
        )
        return group_max

    def k2_group_topk():
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
            values=None,
        )
        return selected_group_ids

    def k3_harvest():
        launch_fp8_paged_mqa_group_harvest(
            case.q,
            packed,
            case.weights,
            case.lengths,
            case.block_tables,
            case.indices,
            selected_group_ids,
            group_ends,
            harvest_scores,
            selected_group_ids,
            harvest_lengths,
            arch=get_gfx(),
            stream=stream,
            emit_positions=False,
        )
        return harvest_scores

    def final_topk():
        top_k_per_row_decode(
            logits=harvest_scores,
            next_n=1,
            seqLens=harvest_lengths,
            indices=final_ordinals,
            numRows=rows,
            stride0=harvest_scores.stride(0),
            stride1=1,
            k=k,
            stable=False,
            values=final_scores,
        )
        return final_scores

    def position_map():
        launch_fp8_paged_mqa_position_map(
            final_ordinals,
            selected_group_ids,
            group_ends,
            case.lengths,
            final_positions,
            arch=get_gfx(),
            stream=stream,
        )
        return final_positions

    return {
        "k1_group_max": k1_group_max,
        "k2_group_topk": k2_group_topk,
        "k3_harvest": k3_harvest,
        "final_topk": final_topk,
        "position_map": position_map,
    }


@benchmark()
def benchmark_group_max_topk(rows, next_n, length, k):
    case, packed, buffers = _benchmark_case(rows, next_n, length, k)
    stages = _pipeline_stages(
        case, packed, buffers, rows=rows, next_n=next_n, k=k
    )

    def preallocated():
        for stage in stages.values():
            stage()
        return buffers["final_scores"], buffers["final_positions"]

    def public_wrapper():
        result = flydsl_fp8_paged_mqa_group_max_topk(
            case.q,
            packed,
            case.weights,
            case.lengths,
            case.block_tables,
            case.indices,
            case.query_start_loc,
            case.decode_lens,
            max_decode_width=next_n,
            k=k,
        )
        return result.scores, result.positions

    candidates = {
        "preallocated": preallocated,
        "public_wrapper": public_wrapper,
    }

    # K1 scores every position; K3 re-scores the selected groups densely. Both
    # do the arithmetic per row, so FLOPs count rows.
    flops = 2 * rows * (length + k * GROUP_SIZE) * HEADS * HEAD_DIM
    # Bytes count what has to leave HBM, which is not the same as per-row
    # traffic: the sibling rows of one request walk the same KV pages in K1,
    # and R=4 exists to exploit that. K3 gathers per row, because siblings
    # select different groups.
    nbytes = (
        case.q.numel() * case.q.element_size()
        + case.weights.numel() * case.weights.element_size()
        + (rows // next_n) * length * TOKEN_BYTES
        + rows * k * GROUP_SIZE * TOKEN_BYTES
        + rows * (length // GROUP_SIZE) * 4 * 2
        + rows * k * GROUP_SIZE * 4 * 2
        + rows * k * 8
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (scores, positions), us = run_perftest(fn)
        err = checkAllclose(
            _rescore_positions(case, 0, positions[0]),
            scores[0].float(),
            rtol=2e-4,
            atol=2e-4,
            msg=f"{name}: group-max topk",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def benchmark_group_max_topk_stages(rows, next_n, length, k):
    """Per-stage attribution, replayed in isolation.

    Each stage is timed by replaying only itself, so its inputs stay resident
    across iterations. That makes these numbers cache-warm lower bounds rather
    than the stage's cost inside the pipeline, where the previous stage has
    just evicted what this one wants. Use `benchmark_group_max_topk` for the
    cost of the pipeline and this table only to attribute within it.
    """
    case, packed, buffers = _benchmark_case(rows, next_n, length, k)
    stages = _pipeline_stages(
        case, packed, buffers, rows=rows, next_n=next_n, k=k
    )
    for stage in stages.values():
        stage()
    torch.cuda.synchronize()

    group_width = length // GROUP_SIZE
    harvest_width = k * GROUP_SIZE
    # Per-stage roofline. The two selectors and the map do no arithmetic, so
    # their TFLOPS column is zero by construction and only TB/s is meaningful.
    # K1's KV bytes are per request because sibling rows share those pages.
    roofline = {
        "k1_group_max": (
            2 * rows * length * HEADS * HEAD_DIM,
            (rows // next_n) * length * TOKEN_BYTES + rows * group_width * 4,
        ),
        "k2_group_topk": (0, rows * group_width * 4 + rows * k * 4),
        "k3_harvest": (
            2 * rows * harvest_width * HEADS * HEAD_DIM,
            rows * harvest_width * TOKEN_BYTES + rows * harvest_width * 4,
        ),
        "final_topk": (0, rows * harvest_width * 4 + rows * k * 8),
        "position_map": (0, rows * k * 4 * 2),
    }

    # One correctness check for the pipeline the stages compose, rather than a
    # per-stage column: the stages are sequential pieces of one result, not
    # interchangeable candidates.
    ret = {
        "gfx": get_gfx(),
        "pipeline err": checkAllclose(
            _rescore_positions(case, 0, buffers["final_positions"][0]),
            buffers["final_scores"][0].float(),
            rtol=2e-4,
            atol=2e-4,
            msg="group-max topk stages",
        ),
    }
    for name, fn in stages.items():
        _, us = run_perftest(fn)
        flops, nbytes = roofline[name]
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def _assert_selector_matches(
    logits, row_ends, actual_indices, reference_indices, actual_values=None
):
    """Check unstable TopK membership and optional score correspondence."""
    for row in range(logits.shape[0]):
        count = min(int(row_ends[row]), actual_indices.shape[1])
        actual = actual_indices[row, :count].long()
        reference = reference_indices[row, :count].long()
        assert actual.unique().numel() == count
        torch.testing.assert_close(
            logits[row, actual].sort().values,
            logits[row, reference].sort().values,
            rtol=0,
            atol=0,
        )
        if actual_values is not None:
            torch.testing.assert_close(
                actual_values[row, :count],
                logits[row, actual],
                rtol=0,
                atol=0,
            )


@benchmark()
def benchmark_group_max_topk_selectors(rows, next_n, length, k):
    """Matched selector comparison at the exact K2 and final-stage shapes."""
    case, packed, buffers = _benchmark_case(rows, next_n, length, k)
    stages = _pipeline_stages(
        case, packed, buffers, rows=rows, next_n=next_n, k=k
    )
    stages["k1_group_max"]()
    stages["k2_group_topk"]()
    stages["k3_harvest"]()
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    wave_size = 64
    ret = {"gfx": get_gfx()}
    selector_specs = {
        "k2": (
            buffers["group_max"],
            buffers["group_ends"],
            False,
        ),
        "final": (
            buffers["harvest_scores"],
            buffers["harvest_lengths"],
            True,
        ),
    }

    for stage_name, (logits, row_ends, write_values) in selector_specs.items():
        outputs = {}

        def make_outputs():
            indices = torch.empty((rows, k), dtype=torch.int32, device="cuda")
            values = (
                torch.empty((rows, k), dtype=torch.float32, device="cuda")
                if write_values
                else None
            )
            return indices, values

        hip_indices, hip_values = make_outputs()

        def hip_one_block():
            _hip_top_k_per_row_decode(
                logits,
                1,
                row_ends,
                hip_indices,
                rows,
                logits.stride(0),
                1,
                k,
                False,
                hip_values,
            )
            return hip_indices

        outputs["hip_one_block"] = (hip_one_block, hip_indices, hip_values)

        multi_indices, multi_values = make_outputs()

        def flydsl_multi():
            flydsl_top_k_per_row_decode(
                logits,
                1,
                row_ends,
                multi_indices,
                rows,
                logits.stride(0),
                1,
                k,
                False,
                multi_values,
            )
            return multi_indices

        outputs["flydsl_multi"] = (flydsl_multi, multi_indices, multi_values)

        one_wg_indices, one_wg_values = make_outputs()
        one_wg_launcher = build_topk_per_row_decode_one_workgroup_module(
            k,
            wave_size=wave_size,
            write_values=write_values,
        )

        def flydsl_one_wg():
            _run_compiled(
                one_wg_launcher,
                logits,
                row_ends,
                one_wg_indices,
                one_wg_values if one_wg_values is not None else logits,
                logits.shape[1],
                1,
                logits.stride(0),
                rows,
                stream,
            )
            return one_wg_indices

        outputs["flydsl_one_wg"] = (
            flydsl_one_wg,
            one_wg_indices,
            one_wg_values,
        )

        hip_one_block()
        torch.cuda.synchronize()
        for implementation, (fn, indices, values) in outputs.items():
            _, us = run_perftest(fn)
            _assert_selector_matches(
                logits, row_ends, indices, hip_indices, values
            )
            ret[f"{stage_name} {implementation} us"] = us

    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_group_max_topk unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Packed-varlen group-max pruning and dense-harvest sweep",
    )
    parser.add_argument("-r", "--rows", type=int, nargs="*", default=[32])
    parser.add_argument("-n", "--next-n", type=int, nargs="*", default=[4, 8])
    parser.add_argument("-l", "--length", type=int, nargs="*", default=[1_048_576])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[2048])
    args = parser.parse_args()

    axes = list(itertools.product(args.rows, args.next_n, args.length, args.topk))

    summary = pd.DataFrame(
        [benchmark_group_max_topk(*axis) for axis in axes]
    )
    aiter.logger.info(
        "fp8_paged_mqa_group_max_topk summary (markdown):\n%s",
        summary.to_markdown(index=False),
    )

    stage_summary = pd.DataFrame(
        [benchmark_group_max_topk_stages(*axis) for axis in axes]
    )
    aiter.logger.info(
        "fp8_paged_mqa_group_max_topk stage summary (markdown):\n%s",
        stage_summary.to_markdown(index=False),
    )

    selector_summary = pd.DataFrame(
        [benchmark_group_max_topk_selectors(*axis) for axis in axes]
    )
    aiter.logger.info(
        "fp8_paged_mqa_group_max_topk selector summary (markdown):\n%s",
        selector_summary.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
