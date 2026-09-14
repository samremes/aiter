# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness proof for packed-varlen group-max pruning and harvest."""

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.ops.flydsl.fp8_paged_mqa_group_max_topk import (
    GROUP_SIZE,
    flydsl_fp8_paged_mqa_group_harvest,
    flydsl_fp8_paged_mqa_group_max,
    flydsl_fp8_paged_mqa_group_max_topk,
    flydsl_fp8_paged_mqa_position_map,
    group_space_split_bounds,
    torch_group_max_topk_prototype,
)
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_local_topk import (
    launch_fp8_paged_mqa_local_topk,
)
from op_tests.test_flydsl_fp8_paged_mqa_local_topk import (
    _make_packed_case,
    _pack_kv,
    _preshuffle_kv,
    _require_supported_gpu,
    run_torch,
)


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


def test_dense_harvest_public_pipeline_short_rows_and_dead_slot():
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
