# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark suite for experimental paged-MQA Stage A."""

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import pytest
import torch

pytest.importorskip("flydsl")

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_local_topk
from aiter.ops.flydsl.fp8_paged_mqa_local_topk import merge_local_topk_candidates
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


def _require_supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/HIP GPU")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(":", 1)[0]
    if arch not in SUPPORTED_GFX:
        pytest.skip(f"Stage A is not yet supported on {arch}")
    return arch


def _make_case(rows, length, page_size, *, seed=17, ragged=False):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    pages = max(1, (length + page_size - 1) // page_size)
    q = (
        torch.randn(rows, 1, HEADS, HEAD_DIM, device=device, generator=generator) * 0.25
    ).to(torch.float8_e4m3fn)
    kv = (
        torch.randn(pages, page_size, HEAD_DIM, device=device, generator=generator)
        * 0.25
    ).to(torch.float8_e4m3fn)
    scales = torch.rand(
        pages, page_size, device=device, generator=generator, dtype=torch.float32
    )
    weights = torch.randn(
        rows, HEADS, device=device, generator=generator, dtype=torch.float32
    )
    lengths = torch.full((rows,), length, device=device, dtype=torch.int32)
    if ragged and rows > 1:
        lengths[0] = 0
        lengths[-1] = max(0, length - 3)
    physical_order = torch.randperm(pages, device=device, generator=generator)
    block_tables = physical_order.repeat(rows, 1).to(torch.int32)
    return Case(q, kv, scales, weights, lengths, block_tables)


def _preshuffle_kv(kv):
    return shuffle_weight(kv, layout=(16, 16)).contiguous()


def run_torch(case):
    """Independent FP32 oracle with explicit page mapping and epilogue order."""
    rows = case.q.shape[0]
    page_size = case.kv.shape[1]
    flat_kv = case.kv.reshape(-1, HEAD_DIM).float()
    flat_scales = case.scales.reshape(-1)
    outputs = []
    for row in range(rows):
        length = int(case.lengths[row].item())
        logical = torch.arange(length, device=case.q.device)
        physical = (
            case.block_tables[row, logical // page_size] * page_size
            + logical % page_size
        )
        keys = flat_kv[physical]
        dots = torch.sum(case.q[row, 0].float()[:, None, :] * keys[None, :, :], dim=-1)
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
                checkAllclose(
                    row_scores[got_positions].float(),
                    scores[row, split, :count].float(),
                    rtol=2e-4,
                    atol=2e-4,
                    msg=f"row={row} split={split} selected scores",
                )
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
    expected = torch.topk(reference[0], k, sorted=False).indices
    assert set(final_positions[0].cpu().tolist()) == set(expected.cpu().tolist())


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
        case.q.numel()
        + case.kv.numel()
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


if __name__ == "__main__":
    main()
