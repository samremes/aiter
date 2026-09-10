# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT coverage for gfx950 packed Stage A and split-merge Stage B."""

from __future__ import annotations

import time
from typing import Any

from aiter.aot.flydsl.common import compile_only_env, override_env
from aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_local_topk import (
    SUPPORTED_K,
    compile_fp8_paged_mqa_local_topk,
    fp8_paged_mqa_local_topk_kernel_name,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.split_topk_merge import _build_split_topk_merge

_ARCH = "gfx950"
_PAGE_SIZE = 64
_PRODUCTION_SPLITS = (2, 4, 8, 16, 32, 64, 88, 128)


def _compile_stage_a_to_cache(**kwargs: Any) -> None:
    import flydsl.expr as fx
    import torch

    topk = int(kwargs["topk"])
    device = torch.device("cpu")
    rows = next_n = num_splits = max_pages = num_pages = 1
    q_fp8 = torch.empty((1, 1, 32, 128), dtype=torch.float8_e4m3fn, device=device)
    kv_cache = torch.empty((1, _PAGE_SIZE, 1, 132), dtype=torch.uint8, device=device)
    k_scales = torch.empty(
        (_PAGE_SIZE * 132) // 4, dtype=torch.float32, device=device
    )
    weights = torch.empty((1, 32), dtype=torch.float32, device=device)
    context_lens = torch.empty((1,), dtype=torch.int32, device=device)
    block_tables = torch.empty((1, 1), dtype=torch.int32, device=device)
    candidate_scores = torch.empty(
        (1, 1, topk), dtype=torch.float32, device=device
    )
    candidate_positions = torch.empty(
        (1, 1, topk), dtype=torch.int32, device=device
    )
    candidate_counts = torch.empty((1, 1), dtype=torch.int32, device=device)
    merge_histogram = torch.empty(
        (1, 1, 2048), dtype=torch.int32, device=device
    )
    merge_state = torch.empty((1, 6), dtype=torch.int32, device=device)

    launcher = compile_fp8_paged_mqa_local_topk(
        topk=topk,
        arch=str(kwargs["arch"]),
        preshuffled=bool(kwargs["preshuffled"]),
        page_size=int(kwargs["page_size"]),
        prepare_merge=bool(kwargs["prepare_merge"]),
        packed=bool(kwargs["packed"]),
        ordered_emit=bool(kwargs["ordered_emit"]),
    )
    with compile_only_env():
        _run_compiled(
            launcher,
            q_fp8,
            kv_cache,
            k_scales,
            weights,
            context_lens,
            block_tables,
            candidate_scores,
            candidate_positions,
            candidate_counts,
            merge_histogram,
            merge_state,
            rows,
            next_n,
            num_splits,
            max_pages,
            num_pages,
            fx.Stream(0),
        )


def _compile_split_merge_to_cache(**kwargs: Any) -> None:
    import flydsl.expr as fx
    import torch

    topk = int(kwargs["topk"])
    splits = int(kwargs["splits"])
    width = splits * topk
    device = torch.device("cpu")
    launcher = _build_split_topk_merge(
        topk,
        splits,
        bool(kwargs["precomputed_first_pass"]),
    )
    scores = torch.empty((1, width), dtype=torch.float32, device=device)
    positions = torch.empty((1, width), dtype=torch.int32, device=device)
    counts = torch.empty((1, splits), dtype=torch.int32, device=device)
    row_ends = torch.empty((1,), dtype=torch.int32, device=device)
    selected_positions = torch.empty((1, topk), dtype=torch.int32, device=device)
    selected_scores = torch.empty((1, topk), dtype=torch.float32, device=device)
    histogram = torch.empty((1, 1, 2048), dtype=torch.int32, device=device)
    state = torch.empty((1, 6), dtype=torch.int32, device=device)
    with compile_only_env():
        _run_compiled(
            launcher,
            scores,
            positions,
            counts,
            row_ends,
            selected_positions,
            selected_scores,
            histogram,
            state,
            width,
            1,
            width,
            1,
            fx.Stream(0),
        )


def collect_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for topk in SUPPORTED_K:
        # Public local bags are ordered; production e2e skips the unused sort
        # and prepares the first radix digit for Stage B.
        for prepare_merge, ordered_emit in ((False, True), (True, False)):
            jobs.append(
                {
                    "kernel_name": fp8_paged_mqa_local_topk_kernel_name(
                        topk=topk,
                        arch=_ARCH,
                        preshuffled=True,
                        packed=True,
                        prepare_merge=prepare_merge,
                        ordered_emit=ordered_emit,
                    ),
                    "kind": "stage_a",
                    "topk": topk,
                    "packed": True,
                    "preshuffled": True,
                    "page_size": _PAGE_SIZE,
                    "prepare_merge": prepare_merge,
                    "ordered_emit": ordered_emit,
                    "arch": _ARCH,
                }
            )
        for splits in _PRODUCTION_SPLITS:
            for precomputed_first_pass in (False, True):
                jobs.append(
                    {
                        "kernel_name": (
                            f"split_topk_merge_k{topk}_s{splits}_"
                            f"pm{int(precomputed_first_pass)}"
                        ),
                        "kind": "split_merge",
                        "topk": topk,
                        "splits": splits,
                        "precomputed_first_pass": precomputed_first_pass,
                    }
                )
    return jobs


def compile_one_config(**kwargs: Any) -> dict[str, Any]:
    kernel_name = str(kwargs.get("kernel_name", "?"))
    result: dict[str, Any] = {"kernel_name": kernel_name, "compile_time": None}
    t0 = time.time()
    try:
        kind = kwargs["kind"]
        if kind == "stage_a":
            with override_env("FLYDSL_GPU_ARCH", kwargs["arch"]):
                _compile_stage_a_to_cache(**kwargs)
        elif kind == "split_merge":
            with override_env("FLYDSL_GPU_ARCH", _ARCH):
                _compile_split_merge_to_cache(**kwargs)
        else:
            raise ValueError(f"unknown Stage A AOT kind {kind!r}")
        result["compile_time"] = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] compile  {kernel_name}: {exc}")
    return result
