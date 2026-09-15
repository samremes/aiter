from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter import dtypes
from aiter.ops.flydsl.kernels.mla_decode_varlen import (
    OCCUPANCY,
    launch_build_varlen_work_info,
    launch_merge_varlen_partials,
    launch_mla_fwd_decode_m16x8_fp8_fp8,
    launch_zero_varlen_empty_rows,
)
from aiter.ops.flydsl.kernels.mla_decode_varlen_h8 import OCCUPANCY as H8_OCCUPANCY
from aiter.ops.flydsl.kernels.mla_decode_varlen_h8 import (
    launch_build_compact_varlen_work_h8,
    launch_finalize_varlen_h8,
    launch_mla_decode_varlen_h8,
)

_MAX_SIGNED_I32_BYTE_SPAN = 2**31


def _require_i32_byte_span(name: str, nbytes: int) -> None:
    if nbytes >= _MAX_SIGNED_I32_BYTE_SPAN:
        raise ValueError(
            f"FlyDSL adaptive MLA requires `{name}` to be smaller than 2 GiB "
            "because the kernel uses signed 32-bit byte offsets"
        )


@dataclass
class MlaDecodeVarlenWorkspace:
    max_rows: int
    num_heads: int
    num_kv_splits: int | None
    stage_splits: int
    work_indptr: torch.Tensor
    work_info: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    row_splits: torch.Tensor
    finalize_count: torch.Tensor
    finalize_jobs: torch.Tensor


@dataclass(frozen=True)
class MlaDecodeVarlenPrepared:
    workspace: MlaDecodeVarlenWorkspace
    query_start_loc: torch.Tensor
    kv_indptr: torch.Tensor
    num_requests: int
    num_kv_splits: int | None


def _normalize_num_kv_splits(num_heads: int, num_kv_splits: int | None) -> int | None:
    if num_kv_splits is None:
        return None if num_heads == 8 else 16
    normalized = int(num_kv_splits)
    if normalized < 1:
        raise ValueError("`num_kv_splits` must be positive")
    return normalized


def create_mla_decode_varlen_workspace(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    *,
    num_kv_splits: int | None = None,
) -> MlaDecodeVarlenWorkspace:
    max_rows, num_heads, _ = q.shape
    head_ctas = 2 if num_heads == 128 else 1
    requested_splits = _normalize_num_kv_splits(num_heads, num_kv_splits)
    if tuple(o.shape) != (max_rows, num_heads, 512):
        raise ValueError("`o` must have shape [max_rows, num_heads, 512]")
    stage_splits = 16 if requested_splits is None else requested_splits
    total = max_rows * stage_splits
    work_items = total * head_ctas
    partial_slots = 0 if stage_splits == 1 else total
    spans = {
        "q": q.nbytes,
        "kv_buffer": kv_buffer.nbytes,
        "o": o.nbytes,
        "workspace.work_indptr": work_items * 2 * torch.int32.itemsize,
        "workspace.work_info": work_items * 8 * torch.int32.itemsize,
        "workspace.partial_output": partial_slots
        * num_heads
        * 512
        * torch.float32.itemsize,
        "workspace.partial_lse": partial_slots * num_heads * torch.float32.itemsize,
        "workspace.finalize_count": torch.int32.itemsize,
        "workspace.finalize_jobs": max_rows * 8 * 2 * torch.int32.itemsize,
    }
    for name, nbytes in spans.items():
        _require_i32_byte_span(name, nbytes)
    return MlaDecodeVarlenWorkspace(
        max_rows=max_rows,
        num_heads=num_heads,
        num_kv_splits=requested_splits,
        stage_splits=stage_splits,
        work_indptr=torch.empty(work_items * 2, dtype=torch.int32, device=q.device),
        work_info=torch.empty((work_items, 8), dtype=torch.int32, device=q.device),
        partial_output=torch.empty(
            (partial_slots, num_heads, 512), dtype=torch.float32, device=q.device
        ),
        partial_lse=torch.empty(
            (partial_slots, num_heads), dtype=torch.float32, device=q.device
        ),
        row_splits=torch.empty(max_rows, dtype=torch.int32, device=q.device),
        finalize_count=torch.empty(1, dtype=torch.int32, device=q.device),
        finalize_jobs=torch.empty(
            (max_rows * 8, 2), dtype=torch.int32, device=q.device
        ),
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
    if q.shape[1] not in (8, 16, 32, 64, 128):
        raise ValueError("`q` must have 8, 16, 32, 64, or 128 heads")
    if kv_buffer.ndim not in (3, 4) or kv_buffer.shape[-1] != 576:
        raise ValueError("`kv_buffer` must have packed width 576")
    if kv_buffer.numel() != kv_buffer.shape[0] * 576:
        raise ValueError("`kv_buffer` must have page size 1 and one KV head")
    if tuple(o.shape) != (q.shape[0], q.shape[1], 512):
        raise ValueError("`o` must match [max_rows, num_heads, 512]")
    tensors = {
        "q": q,
        "kv_buffer": kv_buffer,
        "o": o,
        "query_start_loc": query_start_loc,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor")
        if tensor.device != q.device or tensor.device.type != "cuda":
            raise ValueError(f"`{name}` must be on {q.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"`{name}` must be contiguous")
    if query_start_loc.dtype != torch.int32 or query_start_loc.ndim != 1:
        raise TypeError("`query_start_loc` must be one-dimensional int32")
    if query_start_loc.numel() < 2:
        raise ValueError("`query_start_loc` must contain a start and endpoint")
    if kv_indptr.dtype != torch.int32 or kv_indptr.shape != (q.shape[0] + 1,):
        raise ValueError("`kv_indptr` must be int32 with max_rows + 1 entries")
    if kv_indices.dtype != torch.int32 or kv_indices.ndim != 1:
        raise TypeError("`kv_indices` must be one-dimensional int32")
    if q_scale is None or q_scale.dtype != torch.float32 or q_scale.numel() != 1:
        raise ValueError("`q_scale` must be a scalar float32 tensor")
    if kv_scale is None or kv_scale.dtype != torch.float32 or kv_scale.numel() != 1:
        raise ValueError("`kv_scale` must be a scalar float32 tensor")
    if num_kv_splits is not None and num_kv_splits < 1:
        raise ValueError("`num_kv_splits` must be positive")
    if workspace.max_rows != q.shape[0] or workspace.num_heads != q.shape[1]:
        raise ValueError("workspace shape does not match `q`")
    if workspace.num_kv_splits != num_kv_splits:
        raise ValueError("workspace split count does not match `num_kv_splits`")
    expected_stage_splits = 16 if num_kv_splits is None else num_kv_splits
    if workspace.stage_splits != expected_stage_splits:
        raise ValueError(
            "workspace stage split count does not match the split capacity"
        )
    total = q.shape[0] * workspace.stage_splits
    head_ctas = 2 if q.shape[1] == 128 else 1
    work_items = total * head_ctas
    partial_slots = 0 if workspace.stage_splits == 1 else total
    expected_workspace = {
        "work_indptr": ((work_items * 2,), torch.int32),
        "work_info": ((work_items, 8), torch.int32),
        "partial_output": ((partial_slots, q.shape[1], 512), torch.float32),
        "partial_lse": ((partial_slots, q.shape[1]), torch.float32),
        "row_splits": ((q.shape[0],), torch.int32),
        "finalize_count": ((1,), torch.int32),
        "finalize_jobs": ((q.shape[0] * 8, 2), torch.int32),
    }
    for name, (shape, dtype) in expected_workspace.items():
        tensor = getattr(workspace, name)
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"workspace `{name}` has an invalid shape or dtype")
        if tensor.device != q.device or not tensor.is_contiguous():
            raise ValueError(f"workspace `{name}` must be contiguous on {q.device}")
        _require_i32_byte_span(f"workspace.{name}", tensor.nbytes)
    for name in ("q", "kv_buffer", "o"):
        _require_i32_byte_span(name, tensors[name].nbytes)
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise ValueError(f"FlyDSL adaptive MLA currently supports gfx950, got {arch}")


def prepare_mla_decode_varlen(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    workspace: MlaDecodeVarlenWorkspace | None = None,
    num_kv_splits: int | None = None,
) -> MlaDecodeVarlenPrepared:
    if num_kv_splits is None and workspace is not None:
        num_kv_splits = workspace.num_kv_splits
    num_kv_splits = _normalize_num_kv_splits(q.shape[1], num_kv_splits)
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
    num_requests = query_start_loc.numel() - 1
    stage_splits = workspace.stage_splits
    stream = torch.cuda.current_stream(q.device)
    if q.shape[1] == 8:
        launch_build_compact_varlen_work_h8(
            query_start_loc,
            kv_indptr,
            workspace.work_indptr,
            workspace.work_info.reshape(-1),
            workspace.row_splits,
            workspace.finalize_count,
            workspace.finalize_jobs.reshape(-1),
            num_requests,
            q.shape[0],
            stage_splits,
            0 if num_kv_splits is None else num_kv_splits,
            stream=stream,
        )
    else:
        head_ctas = 2 if q.shape[1] == 128 else 1
        launch_build_varlen_work_info(
            query_start_loc,
            kv_indptr,
            workspace.work_indptr,
            workspace.work_info,
            num_requests,
            q.shape[0],
            stage_splits,
            head_ctas,
            stream=stream,
        )
    return MlaDecodeVarlenPrepared(
        workspace=workspace,
        query_start_loc=query_start_loc,
        kv_indptr=kv_indptr,
        num_requests=num_requests,
        num_kv_splits=num_kv_splits,
    )


def execute_mla_decode_varlen_prepared(
    prepared: MlaDecodeVarlenPrepared,
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    sm_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
):
    workspace = prepared.workspace
    num_kv_splits = prepared.num_kv_splits
    _validate_inputs(
        q,
        kv_buffer,
        o,
        prepared.query_start_loc,
        prepared.kv_indptr,
        kv_indices,
        q_scale,
        kv_scale,
        workspace,
        num_kv_splits,
    )
    if sm_scale is None:
        sm_scale = 1.0 / (576**0.5)
    stage_splits = workspace.stage_splits
    stream = torch.cuda.current_stream(q.device)
    if q.shape[1] == 8:
        num_workers = torch.cuda.get_device_properties(q.device).multi_processor_count
        launch_mla_decode_varlen_h8(
            q.reshape(-1, 576),
            kv_buffer.reshape(-1, 576),
            kv_indices,
            workspace.work_indptr,
            workspace.work_info.reshape(-1),
            o.reshape(-1, 512),
            workspace.partial_output.reshape(-1, 512),
            workspace.partial_lse.reshape(-1),
            q_scale,
            kv_scale,
            float(sm_scale),
            0 if stage_splits == 1 else 2,
            num_workers * H8_OCCUPANCY,
            stream=stream,
        )
        if stage_splits == 1:
            launch_zero_varlen_empty_rows(
                prepared.query_start_loc,
                prepared.kv_indptr,
                o,
                prepared.num_requests,
                q.shape[1],
                q.shape[0],
                stream=stream,
            )
        else:
            launch_finalize_varlen_h8(
                workspace.finalize_count,
                workspace.finalize_jobs.reshape(-1),
                workspace.row_splits,
                workspace.partial_output,
                workspace.partial_lse,
                kv_scale,
                o,
                stage_splits,
                num_workers,
                stream=stream,
            )
    else:
        head_ctas = 2 if q.shape[1] == 128 else 1
        total = q.shape[0] * stage_splits * head_ctas
        launch_mla_fwd_decode_m16x8_fp8_fp8(
            q.reshape(-1, 576),
            kv_buffer.reshape(-1, 576),
            kv_indices,
            workspace.work_indptr,
            workspace.work_info.reshape(-1),
            o.reshape(-1, 512),
            workspace.partial_output.reshape(-1, 512),
            workspace.partial_lse.reshape(-1),
            q_scale,
            kv_scale,
            float(sm_scale),
            q.shape[1],
            stage_splits == 1,
            total,
            163840 // OCCUPANCY,
            stream=stream,
        )
        if stage_splits == 1:
            launch_zero_varlen_empty_rows(
                prepared.query_start_loc,
                prepared.kv_indptr,
                o,
                prepared.num_requests,
                q.shape[1],
                q.shape[0],
                stream=stream,
            )
        else:
            launch_merge_varlen_partials(
                prepared.query_start_loc,
                prepared.kv_indptr,
                workspace.partial_output,
                workspace.partial_lse,
                kv_scale,
                o,
                prepared.num_requests,
                q.shape[1],
                q.shape[0],
                stage_splits,
                stream=stream,
            )
    return o


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
    prepared = prepare_mla_decode_varlen(
        q,
        kv_buffer,
        o,
        query_start_loc,
        kv_indptr,
        kv_indices,
        q_scale=q_scale,
        kv_scale=kv_scale,
        workspace=workspace,
        num_kv_splits=num_kv_splits,
    )
    return execute_mla_decode_varlen_prepared(
        prepared,
        q,
        kv_buffer,
        o,
        kv_indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )


def mla_decode_varlen_scratch_bytes(workspace: MlaDecodeVarlenWorkspace) -> int:
    tensors = (
        workspace.work_indptr,
        workspace.work_info,
        workspace.partial_output,
        workspace.partial_lse,
        workspace.row_splits,
        workspace.finalize_count,
        workspace.finalize_jobs,
    )
    return sum(tensor.nbytes for tensor in tensors)


__all__ = [
    "MlaDecodeVarlenPrepared",
    "MlaDecodeVarlenWorkspace",
    "create_mla_decode_varlen_workspace",
    "execute_mla_decode_varlen_prepared",
    "mla_decode_varlen",
    "mla_decode_varlen_scratch_bytes",
    "prepare_mla_decode_varlen",
]
