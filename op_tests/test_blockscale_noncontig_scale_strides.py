#!/usr/bin/env python3
"""Regression test for blockscale GEMM stride handling.

Verifies that CK and CKTile honor physical tensor strides for:
  1. vLLM-style padded weights,
  2. non-contiguous block-scale weight tensors,
  3. the combination of both.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck, gemm_a8w8_blockscale_cktile

BLOCK = 128


def make_fp8(shape, scale, seed, device="cuda"):
    torch.manual_seed(seed)
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtypes.fp8)


def make_scales(shape, seed, device="cuda"):
    torch.manual_seed(seed)
    return torch.rand(shape, dtype=torch.float32, device=device) + 0.5


def pad_tensor_like_vllm(t):
    num_pad = 256 // t.element_size()
    return F.pad(t, (0, num_pad), "constant", 0)[..., :-num_pad]


def make_noncontiguous_w_scale(scale_n, scale_k, seed, device="cuda"):
    torch.manual_seed(seed)
    wide = torch.rand((scale_n, scale_k + 2), dtype=torch.float32, device=device) + 0.5
    sliced = wide[:, :scale_k]
    assert sliced.stride(-1) == 1
    assert not sliced.is_contiguous()
    return sliced


def torch_reference(x_fp8, w_fp8, x_scale, w_scale):
    m, k = x_fp8.shape
    n = w_fp8.shape[0]
    scale_n = math.ceil(n / BLOCK)
    scale_k = math.ceil(k / BLOCK)

    x_f = x_fp8.float().view(m, scale_k, BLOCK) * x_scale.unsqueeze(-1)
    x_f = x_f.view(m, k)

    w_f = w_fp8.float().view(scale_n, BLOCK, scale_k, BLOCK)
    w_f = w_f * w_scale[:, None, :, None]
    w_f = w_f.view(n, k)

    return (x_f @ w_f.t()).to(torch.bfloat16)


def run_kernel(kind, x_fp8, w_fp8, x_scale, w_scale):
    m, n = x_fp8.shape[0], w_fp8.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x_fp8.device)
    if kind == "ck":
        gemm_a8w8_blockscale_ck(x_fp8, w_fp8, x_scale, w_scale, out)
    elif kind == "cktile":
        gemm_a8w8_blockscale_cktile(x_fp8, w_fp8, x_scale, w_scale, out)
    else:
        raise ValueError(f"unknown kernel kind: {kind}")
    return out


def error_ratio(ref, out):
    ref_f = ref.float()
    out_f = out.float()
    return 1.0 - torch.isclose(ref_f, out_f, rtol=1e-3, atol=1e-3).float().mean().item()


def cosine_sim(ref, out):
    return F.cosine_similarity(ref.float().flatten().unsqueeze(0),
                               out.float().flatten().unsqueeze(0)).item()


def assert_close(label, ref, out):
    cos = cosine_sim(ref, out)
    err = error_ratio(ref, out)
    max_abs = (out.float() - ref.float()).abs().max().item()
    print(f"{label:28s} cos={cos:.8f} err%={err * 100:.2f} max_abs={max_abs:.4e}")
    assert cos > 0.9999, f"{label}: cosine similarity too low ({cos:.8f})"
    assert err < 1e-4, f"{label}: error ratio too high ({err:.6f})"


def main():
    device = "cuda"
    m, n, k = 4, 1024, 4096
    scale_n = math.ceil(n / BLOCK)
    scale_k = math.ceil(k / BLOCK)

    x_fp8 = make_fp8((m, k), scale=0.1, seed=1, device=device)
    w_fp8 = make_fp8((n, k), scale=0.02, seed=2, device=device)
    x_scale = make_scales((m, scale_k), seed=3, device=device)
    w_scale_contig = make_scales((scale_n, scale_k), seed=4, device=device)
    w_scale_nc = make_noncontiguous_w_scale(scale_n, scale_k, seed=5, device=device)
    w_fp8_padded = pad_tensor_like_vllm(w_fp8)

    ref = torch_reference(x_fp8, w_fp8, x_scale, w_scale_contig)

    cases = [
        ("padded_w", w_fp8_padded, w_scale_contig, ref),
        ("noncontig_ws", w_fp8, w_scale_nc, torch_reference(x_fp8, w_fp8, x_scale, w_scale_nc)),
        ("padded_w_noncontig_ws", w_fp8_padded, w_scale_nc,
         torch_reference(x_fp8, w_fp8, x_scale, w_scale_nc)),
    ]

    print(f"WQ padded stride: {w_fp8_padded.stride()}")
    print(f"w_scale contig stride: {w_scale_contig.stride()}")
    print(f"w_scale noncontig stride: {w_scale_nc.stride()}")

    for kind in ("ck", "cktile"):
        for case_name, weight, weight_scale, case_ref in cases:
            out = run_kernel(kind, x_fp8, weight, x_scale, weight_scale)
            assert_close(f"{kind}:{case_name}", case_ref, out)

    print("All stride regression checks passed.")


if __name__ == "__main__":
    main()
