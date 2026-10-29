# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL fused blockscale-fp8 g1u1 silu MoE vs the shipped asm kernel and the torch reference."""

import argparse

import torch
from einops import rearrange

import aiter
from aiter import dtypes, pertoken_quant
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")

BLK = 128


def _block_quant(w):
    e, n, k = w.shape
    tmp = rearrange(
        w.view(e, n // BLK, BLK, k // BLK, BLK),
        "e nb bn kb bk -> e nb kb (bn bk)",
    ).contiguous()
    q, s = pertoken_quant(tmp, quant_dtype=dtypes.fp8)
    q = rearrange(
        q.view(e, n // BLK, k // BLK, BLK, BLK),
        "e nb kb bn bk -> e (nb bn) (kb bk)",
    ).contiguous()
    return q, s.view(e, n // BLK, k // BLK)


def make_case(token, E, topk, model_dim, inter_dim, seed=0):
    torch.manual_seed(seed)
    dtype = dtypes.bf16
    x = torch.randn((token, model_dim), dtype=dtype)
    w1 = torch.randn((E, 2 * inter_dim, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)
    w1_q, w1_s = _block_quant(w1)
    w2_q, w2_s = _block_quant(w2)
    a_q, a_s = pertoken_quant(x.view(token, -1, BLK), quant_dtype=dtypes.fp8)
    return {
        "a_q": a_q.view(token, model_dim),
        "a_s": a_s.squeeze(-1),  # [T, D/128]
        "w1_q": w1_q,
        "w1_s": w1_s,
        "w2_q": w2_q,
        "w2_s": w2_s,
        "w1_sh": shuffle_weight(w1_q, (16, 16)),
        "w2_sh": shuffle_weight(w2_q, (16, 16)),
        "xq": a_s.squeeze(-1).t().contiguous(),  # [D/128, T]
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "E": E,
        "topk": topk,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
    }


def torch_ref(c):
    """f32 reference: dequantized X, W1, W2 (the same math as test_moe_blockscale.torch_moe_blockscale)."""
    T_, D = c["a_q"].shape
    E, I = c["E"], c["inter_dim"]
    x = (c["a_q"].float().view(T_, -1, BLK) * c["a_s"].unsqueeze(-1)).view(T_, D)
    s1 = c["w1_s"].repeat_interleave(BLK, 1).repeat_interleave(BLK, 2)
    s2 = c["w2_s"].repeat_interleave(BLK, 1).repeat_interleave(BLK, 2)
    out = torch.zeros((T_, D), dtype=torch.float32)
    for e in range(E):
        tok, slot = torch.nonzero(c["topk_ids"] == e, as_tuple=True)
        if tok.numel() == 0:
            continue
        w1 = c["w1_q"][e].float() * s1[e]
        w2 = c["w2_q"][e].float() * s2[e]
        gu = x[tok] @ w1.t()
        h = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        out.index_add_(
            0, tok, (h @ w2.t()) * c["topk_weights"][tok, slot].unsqueeze(-1)
        )
    return out.to(dtypes.bf16)


def sort(c, block_m=64):
    return moe_sorting(
        c["topk_ids"], c["topk_weights"], c["E"], c["model_dim"], dtypes.bf16, block_m
    )


def run_asm(c, srt):
    sti, sw, sei, nvi, out = srt
    out.zero_()
    aiter.fmoe_fp8_blockscale_g1u1(
        out,
        c["a_q"],
        c["w1_sh"],
        c["w2_sh"],
        sti,
        sw,
        sei,
        nvi,
        c["topk"],
        c["xq"],
        c["w1_s"].view(c["E"], -1),
        c["w2_s"].view(c["E"], -1),
        "",
        BLK,
        BLK,
        None,
        block_size_M=64,
    )
    return out


def run_flydsl(c, srt, tile_n=None, slices=None):
    from aiter.ops.flydsl.fmoe_blockscale import fmoe_blockscale_g1u1_flydsl

    sti, sw, sei, nvi, out = srt
    out.zero_()
    fmoe_blockscale_g1u1_flydsl(
        out,
        c["a_q"],
        c["w1_sh"],
        c["w2_sh"],
        sti,
        sw,
        sei,
        nvi,
        c["xq"],
        c["w1_s"],
        c["w2_s"],
        tile_n=tile_n,
        slices=slices,
    )
    return out


def report(name, out, ref):
    """rel-RMS and max error in bf16 ULPs of the reference absmax (the Quilt gate passes at <= 8)."""
    out, ref = out.float(), ref.float()
    ulp = 2.0 ** (torch.floor(torch.log2(ref.abs().max())).item() - 7)
    err = (out - ref).abs().max().item()
    rel = ((out - ref).norm() / ref.norm()).item()
    ok = err <= 8 * ulp and torch.isfinite(out).all().item()
    print(
        f"[check] {name}: rel-RMS {rel:.4e}, max err {err:.4g} = {err / ulp:.2f} ULPs {'ok' if ok else 'FAILED'}"
    )
    return ok


def time_pair(fns, rounds=21, iters=50, warmup=5):
    """Paired alternating rounds with CUDA events; returns per-fn list of per-launch us."""
    for f in fns:
        for _ in range(warmup):
            f()
    torch.cuda.synchronize()
    res = [[] for _ in fns]
    for r in range(rounds):
        order = list(range(len(fns)))
        if r % 2:
            order.reverse()
        for i in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            s.record()
            for _ in range(iters):
                fns[i]()
            e.record()
            e.synchronize()
            res[i].append(s.elapsed_time(e) * 1000 / iters)
    return res


ROWS = {
    "small": (128, 8, 2, 1024, 256),
    "tuned_7168": (16384, 257, 9, 7168, 256),
    "tuned_2048": (16384, 513, 11, 2048, 512),
    "wide": (4096, 256, 8, 2048, 512),
    "prefill": (16384, 256, 8, 2048, 512),
    "tuned_3072": (16384, 256, 8, 3072, 768),
    "wide_384": (4096, 256, 8, 2048, 768),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="*", default=["small"])
    p.add_argument("--shape", type=int, nargs=5, help="T E topk model_dim inter_dim")
    p.add_argument("--mode", choices=["check", "bench", "asm"], default="check")
    p.add_argument(
        "--tile-n",
        type=int,
        choices=[256, 384],
        help="column slice width (default: the wrapper's pick)",
    )
    p.add_argument("--rounds", type=int, default=21)
    p.add_argument(
        "--slices",
        type=int,
        help="column slices per tile (default: the wrapper's pick)",
    )
    a = p.parse_args()
    rows = {"custom": tuple(a.shape)} if a.shape else {r: ROWS[r] for r in a.rows}
    for name, shape in rows.items():
        c = make_case(*shape)
        srt = sort(c)
        ref = torch_ref(c) if shape[0] * shape[2] <= 4096 * 8 else None
        o_asm = run_asm(c, srt).clone()
        if ref is not None:
            report(f"{name} asm vs ref", o_asm, ref)
        if a.mode == "asm":
            continue
        o_fly = run_flydsl(c, srt, a.tile_n, a.slices).clone()
        if ref is not None:
            report(f"{name} flydsl vs ref", o_fly, ref)
        report(f"{name} flydsl vs asm", o_fly, o_asm)
        if a.mode == "bench":
            t = time_pair(
                [
                    lambda c=c, srt=srt: run_asm(c, srt),
                    lambda c=c, srt=srt: run_flydsl(c, srt, a.tile_n, a.slices),
                ],
                rounds=a.rounds,
            )
            ta, tf = (sorted(x)[len(x) // 2] for x in t)
            ratios = sorted(f / s for f, s in zip(t[1], t[0]))
            wins = sum(r < 1 for r in ratios)
            print(
                f"[bench] {name} {shape}: asm {ta:.1f} us, flydsl {tf:.1f} us, "
                f"paired median {100 * (ratios[len(ratios) // 2] - 1):+.2f}% (flydsl faster {wins}/{len(ratios)})"
            )


if __name__ == "__main__":
    main()
