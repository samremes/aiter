# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared canonical score primitive for native-FP8 paged MQA kernels."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl

HEADS = 32
MFMA_M = 16
DREG = 4
M_TILES = HEADS // MFMA_M
WAVE_SIZE = 64


def canonical_score_tile(
    q_tiles,
    weight_frags,
    k_pack,
    scale,
    page_ok,
    result_type,
    neutral,
):
    """Score one 16-position MFMA tile in the frozen P1 operation order."""
    total = fx.Float32(0.0)
    for mi in range_constexpr(M_TILES):
        acc = fx.Vector.filled(DREG, 0.0, fx.Float32)
        acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            result_type,
            [
                q_tiles[mi],
                k_pack,
                acc,
                0,
                0,
                0,
                neutral,
                0,
                neutral,
            ],
        )
        frag = fx.Vector(acc)
        for ii in range_constexpr(DREG):
            activated = fx.Float32(frag[ii]).maximumf(fx.Float32(0.0))
            total = total + activated * weight_frags[mi][ii]
    total = total + total.shuffle_xor(16, WAVE_SIZE)
    total = total + total.shuffle_xor(32, WAVE_SIZE)
    total = total * scale.maximumf(fx.Float32(0.0))
    total = page_ok.select(total, fx.Float32(float("-inf")))
    is_nan = total != total  # noqa: PLR0124 - GPU SSA NaN canonicalization.
    return is_nan.select(fx.Float32(float("-inf")), total)
