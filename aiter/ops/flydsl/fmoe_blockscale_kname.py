# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tuned-config kernel names for the FlyDSL fused blockscale-fp8 MoE.

``flydsl_fmoe_blockscale[_r{rows}][_c{cols}][_s{slices}][_ws{ws}]`` in
``kernelName1`` of a ``run_1stage`` row; a missing field takes the wrapper's
default (``fmoe_config``). Kept free of FlyDSL imports so dispatch can parse
names without loading the compiler.
"""

import re

FMOE_BLOCKSCALE_PREFIX = "flydsl_fmoe_blockscale"
_FIELDS = (("r", "rows"), ("c", "cols"), ("s", "slices"), ("ws", "ws"))
_KNAME_RE = re.compile(
    rf"^{FMOE_BLOCKSCALE_PREFIX}"
    + "".join(rf"(?:_{tag}(?P<{key}>\d+))?" for tag, key in _FIELDS)
    + "$"
)


def is_fmoe_blockscale_kname(name) -> bool:
    """Whether ``name`` claims the FlyDSL blockscale kernel (well-formed or not)."""
    return isinstance(name, str) and (
        name == FMOE_BLOCKSCALE_PREFIX or name.startswith(FMOE_BLOCKSCALE_PREFIX + "_")
    )


def parse_fmoe_blockscale_kname(name: str) -> dict[str, int | None]:
    """``{rows, cols, slices, ws}`` from a kernel name; ``None`` means default."""
    m = _KNAME_RE.match(name)
    if m is None:
        raise ValueError(
            f"invalid FlyDSL blockscale kernel name {name!r}: expected "
            f"{FMOE_BLOCKSCALE_PREFIX}[_r<rows>][_c<cols>][_s<slices>][_ws<ws>]"
        )
    return {k: None if v is None else int(v) for k, v in m.groupdict().items()}
