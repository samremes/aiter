# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flat per-row workspace layout shared by Stage A restore and persistent B."""

from .candidate_topk_common import NUM_HIST_BINS
from .topk_per_row_decode import _STATE_SIZE

COUNTER_GROUP = 32
COUNTER_ARRIVALS = 0
COUNTER_PASS_DONE = 1
COUNTER_OUT_ABOVE = 2
COUNTER_OUT_EQUAL = 3
COUNTER_STATUS = 4

HIST0_OFF = COUNTER_GROUP
HIST1_OFF = HIST0_OFF + NUM_HIST_BINS
HIST2_OFF = HIST1_OFF + NUM_HIST_BINS
STATE_GROUP = 32
STATE_OFF = HIST2_OFF + NUM_HIST_BINS
ROW_STRIDE = STATE_OFF + STATE_GROUP
STATE_SIZE = _STATE_SIZE


def persistent_workspace_elems(rows: int) -> int:
    return int(rows) * ROW_STRIDE


def persistent_hist0_view(flat, rows: int):
    return flat.as_strided(
        (rows, 1, NUM_HIST_BINS),
        (ROW_STRIDE, NUM_HIST_BINS, 1),
        storage_offset=HIST0_OFF,
    )


def persistent_state_view(flat, rows: int):
    return flat.as_strided(
        (rows, STATE_SIZE),
        (ROW_STRIDE, 1),
        storage_offset=STATE_OFF,
    )
