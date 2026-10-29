# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  (the per-tile closures are traced within their iteration)

"""Fused blockscale-fp8 g1u1 silu MoE on gfx950, one persistent launch.

The FlyDSL counterpart of the asm ``fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps``
kernels: gate/up GEMM, silu(gate) * up, a per-token 1x128 fp8 requant, the down
GEMM and the routed bf16 merge into the output, per sorted block of ``rows``
(64, or 128 at experimental 128- or 256-column slices) and ``slices * cols``
intermediate columns. 128 rows halve the weight bytes per MFMA.

===========  ========================  =====================================
operand      shape                     notes
===========  ========================  =====================================
``X``        ``[T, D]`` fp8            K-contiguous tokens
``XQ``       ``[D // 128, T]`` f32     1x128 token scales, column-major
``GU``       ``[E, 2I, D]`` fp8        gate rows, then up; shuffle_weight 16x16
``GQ``       ``[E, 2I/128, D/128]``    128x128 block scales, f32
``DW``       ``[E, D, I]`` fp8         shuffle_weight 16x16
``DQ``       ``[E, D/128, I/128]``     128x128 block scales, f32
``O``        ``[T, D]`` bf16           pre-zeroed; routed rows add atomically
===========  ========================  =====================================

Sorting is aiter ``moe_sorting`` with block ``rows``: ``sorted_ids`` holds
``token | slot << 24`` (padding rows hold token T), ``sorted_expert_ids`` one
expert per row block and ``num_valid_ids[0]`` the sorted row count.

Work split. A tile is (row block, ``slices`` column slices of ``cols``, 256 or
384): gate/up runs slice by slice, each leaving its quantized h in its own LDS
panel, then one down pass contracts over all of them (K = ``slices * cols``),
so the output atomics happen once a tile. One workgroup per CU walks the tiles
persistently, each XCD a contiguous run. Four waves split the columns: wave w
owns the 16-column MFMA tiles ``64 j + 16 w``, all the tile's tokens.
MFMA 16x16x128 takes the weights as A and the tokens as B, so C lane l holds
token ``l % 16`` and the four weight rows ``4 (l // 16) + r``.

Token panel (LDS). One K step is 256 K: two 128-K halves (kh) of ``rows / 16``
16-token groups, a 128-byte row per token, ``rows / 4`` KiB, double buffered.
Wave w gathers groups ``w + 4 q``. Its 16-byte chunks
carry the Swizzle(3, 4, 4) of the dense row-major panel (chunk ^= (row >> 1) & 7),
so the 16 token rows a B read spans hit distinct banks. The gather is LDS DMA:
lane l lands 16 bytes at ``base + 16 l`` and fetches the logical chunk that the
swizzle maps there. The quantized h reuses the same panel layout (staging) as
the down GEMM's B operand.

Down. A turn is 256 output columns at 64 rows, 128 at 128 (two relay slots
fit): wave w owns ``64 t + 16 w``. The bf16 results go through an LDS relay,
then each wave adds its own ``rows / 4`` token rows into O with packed bf16
buffer atomics; padding rows land past O's records and drop.
Turn t's merge interleaves with turn t + 1's MFMAs.

Down admission. Down is bound by the device-wide atomic throughput, and with
equal tiles all CUs reach it in lockstep. A global counter (zero between
launches) caps the workgroups in down at once; the rest wait, at most
``ADMIT_WAIT``, which offsets the CUs' phases: about 10% on D = 7168.

Stored partials (``stored``). Rows of routing slots below ``stored`` go to a
bf16 ``P[T, stored, D]`` with plain stores (16 bytes a lane where a whole
store's rows are stored slots) instead of atomics into O; a reduce pass sums
them into O. Stores take HBM bandwidth instead of the L2 atomic units.

Ping-pong (``ws`` 2, 64 x 256 tiles). Eight waves at 256 VGPRs: waves 0-3
run gate/up and the requant of tile r into h buffer r % 2 while waves 4-7
run down and the output of tile r - 1, so the output spreads over the whole
tile instead of a burst at its end. Both roles pass the same unmergeable
barriers a round: gate/up a step (two at one slice: a down turn is 128
columns), one a slice for the amax exchange and the h handoff; down one a
turn, one after each slice's block of turns and the handoff. The relay sits
above the token panels; the h panels and down scales are double buffered.

Scheduling. Loads issue between MFMAs (``mfma_stream`` side thunks), so the
in-order memory queue throttles loads and not the MFMA pipe. LDS addresses
past 64 KiB come from opaque per-lane bases (``opaque``) so constant parts
fold into the ds offset. Each gate/up fold goes through ``settle``: otherwise
the folds of a slice's last step sink to the silu, and the products they
consume spill (384 columns: ~400 VGPRs).

Step wait. The token scales of a step ride in LDS with its token panel (one
more DMA), so the only loads issued after a step's DMA are the next gate
weights, and the step waits with an exact ``vmcnt(4 n_j)``. LLVM does not
track the DMA: ``wait_asyncmark`` lowers to ``vmcnt(0)``, which also drained
the gate weights every step (9% on D = 2048 prefill).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Constexpr, Float32, Int8, Int32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import LOG2E, format_kernel_name

WAVE = 64
THREADS = 256
ROW_TILES = (64, 128)  # sorted rows per tile (the moe_sorting block)
BLK = 128  # scale block edge
STEP_K = 256  # K per gate/up step: two 128-K halves
MFMA_LAG = 3  # products in flight before the first fold reads one
FP8_MAX = 448.0
ADMIT_WAIT = 5000  # longest down-admission wait, 100 MHz ticks (50 us)
LDS_MAX = 160 * 1024
# Profiling knockouts (wrong results, tagged in the kernel name):
# FMOE_KO=atomic drops the whole output (atomics and partial stores).
KO = set(filter(None, os.environ.get("FMOE_KO", "").split(",")))
# Slice widths by tile rows: 64 rows take 256 or 384 columns; 128 rows take
# 128, so the accumulators stay at 128 registers.
COLS = {64: (256, 384), 128: (128, 256)}
MAX_SLICES = {64: 3, 128: 6}


class Geometry:
    """The tile's LDS map, bytes. The down relay overlays the gate/up token
    panels, which are dead by then; at offset 0 every relay access folds its
    constant part into the 16-bit ds offset. The h panels (the down B operand)
    sit above it. Under ``ws`` 2 the down waves run beside gate/up, so the
    relay sits above the panels and the h panels and down scales are double
    buffered."""

    def __init__(self, rows, cols, slices, ws=0):
        self.rows = rows
        self.pp = ws == 2
        # Down output columns per turn: two relay slots of rows x turn_n bf16
        # must fit beside the h panels.
        self.turn_n = 256 if rows == 64 and not self.pp else 128
        # One token panel: (kh 2, group rows / 16, token 16, 128 B).
        self.panel = rows * STEP_K
        self.relay_pitch = self.turn_n + 8  # bf16 per relay row; the pad staggers banks
        self.relay = rows * self.relay_pitch * 2
        # One 128-K block of an h panel: (group rows / 16, token 16, 128 B).
        self.h_block = rows * BLK
        # The requant's amax scratch, f32 [wave 4][block][token], also overlays
        # the relay, past the token panels the next slice's prologue fills.
        self.off_amax = 2 * self.panel
        # The token scales of a step, f32 [buffer 2][kh 2][token], DMA'd with
        # the token panels; past the amax scratch, which is live during the
        # next prologue.
        self.off_xs = self.off_amax + 4 * (cols // BLK) * rows * 4
        end = self.off_xs + 2 * 2 * rows * 4
        self.off_relay = end if self.pp else 0
        relay_end = self.off_relay + 2 * self.relay
        self.off_stage = -(-max(end, relay_end) // 1024) * 1024
        self.nh = 2 if self.pp else 1  # h buffers
        self.h_bytes = slices * (cols // BLK) * self.h_block  # one h buffer
        # The down scales (requant scale x routing weight), f32
        # [slice][block][token], parked in LDS: at 128 rows those of all but
        # the last slice, through the later slices' gate/up, whose
        # accumulators leave no registers for them; under ws 2 all of them,
        # for the down waves.
        self.n_park = slices if self.pp else slices - 1 if rows > 64 else 0
        self.park = self.n_park > 0
        self.act_bytes = self.n_park * (cols // BLK) * rows * 4
        self.off_act = self.off_stage + self.nh * self.h_bytes
        self.lds = self.off_act + self.nh * self.act_bytes


def lds_bytes(rows, cols, slices, ws=0):
    return Geometry(rows, cols, slices, ws).lds


def pp_rates(slices, turn_n):
    """``ws`` 2: barriers a gate/up step and a down turn, so both roles pass
    the same count a tile (slices * n_steps * a == n_turns * b)."""
    k = slices * turn_n  # down turns per gate/up step, times STEP_K
    if k % STEP_K == 0:
        return 1, k // STEP_K
    assert STEP_K % k == 0
    return STEP_K // k, 1


def check_config(model_dim, inter_dim, cols, slices=1, rows=64, ws=0, stored=0):
    if rows not in ROW_TILES:
        raise ValueError(f"rows {rows}: the tile rows are one of {ROW_TILES}")
    if cols not in COLS[rows]:
        raise ValueError(f"cols {cols}: {rows}-row slices are one of {COLS[rows]}")
    if not 1 <= slices <= MAX_SLICES[rows] or inter_dim % (cols * slices):
        raise ValueError(
            f"inter_dim {inter_dim} does not split into {slices} slices of {cols} columns"
        )
    if ws not in (0, 2) or (ws and (rows, cols) != (64, 256)):
        raise ValueError(f"ws {ws}: 0, or 2 at 64 x 256 (256 VGPRs a wave)")
    if stored and slices * cols != inter_dim:
        raise ValueError("stored partials need one tile over all of inter_dim")
    turn_n = Geometry(rows, cols, slices, ws).turn_n
    if model_dim % (2 * turn_n * (slices if ws else 1)):
        raise ValueError(f"model_dim {model_dim} does not split into the down turns")
    if lds_bytes(rows, cols, slices, ws) > LDS_MAX:
        raise ValueError(f"{rows}x{cols}x{slices}: the LDS map exceeds {LDS_MAX}")


@flyc.jit
def launch_fmoe_blockscale(
    arg_o: fx.Pointer,
    arg_x: fx.Pointer,
    arg_gu: fx.Pointer,
    arg_dw: fx.Pointer,
    arg_ids: fx.Pointer,
    arg_rw: fx.Pointer,
    arg_eids: fx.Pointer,
    arg_nvalid: fx.Pointer,
    arg_xq: fx.Pointer,
    arg_gq: fx.Pointer,
    arg_dq: fx.Pointer,
    arg_sem: fx.Pointer,
    arg_p: fx.Pointer,
    i32_tokens: fx.Int32,
    i32_experts: fx.Int32,
    stream: fx.Stream,
    D: Constexpr[int],
    I: Constexpr[int],
    cols: Constexpr[int],
    slices: Constexpr[int],
    grid: Constexpr[int],
    rows: Constexpr[int] = 64,
    ws: Constexpr[int] = 0,
    stored: Constexpr[int] = 0,
    admit: Constexpr[int] = 0,
):
    check_config(D, I, cols, slices, rows, ws, stored)
    geo = Geometry(rows, cols, slices, ws)
    pp = geo.pp  # ping-pong: gate/up waves 0-3, down + output waves 4-7
    threads = THREADS * (2 if pp else 1)
    ROWS = rows
    TURN_N = geo.turn_n
    PANEL = geo.panel
    RELAY = geo.relay
    RELAY_PITCH = geo.relay_pitch
    H_BLOCK = geo.h_block
    OFF_AMAX = geo.off_amax
    OFF_XS = geo.off_xs
    OFF_STAGE = geo.off_stage
    OFF_ACT = geo.off_act
    OFF_RELAY = geo.off_relay
    n_park = geo.n_park
    ARENA = geo.lds
    ps_deno = I // (cols * slices)  # column tiles per row block
    assert grid % 16 == 0 and grid % ps_deno == 0
    # Workgroups let into down at once. Down is bound by the device's atomic
    # throughput and moves the same output bytes per tile whatever its width,
    # so narrower tiles (less MFMA per tile) admit fewer: (2 + n) / 8 of the
    # grid for n * 256 columns.
    # (admit: the limit in grid / 16; 16 or ws 2: no admission.)
    down_limit = grid * admit // 16 if admit else grid * (512 + slices * cols) // 2048
    if pp:
        down_limit = grid  # down never waits on gate/up
    n_steps = D // STEP_K
    n_turns = D // TURN_N
    # ws 2: barriers a gate/up step (pp_a) and a down turn (pp_b); each role
    # also passes one a slice (gate/up's amax exchange) and the h handoff, so
    # both pass n_pp a round.
    pp_a, pp_b = pp_rates(slices, TURN_N) if pp else (1, 1)
    assert not pp or (pp_b == 1 and slices * n_steps * pp_a == n_turns)
    n_pp = slices * (n_steps * pp_a + 1) + 1
    n_j = cols // 64  # 16-column MFMA tiles per wave
    n_groups = ROWS // 16
    # Gate/up phases of a K step, (up, K halves), each with its weights in one
    # of two ring buffers while the next phase's load. 64 rows run a
    # projection over both halves; 128 rows (twice the MFMAs per weight byte)
    # one half at a time, so the ring holds half the bytes for as many MFMAs.
    # Under ws 2 (256 VGPRs a wave) 64 rows split the halves too.
    halves = [(0, 1)] if ROWS == 64 and not pp else [(0,), (1,)]
    phases = [(up, khs) for khs in halves for up in (False, True)]
    n_t = TURN_N // 64  # 16-column MFMA tiles per wave per down turn
    n_parts = slices * n_groups  # merge parts per turn
    rows_w = ROWS // 4  # relay rows each wave merges
    pieces = TURN_N // 128  # 128-column runs of a relay row
    n_dma = n_groups // 4  # token groups each wave gathers
    kd = D // BLK  # K blocks of the gate/up GEMM
    ki = I // BLK  # K blocks of the down GEMM
    n_blk = cols // BLK  # 128-column blocks of a slice: down K blocks, scale rows
    h_panel = n_blk * H_BLOCK  # bytes of one slice's h panel
    lpr = TURN_N // 8  # lanes a relay row in 16-byte pieces
    rpi = WAVE // lpr  # relay rows one 16-byte partial store covers
    kernel_name = format_kernel_name(
        f"fmoe_blockscale_g1u1_silu_{ROWS}x{cols}x{slices}_d{D}_i{I}"
        + (f"_p{stored}" if stored else "")
        + ("_pp" if pp else "")
        + (f"_a{admit}" if admit else "")
        + (f"_ko{'_'.join(sorted(KO))}" if KO else "")
    )

    @fx.struct
    class Arena:
        data: fx.Array[Int8, ARENA, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def kernel_fmoe(
        arg_o: fx.Int64,
        arg_x: fx.Int64,
        arg_gu: fx.Int64,
        arg_dw: fx.Int64,
        arg_ids: fx.Int64,
        arg_rw: fx.Int64,
        arg_eids: fx.Int64,
        arg_nvalid: fx.Int64,
        arg_xq: fx.Int64,
        arg_gq: fx.Int64,
        arg_dq: fx.Int64,
        arg_sem: fx.Int64,
        arg_p: fx.Int64,
        i32_tokens: fx.Int32,
        i32_experts: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave_g = rocdl.readfirstlane(T.i32, tid // WAVE)
        # The wave's index in its role (under ws 2 waves 4-7 run down).
        wave = wave_g % 4 if const_expr(pp) else wave_g
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16

        def _buffer(base, elem, align, nbytes):
            ptr_ty = fx.PointerType.get(
                elem, address_space=fx.AddressSpace.Global, alignment=align
            )
            view = fx.make_view(fx.inttoptr(ptr_ty, base), fx.make_layout(1 << 30, 1))
            buf = fx.rocdl.make_buffer_tensor(
                fx.Tensor(view), max_size=False, num_records_bytes=nbytes
            )
            return fx.logical_divide(buf, fx.make_layout(1, 1))

        tokens64 = fx.Int64(i32_tokens)
        x_src = _buffer(arg_x, T.i8, 16, tokens64 * D)  # the padding token reads 0
        xq_src = _buffer(arg_xq, T.f32, 4, tokens64 * (kd * 4))
        ids_src = _buffer(arg_ids, T.i32, 4, fx.Int64(1 << 30))
        rw_src = _buffer(arg_rw, T.f32, 4, fx.Int64(1 << 30))
        eids_src = _buffer(arg_eids, T.i32, 4, fx.Int64(1 << 30))
        nvalid_src = _buffer(arg_nvalid, T.i32, 4, fx.Int64(4))
        experts64 = fx.Int64(i32_experts)
        gq_src = _buffer(arg_gq, T.f32, 4, experts64 * (2 * I // BLK * kd * 4))
        dq_src = _buffer(arg_dq, T.f32, 4, experts64 * (kd * ki * 4))
        o_dst = _buffer(arg_o, T.bf16, 4, tokens64 * (D * 2))  # padding rows drop
        # Stored partials P[T, stored, D] (slots below ``stored``): the token
        # rides in voffset, so padding rows drop as in O.
        p_bytes = tokens64 * (max(stored, 1) * D * 2)
        p_dst = _buffer(arg_p, T.bf16, 4, p_bytes)
        p_i32 = _buffer(arg_p, T.i32, 16, p_bytes)

        def sload(src, idx):
            return rocdl.s_buffer_load(fx.get_iter(src), idx)

        def sload_f32(src, idx):
            return Int32(sload(src, idx)).bitcast(Float32)

        load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
        load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
        load_w = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        dma16 = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)
        dma4 = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS32b(), 32)
        read16 = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        write8 = fx.make_copy_atom(fx.UniversalCopy64b(), BFloat16)
        atomic_bf16x2 = fx.make_copy_atom(
            fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16
        )
        store_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), BFloat16)
        store_x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)

        def vload(atom, src, idx, elem):
            frag = fx.make_rmem_tensor(1, elem)
            fx.copy(atom, fx.slice(src, (None, idx)), frag)
            return frag.load()[0]

        lds = fx.SharedAllocator().allocate(Arena).peek()
        lds_i8 = fx.recast_iter(
            fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512), lds.data.ptr
        )
        lds_i32 = fx.recast_iter(Int32, lds.data.ptr)
        lds_f32 = fx.recast_iter(Float32, lds.data.ptr)
        lds_bf16 = fx.recast_iter(
            fx.PointerType.get(BFloat16.ir_type, fx.AddressSpace.Shared, 8),
            lds.data.ptr,
        )

        # Token panel: dense (token 16, byte 128) rows under Swizzle(3, 4, 4);
        # groups, halves and buffers stack above bit 10, which the swizzle
        # leaves alone, so only the in-group offset goes through the layout.
        panel_swz = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 4, 4)),
            fx.make_ordered_layout((16, 128), (1, 0)),
        )

        def panel_off(t, byte):
            return fx.get_scalar(fx.crd2idx((t, byte), panel_swz))

        def stack_off(kh, gi):
            return kh * H_BLOCK + gi * 2048

        # B reads: token lane16, 16-byte chunks kgrp and 4 + kgrp (the f8f6f4
        # operand pairs halves 64 K apart).
        b_lo = panel_off(lane16, kgrp * 16) // 4
        b_hi = panel_off(lane16, 64 + kgrp * 16) // 4
        # DMA: lane l lands on token 8 b + (l >> 3), physical chunk l & 7;
        # the swizzle is an involution, so it names the logical chunk too.
        dma_byte = [
            panel_off(8 * b + lane // 8, lane % 8 * 16) % 128
            for b in range_constexpr(2)
        ]

        def read_frag(off_i32):
            frag = fx.make_rmem_tensor(4, Int32)
            fx.copy(
                read16,
                fx.make_view(fx.add_offset(lds_i32, off_i32), fx.make_layout(4, 1)),
                frag,
            )
            return fx.memref_load_vec(frag)

        def read_b(base_i32, frag, lo=b_lo, hi=b_hi):
            frag.store(
                read_frag(base_i32 + lo).shuffle(
                    read_frag(base_i32 + hi), list(range(8))
                )
            )

        def opaque(v):
            """``v`` in a VGPR the optimizer can't see through: constants added
            to it then fold into the 16-bit ds offset instead of each becoming a
            hoisted address register."""
            return Int32(
                llvm.inline_asm(
                    T.i32, [Int32(v).ir_value()], "v_mov_b32 $0, $1", "=v,v"
                )
            )

        def settle(v):
            """``v`` computed here: a side-effecting no-op on it keeps the
            scheduler from sinking its producer down to the first real use."""
            x = v.ir_value()
            return Vec(
                llvm.inline_asm(x.type, [x], "", "=v,0", has_side_effects=True),
                dtype=Float32,
            )

        def h_bases(hb):
            """Per-lane bases of h buffer ``hb`` (0 but under ws 2), past
            64 KiB: the h panels' B reads and the requant's writes (column
            half c of a 128-K block), and the parked down scales."""
            hp = hb * (geo.h_bytes // 4)  # i32
            st_lo = opaque(OFF_STAGE // 4 + hp + b_lo)
            st_hi = opaque(OFF_STAGE // 4 + hp + b_hi)
            st_w = [
                opaque(
                    (OFF_STAGE + panel_off(lane16, c * 64 + wave * 16 + kgrp * 4)) // 4
                    + hp
                )
                for c in range_constexpr(2)
            ]
            act_r = opaque(OFF_ACT // 4 + hb * (geo.act_bytes // 4) + lane16)
            return st_lo, st_hi, st_w, act_r

        if const_expr(not pp):
            h_base0 = h_bases(0)
        # Likewise the amax scratch (written by lanes < 16 of each wave, read
        # by token lane16) and the relay (written by C lanes, read by the
        # merge as two dwords a lane per row), in their element units.
        amax_w = opaque(OFF_AMAX // 4 + wave * (n_blk * ROWS) + lane16)
        amax_r = opaque(OFF_AMAX // 4 + lane16)
        relay_w = opaque(OFF_RELAY // 2 + lane16 * RELAY_PITCH + wave * 16 + kgrp * 4)
        xs_r = OFF_XS // 4 + lane16

        def act_off(act_r, sl, b, gi):
            return act_r + (sl * n_blk + b) * ROWS + gi * 16

        relay_r = opaque(OFF_RELAY // 4 + wave * (rows_w * RELAY_PITCH // 2) + lane)
        # Wide partial stores: 16 bytes a lane, lpr lanes a relay row, so one
        # store covers rpi whole rows (row lane_row of the instruction).
        lane_row = lane // lpr
        relay_r4 = opaque(
            OFF_RELAY // 4
            + wave * (rows_w * RELAY_PITCH // 2)
            + lane_row * (RELAY_PITCH // 2)
            + lane % lpr * 4
        )

        mma = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, opsel_a=0, opsel_b=0
            )
        )
        unit = fx.Int32(127)  # e8m0 1.0

        def product(a, b):
            prod = fx.make_rmem_tensor(4, Float32)
            fx.gemm(mma, prod, a, b, zero, scale_a=unit, scale_b=unit)
            return prod.load()

        def mfma_stream(jobs, use, side=(), pre=None):
            """MFMA products into fresh registers; ``use(product, tag)`` of job i
            follows the issue of job i + MFMA_LAG, so the folds don't stall on the
            MFMA latency. The ``side`` thunks (loads) spread evenly between the
            MFMAs, so a full memory queue stalls them rather than the MFMAs.
            ``pre[i]`` runs just before job i; a job's B may be a thunk, taken
            when the job issues."""
            pend = []
            n_side = len(side)
            for i, (a, b, tag) in enumerate(jobs):
                if const_expr(pre is not None and i in pre):
                    pre[i]()
                for k in range_constexpr(
                    n_side * i // len(jobs), n_side * (i + 1) // len(jobs)
                ):
                    side[k]()
                pend.append((product(a, b() if callable(b) else b), tag))
                if const_expr(len(pend) > MFMA_LAG):
                    use(*pend.pop(0))
            for p, tag in pend:
                use(p, tag)

        # Registers that live across the loops.
        zero = fx.make_rmem_tensor(4, Float32)
        zero.store(fx.full(4, 0.0, Float32))
        acc_g = [
            [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_groups)]
            for _ in range_constexpr(n_j)
        ]
        acc_u = [
            [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_groups)]
            for _ in range_constexpr(n_j)
        ]
        b_dn = [
            [fx.make_rmem_tensor(8, Int32) for _ in range_constexpr(n_blk)]
            for _ in range_constexpr(2)
        ]  # [buffer][kb]
        w_gu = [
            [
                [fx.make_rmem_tensor(8, Int32) for _ in range_constexpr(n_j)]
                for _ in range_constexpr(len(phases[0][1]))
            ]
            for _ in range_constexpr(2)
        ]  # [buffer][K half of the phase][j]
        w_d = [
            [
                [fx.make_rmem_tensor(8, Int32) for _ in range_constexpr(n_blk)]
                for _ in range_constexpr(n_t)
            ]
            for _ in range_constexpr(2)
        ]  # [buffer][t][kb]

        # Persistent walk: pid % 8 picks the XCD; each XCD takes a contiguous
        # run of (row block, column tile) pairs, column tile fastest. The last,
        # partial round goes to every other slot of all XCDs alike.
        xcd = bid % 8
        slot = bid // 8
        lin = xcd * (grid // 8) + slot
        n_valid = sload(nvalid_src, 0)
        n_work = (n_valid + (ROWS - 1)) // ROWS * ps_deno
        full = n_work // grid
        tail = slot % 2 * (grid // 2) + xcd * (grid // 16) + slot // 2
        n_tiles = full + (tail < n_work - full * grid).select(fx.Int32(1), fx.Int32(0))

        def now():
            """The 100 MHz constant clock, low word."""
            return Int32(
                llvm.inline_asm(
                    T.i32,
                    [],
                    "s_memrealtime s[0:1]\ns_waitcnt lgkmcnt(0)\ns_mov_b32 $0, s0",
                    "=s,~{s0},~{s1}",
                    has_side_effects=True,
                )
            )

        # Down admission: a device-wide count of workgroups in their down phase.
        sem = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), arg_sem.ir_value())

        def sem_add(v):
            llvm.atomicrmw(
                llvm.AtomicBinOp.add,
                sem,
                fx.Int32(v).ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

        def sem_load():
            return Int32(
                llvm.load(
                    T.i32,
                    sem,
                    ordering=llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
            )

        def barrier():
            """``gpu.barrier`` that is never merged into an adjacent one (the
            canonicalizer drops back-to-back barriers): under ``ws`` 2 each
            role counts the other's barriers one for one."""
            llvm.inline_asm(None, [], "", "", has_side_effects=True)
            gpu.barrier()

        def step_barrier():
            """The gate/up step, amax, down turn and handoff barrier:
            unmergeable under ws 2."""
            if const_expr(pp):
                barrier()
            else:
                gpu.barrier()

        def idle(n):
            """ws 2: a role with no tile this round passes the other's barriers."""
            for _ in range(n):
                barrier()

        def make_merge(row0):
            """The merge of a tile's relay turns into O, or under ``stored``
            into the partials P of slots below it, for the wave's relay rows
            ``rows_w wave ..``: their sorted ids are wave-uniform."""
            out_id = [
                Int32(sload(ids_src, row0 + wave * rows_w + i))
                for i in range_constexpr(rows_w)
            ]
            out_tok = [v & 0xFFFFFF for v in out_id]
            out_slot = [v >> 24 for v in out_id]
            n_runs = rows_w * pieces

            def merge(turn, slot, part, n_part):
                """Adds part ``part`` of ``n_part`` of the wave's relay rows into O:
                a dword a 128-column run a lane, each atomic one contiguous
                256-byte run."""
                if const_expr("atomic" in KO):
                    return
                if const_expr(stored > 0):
                    n_q = rows_w // rpi
                    qs = list(range(n_q))[
                        part * n_q // n_part : (part + 1) * n_q // n_part
                    ]
                    if const_expr(len(qs) > 1):
                        merge_batch(turn, slot, qs)
                    else:
                        for q in qs:
                            merge_wide(turn, slot, q)
                    return
                sel = list(range(n_runs))[
                    part * n_runs // n_part : (part + 1) * n_runs // n_part
                ]
                pairs = [
                    fx.ptr_load(
                        fx.add_offset(
                            lds_i32,
                            relay_r
                            + (
                                slot * RELAY // 4
                                + i // pieces * (RELAY_PITCH // 2)
                                + i % pieces * 64
                            ),
                        )
                    )
                    for i in sel
                ]
                # The turn offset rides in soffset; the token row stays in
                # voffset, the only part the range check sees.
                atom = atomic_bf16x2.set_value("soffset", turn * TURN_N)
                for i, pair in zip(sel, pairs):
                    col = (i % pieces) * 128 + lane * 2
                    frag = fx.make_rmem_tensor(2, BFloat16)
                    frag.store(
                        Vec.from_elements([Int32(pair)], dtype=Int32).bitcast(BFloat16)
                    )
                    fx.copy(
                        atom,
                        frag,
                        fx.slice(o_dst, (None, out_tok[i // pieces] * D + col)),
                    )

            def read_quad(slot, q):
                """Relay rows rpi q .. rpi q + rpi - 1, 16 bytes a lane."""
                frag = fx.make_rmem_tensor(4, Int32)
                fx.copy(
                    read16,
                    fx.make_view(
                        fx.add_offset(
                            lds_i32,
                            relay_r4
                            + (slot * RELAY // 4 + q * rpi * (RELAY_PITCH // 2)),
                        ),
                        fx.make_layout(4, 1),
                    ),
                    frag,
                )
                return frag

            def quad_store(turn, q, frag):
                """One 16-byte partial store a lane: the lane's row, of rpi."""
                tok = out_tok[q * rpi]
                ps = out_slot[q * rpi]
                for j in range_constexpr(1, rpi):
                    tok = (lane_row == j).select(out_tok[q * rpi + j], tok)
                    ps = (lane_row == j).select(out_slot[q * rpi + j], ps)
                st = store_x4.set_value("soffset", turn * (TURN_N // 2))
                p_at = fx.slice(
                    p_i32, (None, (tok * stored + ps) * (D // 2) + lane % lpr * 4)
                )
                fx.copy(st, frag, p_at)

            def all_stored(rows):
                ok = out_slot[rows[0]] < stored
                for r in rows[1:]:
                    ok = ok & (out_slot[r] < stored)
                return ok

            def merge_batch(turn, slot, qs):
                """merge_wide over ``qs`` with every relay read issued before
                the first store when all the rows are stored slots: the stores
                then leave back to back, not one per LDS round trip."""
                rows = [q * rpi + j for q in qs for j in range_constexpr(rpi)]
                if all_stored(rows):
                    frags = [read_quad(slot, q) for q in qs]
                    for q, frag in zip(qs, frags):
                        quad_store(turn, q, frag)
                else:
                    for q in qs:
                        merge_wide(turn, slot, q)

            def merge_wide(turn, slot, q):
                """Relay rows rpi q .. of the wave, one 16-byte store a lane
                when all of them are stored slots; otherwise a dword at a time
                (stores or atomics)."""
                rows = [q * rpi + j for j in range_constexpr(rpi)]
                if all_stored(rows):
                    quad_store(turn, q, read_quad(slot, q))
                else:
                    merge_rows(turn, slot, rows)

            def merge_rows(turn, slot, rows):
                atom = atomic_bf16x2.set_value("soffset", turn * TURN_N)
                for r in rows:
                    ps = out_slot[r]
                    st = store_bf16x2.set_value("soffset", ps * D + turn * TURN_N)
                    for d in range_constexpr(pieces):
                        pair = fx.ptr_load(
                            fx.add_offset(
                                lds_i32,
                                relay_r
                                + (slot * RELAY // 4 + r * (RELAY_PITCH // 2) + d * 64),
                            )
                        )
                        frag = fx.make_rmem_tensor(2, BFloat16)
                        frag.store(
                            Vec.from_elements([Int32(pair)], dtype=Int32).bitcast(
                                BFloat16
                            )
                        )
                        col = d * 128 + lane * 2
                        # Slots below ``stored`` store into their P plane, the
                        # rest add into O (a wave-uniform branch).
                        if ps < stored:
                            fx.copy(
                                st,
                                frag,
                                fx.slice(
                                    p_dst, (None, out_tok[r] * (stored * D) + col)
                                ),
                            )
                        else:
                            fx.copy(
                                atom,
                                frag,
                                fx.slice(o_dst, (None, out_tok[r] * D + col)),
                            )

            return merge

        def compute_tile(eb, nb, gu=True, dn=True, hb=0):
            """Tile (eb, nb): gate/up and the requant (``gu``) into h buffer
            ``hb``, then down and the output (``dn``); ws 2 splits the roles."""
            row0 = eb * ROWS
            expert = sload(eids_src, eb)
            e64 = fx.Int64(expert)
            gu_w = _buffer(arg_gu + e64 * (2 * I * D), T.i32, 16, fx.Int64(2 * I * D))
            dw_w = _buffer(arg_dw + e64 * (D * I), T.i32, 16, fx.Int64(D * I))
            if const_expr(pp):
                st_lo, st_hi, st_w, act_r = h_bases(hb)
            else:
                st_lo, st_hi, st_w, act_r = h_base0

            # Wave w gathers token groups w + 4 q: rows 16 (w + 4 q) + 8 b + (l >> 3).
            dma_src = []
            for q in range_constexpr(n_dma * gu):
                for b in range_constexpr(2):
                    row = row0 + (wave + 4 * q) * 16 + b * 8 + lane // 8
                    tok = vload(load_i32, ids_src, row, Int32) & 0xFFFFFF
                    dma_src.append(tok * D + dma_byte[b])
            # The token-scale DMA: wave w moves half kh = w & 1 of rows
            # 64 xh + l, xh = (w >> 1) % (rows / 64) (at 64 rows waves 2, 3
            # repeat 0, 1). The padding token clamps to a real one: its
            # products are zero anyway. Routing weights of the B lanes, group gi.
            xh = wave // 2 % (ROWS // 64)
            tok = vload(load_i32, ids_src, row0 + xh * 64 + lane, Int32) & 0xFFFFFF
            tok_x = (tok < i32_tokens).select(tok, i32_tokens - 1)
            rw_b = [
                vload(load_f32, rw_src, row0 + gi * 16 + lane16, Float32)
                for gi in range_constexpr(n_groups)
            ]

            def issue_tokens(s, buf):
                for kh in range_constexpr(2):
                    for q in range_constexpr(n_dma):
                        for b in range_constexpr(2):
                            lds_off = (
                                buf * PANEL
                                + stack_off(kh, 0)
                                + (wave + 4 * q) * 2048
                                + b * 1024
                            )
                            dst = fx.make_view(
                                fx.add_offset(
                                    lds_i8, rocdl.readfirstlane(T.i32, lds_off)
                                ),
                                fx.make_layout(1, 1),
                            )
                            fx.copy(
                                dma16.set_value("soffset", (s * 2 + kh) * BLK),
                                fx.slice(x_src, (None, dma_src[q * 2 + b])),
                                dst,
                            )
                xs_dst = fx.make_view(
                    fx.add_offset(
                        lds_f32,
                        rocdl.readfirstlane(
                            T.i32, OFF_XS // 4 + (buf * 2 + wave % 2) * ROWS + xh * 64
                        ),
                    ),
                    fx.make_layout(1, 1),
                )
                fx.copy(
                    dma4.set_value("soffset", (s * 2 + wave % 2) * i32_tokens),
                    fx.slice(xq_src, (None, tok_x)),
                    xs_dst,
                )

            # Weight chunk of lane l: row group rg, 64-K plane p at
            # rg * 16 * K + p * 1024 + 16 l (bytes) of the expert's matrix. All
            # but the lane term is wave-uniform and rides in soffset, so every
            # weight load shares the one lane VGPR.
            lane_w = lane * 4  # i32

            def load_weights(src, rg, k_rows, plane, frag):
                atom = load_w.set_value(
                    "soffset", rg * (16 * k_rows // 4) + plane * 256
                )
                lo = fx.make_rmem_tensor(4, Int32)
                hi = fx.make_rmem_tensor(4, Int32)
                fx.copy(atom, fx.slice(src, (None, lane_w)), lo)
                fx.copy(atom, fx.slice(src, (None, lane_w + 256)), hi)
                frag.store(lo.load().shuffle(hi.load(), list(range(8))))

            def matrix_loads(w, s, phase, cs):
                up, khs = phase

                def one(k, kh, j):
                    rg = cs * (cols // 16) + wave + 4 * j + (I // 16 if up else 0)
                    return lambda: load_weights(gu_w, rg, D, s * 4 + kh * 2, w[k][j])

                return [
                    one(k, kh, j)
                    for k, kh in enumerate(khs)
                    for j in range_constexpr(n_j)
                ]

            def load_matrix(w, s, phase, cs):
                for f in matrix_loads(w, s, phase, cs):
                    f()

            def down_loads(turn, cs, w):
                def one(t, kb):
                    rg = turn * (TURN_N // 16) + 4 * t + wave
                    return lambda: load_weights(
                        dw_w, rg, I, cs * (cols // 64) + kb * 2, w[t][kb]
                    )

                return [
                    one(t, kb)
                    for t in range_constexpr(n_t)
                    for kb in range_constexpr(n_blk)
                ]

            def load_down(turn, cs, w):
                for f in down_loads(turn, cs, w):
                    f()

            # The token B operands of a phase, (kh, group) in MFMA order. Each
            # is read from the panel one group ahead and lives for its n_j
            # MFMAs only; the gate and up phases each read the panel once.
            def read_tok(buf, kh, gi):
                frag = fx.make_rmem_tensor(8, Int32)
                read_b(buf * (PANEL // 4) + stack_off(kh, gi) // 4, frag)
                return frag

            def fold(w, acc, ws, khs, xs, side, buf, b0, b_next):
                """One phase's sweep over K halves ``khs``: ``b0`` is its first B,
                already read; returns ``b_next()``, read during its last group."""
                order = [(kh, gi) for kh in khs for gi in range_constexpr(n_groups)]
                frags = {0: b0}

                def ahead(q):
                    def f():
                        if const_expr(q + 1 < len(order)):
                            frags[q + 1] = read_tok(buf, *order[q + 1])
                        else:
                            frags["next"] = b_next()

                    return f

                def use(p, tag):
                    kh, j, gi = tag
                    a = acc[j][gi]
                    scale = Vec.filled(
                        4, xs[kh * n_groups + gi] * ws[kh][j * 64 // BLK], Float32
                    )
                    a.store(settle(fx.fma(p, scale, a.load())))

                mfma_stream(
                    [
                        (w[k][j], lambda q=q: frags[q], (kh, j, gi))
                        for q, (kh, gi) in enumerate(order)
                        for j in range_constexpr(n_j)
                        for k in [khs.index(kh)]
                    ],
                    use,
                    side,
                    {q * n_j: ahead(q) for q in range_constexpr(len(order))},
                )
                return frags["next"]

            def gate_up_prologue(cs):
                # Each phase's weights load during the phase before; the step's
                # first, with the token scales, during the last of the previous.
                # The DMA goes first: the step's wait then covers it alone.
                issue_tokens(0, 0)
                rocdl.sched_barrier(0)
                load_matrix(w_gu[0], 0, phases[0], cs)

            def gate_up_step(s, last, cs):
                buf = s % 2
                gq_row = expert * (2 * I // BLK) + cs * (cols // BLK)
                ws = [
                    [
                        [
                            sload_f32(
                                gq_src,
                                (gq_row + up * (I // BLK) + b) * kd + s * 2 + kh,
                            )
                            for b in range_constexpr(cols // BLK)
                        ]
                        for kh in range_constexpr(2)
                    ]
                    for up in range_constexpr(2)
                ]
                # vmcnt is in order: waiting for the step's token DMA means
                # waiting until only the first phase's weight loads, issued
                # after it, are in flight. Those stay in flight into its MFMAs.
                rocdl.s_waitcnt(vmcnt=2 * len(phases[0][1]) * n_j)
                step_barrier()
                b = read_tok(buf, phases[0][1][0], 0)
                # Each phase loads the next one's weights; the next step's
                # tokens load during the second-to-last phase.
                sides = []
                for p in range_constexpr(len(phases)):
                    if const_expr(p + 1 < len(phases)):
                        sides.append(
                            matrix_loads(w_gu[(p + 1) % 2], s, phases[p + 1], cs)
                        )
                    elif const_expr(not last):
                        sides.append(matrix_loads(w_gu[0], s + 1, phases[0], cs))
                    else:
                        sides.append([])
                if const_expr(not last):
                    sides[-2].append(lambda: issue_tokens(s + 1, 1 - buf))
                xs = [
                    fx.ptr_load(
                        fx.add_offset(
                            lds_f32,
                            xs_r + (buf * 2 + kh) * ROWS + gi * 16,
                        )
                    )
                    for kh in range_constexpr(2)
                    for gi in range_constexpr(n_groups)
                ]
                for p, (up, khs) in enumerate(phases):
                    if const_expr(p + 1 < len(phases)):
                        nxt = phases[p + 1][1][0]
                        b_next = lambda kh=nxt: read_tok(buf, kh, 0)
                    else:
                        b_next = lambda: None
                    if const_expr(pp and p > 0 and p % (len(phases) // pp_a) == 0):
                        barrier()  # ws 2: one down turn a len(phases) / pp_a phases
                    rocdl.sched_barrier(0)
                    b = fold(
                        w_gu[p % 2],
                        acc_u if up else acc_g,
                        ws[up],
                        khs,
                        xs,
                        sides[p],
                        buf,
                        b,
                        b_next,
                    )

            # The tile's column slices run gate/up one after another; each parks
            # its quantized h in its own stage panel for one down pass over all.
            act_s = [
                [[None] * n_blk for _ in range_constexpr(n_groups)]
                for _ in range_constexpr(slices)
            ]
            if const_expr(gu):
                gate_up_prologue(nb * slices)
                for sl in range_constexpr(slices):
                    cs = nb * slices + sl
                    for j in range_constexpr(n_j):
                        for gi in range_constexpr(n_groups):
                            acc_g[j][gi].store(fx.full(4, 0.0, Float32))
                            acc_u[j][gi].store(fx.full(4, 0.0, Float32))
                    for s in range(n_steps - 1):
                        gate_up_step(s, False, cs)
                    gate_up_step(n_steps - 1, True, cs)

                    # h = silu(g) * u, then the per-token 1x128 requant. Column block
                    # b holds tiles j = 2 b, 2 b + 1.
                    h = [[None] * n_groups for _ in range_constexpr(n_j)]
                    for j in range_constexpr(n_j):
                        for gi in range_constexpr(n_groups):
                            g = acc_g[j][gi].load()
                            u = acc_u[j][gi].load()
                            vals = []
                            for r in range_constexpr(4):
                                e = fx.exp2(g[r] * fx.Float32(-LOG2E), fastmath="afn")
                                sig = Float32(
                                    rocdl.rcp(T.f32, (fx.Float32(1.0) + e).ir_value())
                                )
                                vals.append(g[r] * sig * u[r])
                            h[j][gi] = vals
                    # The next slice's first loads, or the first down weights, land
                    # during the requant; issued only now, with h (half the
                    # accumulators' registers) live, nothing spills.
                    rocdl.sched_barrier(0)
                    if const_expr(sl + 1 < slices):
                        gate_up_prologue(cs + 1)
                    else:
                        if const_expr(dn):
                            load_down(fx.Int32(0), nb * slices, w_d[0])
                    rocdl.sched_barrier(0)
                    for b in range_constexpr(n_blk):
                        for gi in range_constexpr(n_groups):
                            m = fx.Float32(0.0)
                            for j in range_constexpr(2 * b, 2 * b + 2):
                                for r in range_constexpr(4):
                                    m = fx.maxnumf(
                                        m, fx.maxnumf(h[j][gi][r], -h[j][gi][r])
                                    )
                            m = fx.maxnumf(m, m.shuffle_xor(16, WAVE))
                            m = fx.maxnumf(m, m.shuffle_xor(32, WAVE))
                            # All four lanes of a token store the same max: no
                            # branch, so the folds above aren't sunk past one.
                            fx.ptr_store(
                                m, fx.add_offset(lds_f32, amax_w + b * ROWS + gi * 16)
                            )
                    step_barrier()
                    for gi in range_constexpr(n_groups):
                        for b in range_constexpr(n_blk):
                            m = fx.Float32(1e-6)
                            for w in range_constexpr(4):
                                m = fx.maxnumf(
                                    m,
                                    fx.ptr_load(
                                        fx.add_offset(
                                            lds_f32,
                                            amax_r + (w * n_blk + b) * ROWS + gi * 16,
                                        )
                                    ),
                                )
                            q = fx.Float32(FP8_MAX) * Float32(
                                rocdl.rcp(T.f32, Float32(m).ir_value())
                            )
                            a = Float32(m) * fx.Float32(1.0 / FP8_MAX) * rw_b[gi]
                            if const_expr(sl < n_park):
                                # Every lane of the token stores the same value.
                                fx.ptr_store(
                                    a, fx.add_offset(lds_f32, act_off(act_r, sl, b, gi))
                                )
                            else:
                                act_s[sl][gi][b] = a
                            for j in range_constexpr(2 * b, 2 * b + 2):
                                v = [
                                    (h[j][gi][r] * q).ir_value()
                                    for r in range_constexpr(4)
                                ]
                                pk = rocdl.cvt_pk_fp8_f32(
                                    T.i32, v[0], v[1], fx.Int32(0).ir_value(), 0
                                )
                                pk = rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], pk, 1)
                                # C lane: token lane16, column 64 (j & 1) + 16 w + 4 kgrp of block b.
                                off = (
                                    st_w[j % 2] + (sl * h_panel + stack_off(b, gi)) // 4
                                )
                                fx.ptr_store(Int32(pk), fx.add_offset(lds_i32, off))
                step_barrier()  # ws 2: h of this tile handed to the down waves

            if const_expr(dn):
                if const_expr(not gu):
                    # ws 2: the first down weights (gate/up issues them during
                    # its requant otherwise).
                    load_down(fx.Int32(0), nb * slices, w_d[0])
                for sl in range_constexpr(n_park):
                    for gi in range_constexpr(n_groups):
                        for b in range_constexpr(n_blk):
                            act_s[sl][gi][b] = fx.ptr_load(
                                fx.add_offset(lds_f32, act_off(act_r, sl, b, gi))
                            )

                # Admission: wait (bounded) while down_limit workgroups are in down.
                if const_expr(down_limit < grid):  # noqa: SIM102 (trace-time if)
                    if tid == 0:
                        t_dead = now() + ADMIT_WAIT
                        while (sem_load() >= down_limit) & (now() - t_dead < 0):
                            llvm.inline_asm(
                                None, [], "s_sleep 2", "", has_side_effects=True
                            )
                        sem_add(1)
                if const_expr(not pp):
                    gpu.barrier()
                merge = make_merge(row0)

                def down_turn(turn, slot, u0, prev):
                    """MFMAs of ``turn`` into relay ``slot``, one unit per column slice
                    (K = cols, ``n_blk`` 128-K blocks), the weights of unit u in ``w_d[u % 2]``; meanwhile loads
                    the next unit's weights and merges turn ``prev`` (the other slot).
                    """
                    y = {}
                    for sl in range_constexpr(slices):
                        u = u0 + sl
                        if const_expr(sl + 1 < slices):
                            side = down_loads(
                                turn, nb * slices + sl + 1, w_d[(u + 1) % 2]
                            )
                        else:
                            # The last turn re-reads its own weights (an L2 hit).
                            nxt = (turn + 1 < n_turns).select(
                                turn + 1, fx.Int32(n_turns - 1)
                            )
                            side = down_loads(nxt, nb * slices, w_d[(u + 1) % 2])
                        w = w_d[u % 2]

                        def use(p, tag):
                            t, gi, kb, s = tag
                            scale = Vec.filled(4, act_s[sl][gi][kb] * s, Float32)
                            if const_expr(sl == 0 and kb == 0):
                                y[t, gi] = p * scale
                            elif const_expr(sl + 1 < slices or kb + 1 < n_blk):
                                y[t, gi] = fx.fma(p, scale, y[t, gi])
                            else:
                                v = fx.fma(p, scale, y.pop((t, gi)))
                                frag = fx.make_rmem_tensor(4, BFloat16)
                                frag.store(v.to(BFloat16))
                                idx = relay_w + (
                                    slot * RELAY // 2 + gi * 16 * RELAY_PITCH + t * 64
                                )
                                fx.copy(
                                    write8,
                                    frag,
                                    fx.make_view(
                                        fx.add_offset(lds_bf16, idx),
                                        fx.make_layout(4, 1),
                                    ),
                                )

                        # Weight scales [t][kb]: tiles 2 c, 2 c + 1 share output block c.
                        ws = [
                            [
                                sload_f32(
                                    dq_src,
                                    (
                                        expert * kd
                                        + turn * (TURN_N // BLK)
                                        + t * 64 // BLK
                                    )
                                    * ki
                                    + (nb * slices + sl) * (cols // BLK)
                                    + kb,
                                )
                                for kb in range_constexpr(n_blk)
                            ]
                            for t in range_constexpr(n_t)
                        ]

                        def read_dn(gi, frags):
                            for kb in range_constexpr(n_blk):
                                read_b(
                                    (sl * h_panel + stack_off(kb, gi)) // 4,
                                    frags[kb],
                                    st_lo,
                                    st_hi,
                                )

                        # Token group major: a group's B is read from the h panel one
                        # group ahead and lives for its n_t * n_blk MFMAs only; the
                        # merge of the previous turn follows each group. One stream,
                        # so the products of a group fold under the next group's.
                        per = n_t * n_blk

                        def at_group(gi):
                            def f():
                                if const_expr(gi > 0 and prev is not None):
                                    merge(
                                        prev, 1 - slot, sl * n_groups + gi - 1, n_parts
                                    )
                                if const_expr(gi + 1 < n_groups):
                                    read_dn(gi + 1, b_dn[(gi + 1) % 2])

                            return f

                        read_dn(0, b_dn[0])
                        mfma_stream(
                            [
                                (w[t][kb], b_dn[gi % 2][kb], (t, gi, kb, ws[t][kb]))
                                for gi in range_constexpr(n_groups)
                                for t in range_constexpr(n_t)
                                for kb in range_constexpr(n_blk)
                            ],
                            use,
                            side,
                            {
                                gi * per: at_group(gi)
                                for gi in range_constexpr(n_groups)
                            },
                        )
                        if const_expr(prev is not None):
                            merge(prev, 1 - slot, sl * n_groups + n_groups - 1, n_parts)
                    step_barrier()

                # Turns go in pairs so the relay slots alternate; turn t's merge
                # runs under turn t + 1's MFMAs. Under ws 2 the turns run in one
                # block a slice, each closed by a barrier that matches gate/up's
                # amax exchange.
                u1 = slices % 2  # the second turn's first weight buffer
                per = n_turns // slices if const_expr(pp) else n_turns
                for blk in range_constexpr(slices if pp else 1):
                    if const_expr(blk == 0):
                        down_turn(fx.Int32(0), 0, 0, None)
                        down_turn(fx.Int32(1), 1, u1, fx.Int32(0))
                    for pair in range(max(1, blk * per // 2), (blk + 1) * per // 2):
                        turn = pair * 2
                        down_turn(turn, 0, 0, turn - 1)
                        down_turn(turn + 1, 1, u1, turn)
                    if const_expr(pp):
                        barrier()
                merge(fx.Int32(n_turns - 1), 1, 0, 1)
                if const_expr(pp):
                    barrier()  # matches gate/up's h handoff
                else:
                    # The next tile's first DMA lands in the relay: all merges first.
                    gpu.barrier()
                if const_expr(down_limit < grid):  # noqa: SIM102 (trace-time if)
                    if tid == 0:
                        sem_add(-1)

        def tile_at(it):
            w_idx = (it < full).select(it * grid + lin, full * grid + tail)
            return w_idx // ps_deno, w_idx % ps_deno

        if const_expr(pp):
            # Round r: waves 0-3 run gate/up of tile r into h buffer r % 2,
            # waves 4-7 down and the output of tile r - 1, both passing n_pp
            # barriers.
            for it in range(n_tiles + 1):
                eb, nb = tile_at(it)
                eb_d, nb_d = tile_at(it - 1)
                if wave_g < 4:
                    if it < n_tiles:
                        compute_tile(eb, nb, dn=False, hb=it % 2)
                    else:
                        idle(n_pp)
                else:
                    if it > 0:
                        compute_tile(eb_d, nb_d, gu=False, hb=(it - 1) % 2)
                    else:
                        idle(n_pp)
        else:
            for it in range(n_tiles):
                eb, nb = tile_at(it)
                compute_tile(eb, nb)

    kernel_fmoe(
        fx.Int64(fx.ptrtoint(arg_o)),
        fx.Int64(fx.ptrtoint(arg_x)),
        fx.Int64(fx.ptrtoint(arg_gu)),
        fx.Int64(fx.ptrtoint(arg_dw)),
        fx.Int64(fx.ptrtoint(arg_ids)),
        fx.Int64(fx.ptrtoint(arg_rw)),
        fx.Int64(fx.ptrtoint(arg_eids)),
        fx.Int64(fx.ptrtoint(arg_nvalid)),
        fx.Int64(fx.ptrtoint(arg_xq)),
        fx.Int64(fx.ptrtoint(arg_gq)),
        fx.Int64(fx.ptrtoint(arg_dq)),
        fx.Int64(fx.ptrtoint(arg_sem)),
        fx.Int64(fx.ptrtoint(arg_p)),
        i32_tokens,
        i32_experts,
        value_attrs={"rocdl.waves_per_eu": 2 if pp else 1},
    ).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)
