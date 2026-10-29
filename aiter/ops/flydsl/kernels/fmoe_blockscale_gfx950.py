# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  (the per-tile closures are traced within their iteration)

"""Fused blockscale-fp8 g1u1 silu MoE on gfx950, one persistent launch.

The FlyDSL counterpart of the asm ``fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps``
kernels: gate/up GEMM, silu(gate) * up, a per-token 1x128 fp8 requant, the down
GEMM and the routed bf16 merge into the output, per 64-row sorted block and
``slices * cols`` intermediate columns.

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

Sorting is aiter ``moe_sorting`` with block 64: ``sorted_ids`` holds
``token | slot << 24`` (padding rows hold token T), ``sorted_expert_ids`` one
expert per 64-row block and ``num_valid_ids[0]`` the sorted row count.

Work split. A tile is (row block, ``slices`` column slices of ``cols``, 256 or
384): gate/up runs slice by slice, each leaving its quantized h in its own LDS
panel, then one down pass contracts over all of them (K = ``slices * cols``),
so the output atomics happen once a tile. One workgroup per CU walks the tiles
persistently, each XCD a contiguous run. Four waves split the columns: wave w
owns the 16-column MFMA tiles ``64 j + 16 w``, all 64 tokens.
MFMA 16x16x128 takes the weights as A and the tokens as B, so C lane l holds
token ``l % 16`` and the four weight rows ``4 (l // 16) + r``.

Token panel (LDS). One K step is 256 K: two 128-K halves (kh) of four 16-token
groups, a 128-byte row per token, 16 KiB, double buffered. Its 16-byte chunks
carry the Swizzle(3, 4, 4) of the dense row-major panel (chunk ^= (row >> 1) & 7),
so the 16 token rows a B read spans hit distinct banks. The gather is LDS DMA:
lane l lands 16 bytes at ``base + 16 l`` and fetches the logical chunk that the
swizzle maps there. The quantized h reuses the same panel layout (staging) as
the down GEMM's B operand.

Down. A turn is 256 output columns: wave w owns ``64 t + 16 w``. The bf16
results go through an LDS relay, then each wave adds its own 16 token rows into
O with packed bf16 buffer atomics; padding rows land past O's records and drop.
Turn t's merge interleaves with turn t + 1's MFMAs.

Down admission. Down is bound by the device-wide atomic throughput, and with
equal tiles all CUs reach it in lockstep. A global counter (zero between
launches) caps the workgroups in down at once; the rest wait, at most
``ADMIT_WAIT``, which offsets the CUs' phases: about 10% on D = 7168.

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
ROWS = 64  # sorted rows per tile (moe_sorting block)
BLK = 128  # scale block edge
STEP_K = 256  # K per gate/up step: two 128-K halves
TURN_N = 256  # down output columns per turn
MFMA_LAG = 3  # products in flight before the first fold reads one
FP8_MAX = 448.0
ADMIT_WAIT = 5000  # longest down-admission wait, 100 MHz ticks (50 us)

# LDS map, bytes. The down relay overlays the gate/up token panels, which are
# dead by then; at offset 0 every relay access folds its constant part into the
# 16-bit ds offset. The h panels (the down B operand) sit above it.
PANEL = 16 * 1024  # one token panel: (kh 2, group 4, token 16, 128 B)
RELAY_PITCH = TURN_N + 8  # bf16 per relay row; the pad staggers banks
OFF_RELAY = 0
RELAY = ROWS * RELAY_PITCH * 2
OFF_STAGE = OFF_RELAY + 2 * RELAY  # one h panel per column slice
H_BLOCK = 8 * 1024  # one 128-K block of an h panel: (group 4, token 16, 128 B)
# The requant's amax scratch, f32 [wave 4][block <= 3][token 64], also overlays
# the relay, past the token panels the next slice's prologue fills.
OFF_AMAX = 2 * PANEL
# The token scales of a step, f32 [buffer 2][kh 2][token 64], DMA'd with the
# token panels; past the amax scratch, which is live during the next prologue.
OFF_XS = OFF_AMAX + 4 * 3 * ROWS * 4
assert OFF_XS + 2 * 2 * ROWS * 4 <= OFF_STAGE and OFF_STAGE % 1024 == 0
COLS = (256, 384)  # columns of one slice


def lds_bytes(cols, slices):
    return OFF_STAGE + slices * (cols // BLK) * H_BLOCK


def check_config(model_dim, inter_dim, cols, slices=1):
    if cols not in COLS:
        raise ValueError(f"cols {cols}: the slice width is one of {COLS}")
    if slices not in (1, 2, 3) or inter_dim % (cols * slices):
        raise ValueError(
            f"inter_dim {inter_dim} does not split into {slices} slices of {cols} columns"
        )
    if model_dim % (2 * TURN_N):
        raise ValueError(f"model_dim {model_dim} is not a multiple of {2 * TURN_N}")


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
    i32_tokens: fx.Int32,
    i32_experts: fx.Int32,
    stream: fx.Stream,
    D: Constexpr[int],
    I: Constexpr[int],
    cols: Constexpr[int],
    slices: Constexpr[int],
    grid: Constexpr[int],
):
    check_config(D, I, cols, slices)
    ps_deno = I // (cols * slices)  # column tiles per row block
    ARENA = lds_bytes(cols, slices)
    assert grid % 16 == 0 and grid % ps_deno == 0
    # Workgroups let into down at once. Down is bound by the device's atomic
    # throughput and moves the same output bytes per tile whatever its width,
    # so narrower tiles (less MFMA per tile) admit fewer: (2 + n) / 8 of the
    # grid for n * 256 columns.
    down_limit = grid * (2 * TURN_N + slices * cols) // (8 * TURN_N)
    n_steps = D // STEP_K
    n_turns = D // TURN_N
    n_j = cols // 64  # 16-column MFMA tiles per wave
    n_groups = ROWS // 16
    kd = D // BLK  # K blocks of the gate/up GEMM
    ki = I // BLK  # K blocks of the down GEMM
    n_blk = cols // BLK  # 128-column blocks of a slice: down K blocks, scale rows
    h_panel = n_blk * H_BLOCK  # bytes of one slice's h panel
    kernel_name = format_kernel_name(
        f"fmoe_blockscale_g1u1_silu_{ROWS}x{cols}x{slices}_d{D}_i{I}"
    )

    @fx.struct
    class Arena:
        data: fx.Array[Int8, ARENA, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
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
        i32_tokens: fx.Int32,
        i32_experts: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
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
            return kh * 8192 + gi * 2048

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

        # The h panels sit past 64 KiB: per-lane bases for their B reads and
        # the requant's writes (column half c of a 128-K block).
        st_lo = opaque(OFF_STAGE // 4 + b_lo)
        st_hi = opaque(OFF_STAGE // 4 + b_hi)
        st_w = [
            opaque((OFF_STAGE + panel_off(lane16, c * 64 + wave * 16 + kgrp * 4)) // 4)
            for c in range_constexpr(2)
        ]
        # Likewise the amax scratch (written by lanes < 16 of each wave, read
        # by token lane16) and the relay (written by C lanes, read by the
        # merge as two dwords a lane per row), in their element units.
        amax_w = opaque(OFF_AMAX // 4 + wave * (n_blk * ROWS) + lane16)
        amax_r = opaque(OFF_AMAX // 4 + lane16)
        relay_w = opaque(OFF_RELAY // 2 + lane16 * RELAY_PITCH + wave * 16 + kgrp * 4)
        xs_r = OFF_XS // 4 + lane16
        relay_r = opaque(OFF_RELAY // 4 + wave * (16 * RELAY_PITCH // 2) + lane)

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
        w_g = [
            [fx.make_rmem_tensor(8, Int32) for _ in range_constexpr(2)]
            for _ in range_constexpr(n_j)
        ]
        w_u = [
            [fx.make_rmem_tensor(8, Int32) for _ in range_constexpr(2)]
            for _ in range_constexpr(n_j)
        ]
        w_d = [
            [
                [fx.make_rmem_tensor(8, Int32) for _ in range_constexpr(n_blk)]
                for _ in range_constexpr(4)
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

        for it in range(n_tiles):
            w_idx = (it < full).select(it * grid + lin, full * grid + tail)
            eb = w_idx // ps_deno
            nb = w_idx % ps_deno
            row0 = eb * ROWS
            expert = sload(eids_src, eb)
            e64 = fx.Int64(expert)
            gu_w = _buffer(arg_gu + e64 * (2 * I * D), T.i32, 16, fx.Int64(2 * I * D))
            dw_w = _buffer(arg_dw + e64 * (D * I), T.i32, 16, fx.Int64(D * I))

            # Wave w gathers token group w: rows 16 w + 8 b + (l >> 3).
            dma_src = []
            for b in range_constexpr(2):
                tok = (
                    vload(
                        load_i32, ids_src, row0 + wave * 16 + b * 8 + lane // 8, Int32
                    )
                    & 0xFFFFFF
                )
                dma_src.append(tok * D + dma_byte[b])
            # The token-scale DMA's lane token, row l (the padding token clamps
            # to a real one: its products are zero anyway); routing weights of
            # the B lanes, group gi.
            tok = vload(load_i32, ids_src, row0 + lane, Int32) & 0xFFFFFF
            tok_x = (tok < i32_tokens).select(tok, i32_tokens - 1)
            rw_b = [
                vload(load_f32, rw_src, row0 + gi * 16 + lane16, Float32)
                for gi in range_constexpr(n_groups)
            ]

            def issue_tokens(s, buf):
                for kh in range_constexpr(2):
                    for b in range_constexpr(2):
                        lds_off = (
                            buf * PANEL + stack_off(kh, 0) + wave * 2048 + b * 1024
                        )
                        dst = fx.make_view(
                            fx.add_offset(lds_i8, rocdl.readfirstlane(T.i32, lds_off)),
                            fx.make_layout(1, 1),
                        )
                        fx.copy(
                            dma16.set_value("soffset", (s * 2 + kh) * BLK),
                            fx.slice(x_src, (None, dma_src[b])),
                            dst,
                        )
                # Token scales of half kh = wave & 1 (waves 2, 3 repeat 0, 1).
                xs_dst = fx.make_view(
                    fx.add_offset(
                        lds_f32,
                        rocdl.readfirstlane(
                            T.i32, OFF_XS // 4 + (buf * 2 + wave % 2) * ROWS
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

            def matrix_loads(w, s, up, cs):
                def one(kh, j):
                    rg = cs * (cols // 16) + wave + 4 * j + (I // 16 if up else 0)
                    return lambda: load_weights(gu_w, rg, D, s * 4 + kh * 2, w[j][kh])

                return [
                    one(kh, j)
                    for kh in range_constexpr(2)
                    for j in range_constexpr(n_j)
                ]

            def load_matrix(w, s, up, cs):
                for f in matrix_loads(w, s, up, cs):
                    f()

            def down_loads(turn, cs, w):
                def one(t, kb):
                    rg = turn * (TURN_N // 16) + 4 * t + wave
                    return lambda: load_weights(
                        dw_w, rg, I, cs * (cols // 64) + kb * 2, w[t][kb]
                    )

                return [
                    one(t, kb)
                    for t in range_constexpr(4)
                    for kb in range_constexpr(n_blk)
                ]

            def load_down(turn, cs, w):
                for f in down_loads(turn, cs, w):
                    f()

            # The token B operands of a step, (kh, group) in MFMA order. Each
            # is read from the panel one group ahead and lives for its n_j
            # MFMAs only; the gate and up sweeps each read the panel once.
            b_order = [
                (kh, gi)
                for kh in range_constexpr(2)
                for gi in range_constexpr(n_groups)
            ]

            def read_tok(buf, q):
                kh, gi = b_order[q]
                frag = fx.make_rmem_tensor(8, Int32)
                read_b(buf * (PANEL // 4) + stack_off(kh, gi) // 4, frag)
                return frag

            def fold(w, acc, ws, xs, side, buf, b0, b_next):
                """One projection's sweep over the step: ``b0`` is its first B,
                already read; returns ``b_next()``, read during its last group."""
                frags = {0: b0}

                def ahead(q):
                    def f():
                        if const_expr(q + 1 < len(b_order)):
                            frags[q + 1] = read_tok(buf, q + 1)
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
                        (w[j][kh], lambda q=q: frags[q], (kh, j, gi))
                        for q, (kh, gi) in enumerate(b_order)
                        for j in range_constexpr(n_j)
                    ],
                    use,
                    side,
                    {q * n_j: ahead(q) for q in range_constexpr(len(b_order))},
                )
                return frags["next"]

            def gate_up_prologue(cs):
                # Gate weights and token scales run a step ahead and land during
                # the up MFMAs; the up weights land during the gate MFMAs.
                # The DMA goes first: the step's wait then covers it alone.
                issue_tokens(0, 0)
                rocdl.sched_barrier(0)
                load_matrix(w_g, 0, False, cs)

            def gate_up_step(s, last, cs):
                buf = s % 2
                gq_row = expert * (2 * I // BLK) + cs * (cols // BLK)
                ws_g = [
                    [
                        sload_f32(gq_src, (gq_row + b) * kd + s * 2 + kh)
                        for b in range_constexpr(cols // BLK)
                    ]
                    for kh in range_constexpr(2)
                ]
                ws_u = [
                    [
                        sload_f32(gq_src, (gq_row + I // BLK + b) * kd + s * 2 + kh)
                        for b in range_constexpr(cols // BLK)
                    ]
                    for kh in range_constexpr(2)
                ]
                # vmcnt is in order: waiting for the step's token DMA means
                # waiting until only the 4 n_j gate-weight loads issued after it
                # are in flight. Those stay in flight into the gate MFMAs.
                rocdl.s_waitcnt(vmcnt=4 * n_j)
                gpu.barrier()
                b0 = read_tok(buf, 0)
                # The up weights and the next step's tokens load during the gate
                # MFMAs; the next gate weights and token scales during the up.
                side_g = matrix_loads(w_u, s, True, cs)
                side_u = []
                if const_expr(not last):
                    side_g.append(lambda: issue_tokens(s + 1, 1 - buf))
                    side_u = matrix_loads(w_g, s + 1, False, cs)
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
                rocdl.sched_barrier(0)
                b0 = fold(
                    w_g, acc_g, ws_g, xs, side_g, buf, b0, lambda: read_tok(buf, 0)
                )
                rocdl.sched_barrier(0)
                fold(w_u, acc_u, ws_u, xs, side_u, buf, b0, lambda: None)

            # The tile's column slices run gate/up one after another; each parks
            # its quantized h in its own stage panel for one down pass over all.
            act_s = [
                [[None] * n_blk for _ in range_constexpr(n_groups)]
                for _ in range_constexpr(slices)
            ]
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
                    load_down(fx.Int32(0), nb * slices, w_d[0])
                rocdl.sched_barrier(0)
                for b in range_constexpr(n_blk):
                    for gi in range_constexpr(n_groups):
                        m = fx.Float32(0.0)
                        for j in range_constexpr(2 * b, 2 * b + 2):
                            for r in range_constexpr(4):
                                m = fx.maxnumf(m, fx.maxnumf(h[j][gi][r], -h[j][gi][r]))
                        m = fx.maxnumf(m, m.shuffle_xor(16, WAVE))
                        m = fx.maxnumf(m, m.shuffle_xor(32, WAVE))
                        # All four lanes of a token store the same max: no
                        # branch, so the folds above aren't sunk past one.
                        fx.ptr_store(
                            m, fx.add_offset(lds_f32, amax_w + b * ROWS + gi * 16)
                        )
                gpu.barrier()
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
                        act_s[sl][gi][b] = (
                            Float32(m) * fx.Float32(1.0 / FP8_MAX) * rw_b[gi]
                        )
                        for j in range_constexpr(2 * b, 2 * b + 2):
                            v = [
                                (h[j][gi][r] * q).ir_value() for r in range_constexpr(4)
                            ]
                            pk = rocdl.cvt_pk_fp8_f32(
                                T.i32, v[0], v[1], fx.Int32(0).ir_value(), 0
                            )
                            pk = rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], pk, 1)
                            # C lane: token lane16, column 64 (j & 1) + 16 w + 4 kgrp of block b.
                            off = st_w[j % 2] + (sl * h_panel + stack_off(b, gi)) // 4
                            fx.ptr_store(Int32(pk), fx.add_offset(lds_i32, off))
            gpu.barrier()

            # Admission: wait (bounded) while down_limit workgroups are in down.
            if tid == 0:
                t_dead = now() + ADMIT_WAIT
                while (sem_load() >= down_limit) & (now() - t_dead < 0):
                    llvm.inline_asm(None, [], "s_sleep 2", "", has_side_effects=True)
                sem_add(1)
            gpu.barrier()

            # The wave's 16 relay rows: its token ids, wave-uniform.
            out_tok = [
                sload(ids_src, row0 + wave * 16 + i) & 0xFFFFFF
                for i in range_constexpr(16)
            ]

            def merge(turn, slot, part, n_part):
                """Adds part ``part`` of ``n_part`` of the wave's relay rows into O:
                two dwords a row a lane, each atomic one contiguous 256-byte run."""
                sel = list(range(32))[part * 32 // n_part : (part + 1) * 32 // n_part]
                pairs = [
                    fx.ptr_load(
                        fx.add_offset(
                            lds_i32,
                            relay_r
                            + (
                                slot * RELAY // 4
                                + i // 2 * (RELAY_PITCH // 2)
                                + i % 2 * 64
                            ),
                        )
                    )
                    for i in sel
                ]
                # The turn offset rides in soffset; the token row stays in
                # voffset, the only part the range check sees.
                atom = atomic_bf16x2.set_value("soffset", turn * TURN_N)
                for i, pair in zip(sel, pairs):
                    col = (i % 2) * 128 + lane * 2
                    frag = fx.make_rmem_tensor(2, BFloat16)
                    frag.store(
                        Vec.from_elements([Int32(pair)], dtype=Int32).bitcast(BFloat16)
                    )
                    fx.copy(
                        atom, frag, fx.slice(o_dst, (None, out_tok[i // 2] * D + col))
                    )

            def down_turn(turn, slot, u0, prev):
                """MFMAs of ``turn`` into relay ``slot``, one unit per column slice
                (K = cols, ``n_blk`` 128-K blocks), the weights of unit u in ``w_d[u % 2]``; meanwhile loads
                the next unit's weights and merges turn ``prev`` (the other slot)."""
                y = {}
                for sl in range_constexpr(slices):
                    u = u0 + sl
                    if const_expr(sl + 1 < slices):
                        side = down_loads(turn, nb * slices + sl + 1, w_d[(u + 1) % 2])
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
                                    fx.add_offset(lds_bf16, idx), fx.make_layout(4, 1)
                                ),
                            )

                    # Weight scales [t][kb]: tiles 2 c, 2 c + 1 share output block c.
                    ws = [
                        [
                            sload_f32(
                                dq_src,
                                (expert * kd + turn * (TURN_N // BLK) + t * 64 // BLK)
                                * ki
                                + (nb * slices + sl) * (cols // BLK)
                                + kb,
                            )
                            for kb in range_constexpr(n_blk)
                        ]
                        for t in range_constexpr(4)
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
                    # group ahead and lives for its eight MFMAs only.
                    read_dn(0, b_dn[0])
                    for gi in range_constexpr(n_groups):
                        if const_expr(gi + 1 < n_groups):
                            read_dn(gi + 1, b_dn[(gi + 1) % 2])
                        b = b_dn[gi % 2]
                        mfma_stream(
                            [
                                (w[t][kb], b[kb], (t, gi, kb, ws[t][kb]))
                                for t in range_constexpr(4)
                                for kb in range_constexpr(n_blk)
                            ],
                            use,
                            side[gi * n_blk : (gi + 1) * n_blk],
                        )
                        if const_expr(prev is not None):
                            merge(prev, 1 - slot, sl * 4 + gi, 4 * slices)
                gpu.barrier()

            # Turns go in pairs so the relay slots alternate; turn t's merge runs
            # under turn t + 1's MFMAs.
            u1 = slices % 2  # the second turn's first weight buffer
            down_turn(fx.Int32(0), 0, 0, None)
            down_turn(fx.Int32(1), 1, u1, fx.Int32(0))
            for pair in range(1, n_turns // 2):
                turn = pair * 2
                down_turn(turn, 0, 0, turn - 1)
                down_turn(turn + 1, 1, u1, turn)
            merge(fx.Int32(n_turns - 1), 1, 0, 1)
            if tid == 0:
                sem_add(-1)

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
        i32_tokens,
        i32_experts,
        value_attrs={"rocdl.waves_per_eu": 1},
    ).launch(grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream)
