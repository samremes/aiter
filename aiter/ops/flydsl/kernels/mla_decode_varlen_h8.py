"""Native eight-head sparse MLA decode for gfx950.

The product path uses one eight-wave CTA per ``(query, split)`` task.  Each
wave owns sixteen selected keys for QK and sixty-four value columns for PV.
Both products use ``v_mfma_f32_16x16x128_f8f6f4``; only the low eight rows of
the internal M16 fragment are live.
"""

# ruff: noqa: B008, SIM102

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import atomic_add_i32

NUM_HEADS = 8
QK_DIM = 576
V_DIM = 512
BLOCK_N = 128
NUM_WARPS = 8
WARP_SIZE = 64
NUM_THREADS = NUM_WARPS * WARP_SIZE
Q_DWORDS = NUM_HEADS * QK_DIM // 4
KV_ROW_DWORDS = QK_DIM // 4
KV_COL_BLOCK = 64
KV_COL_BLOCKS = QK_DIM // KV_COL_BLOCK
KV_SUB_ROWS = 4
KV_SUB_COLS = 32
KV_SLOT_BYTES = KV_SUB_ROWS * KV_SUB_COLS * 2 + 8
KV_SLOTS_PER_PASS = 8
KV_PASSES = BLOCK_N // 32
KV_BLOCK_BYTES = KV_SLOT_BYTES * KV_SLOTS_PER_PASS * KV_PASSES
KV_BUFFER_BYTES = KV_BLOCK_BYTES * KV_COL_BLOCKS

Q_OFFSET = 0
P_OFFSET = Q_OFFSET + NUM_HEADS * QK_DIM
P_RESIDUAL_OFFSET = P_OFFSET + NUM_HEADS * BLOCK_N
STATS_OFFSET = P_RESIDUAL_OFFSET + NUM_HEADS * BLOCK_N
KV0_OFFSET = STATS_OFFSET + NUM_WARPS * NUM_HEADS * 2 * 4
KV1_OFFSET = KV0_OFFSET + KV_BUFFER_BYTES
TOTAL_LDS_BYTES = KV1_OFFSET + KV_BUFFER_BYTES
OCCUPANCY = 1

assert TOTAL_LDS_BYTES == 159232
assert TOTAL_LDS_BYTES <= 160 * 1024


@fx.struct
class SharedStorage:
    storage: fx.Array[fx.Uint8, TOTAL_LDS_BYTES, 16]


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_decode_varlen_h8_fp8(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    kv_page_indices: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    q_scale: fx.Tensor,
    kv_scale: fx.Tensor,
    softmax_scale: fx.Float32,
    output_mode: fx.Constexpr,
    num_workers: fx.Constexpr,
):
    query_rsrc = buffer_ops.create_buffer_resource(query)
    kv_rsrc = buffer_ops.create_buffer_resource(kv_buffer)
    indices_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
    work_rsrc = buffer_ops.create_buffer_resource(work_info_set)
    work_indptr_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    split_output_rsrc = buffer_ops.create_buffer_resource(split_output)
    lse_rsrc = buffer_ops.create_buffer_resource(split_lse)
    q_scale_rsrc = buffer_ops.create_buffer_resource(q_scale)
    kv_scale_rsrc = buffer_ops.create_buffer_resource(kv_scale)

    q_scale_value = fx.Float32(
        buffer_ops.buffer_load(q_scale_rsrc, 0, vec_width=1, dtype=T.f32)
    )
    kv_scale_value = fx.Float32(
        buffer_ops.buffer_load(kv_scale_rsrc, 0, vec_width=1, dtype=T.f32)
    )
    qk_scale = softmax_scale * q_scale_value * kv_scale_value
    log2e = fx.Float32(1.4426950408889634)
    probability_scale = fx.Float32(256.0)
    probability_scale_recip = fx.Float32(1.0 / 256.0)
    neg_inf = fx.Float32(-1.0e30)
    zero_f32 = fx.Float32(0.0)
    zero_i32 = fx.Int32(0)

    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    lds_u8 = fx.recast_iter(fx.Uint8, lds.storage.ptr)
    transpose_v_copy = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(8, 64), fx.Uint8)

    def _lds_ptr(byte_offset):
        return lds_u8 + fx.Int32(byte_offset)

    def _load_i32(byte_offset):
        return _raw(
            fx.ptr_load(
                _lds_ptr(byte_offset), result_type=fx.Vector.make_type(4, fx.Uint8)
            ).bitcast(fx.Int32)[0]
        )

    def _store_i32(byte_offset, value):
        packed = Vec.from_elements([fx.Int32(value)], fx.Int32).bitcast(fx.Uint8)
        fx.ptr_store(packed, _lds_ptr(byte_offset))

    def _load_f32(byte_offset):
        return fx.Float32(
            fx.ptr_load(
                _lds_ptr(byte_offset), result_type=fx.Vector.make_type(4, fx.Uint8)
            ).bitcast(fx.Float32)[0]
        )

    def _store_f32(byte_offset, value):
        packed = Vec.from_elements([fx.Float32(value)], fx.Float32).bitcast(fx.Uint8)
        fx.ptr_store(packed, _lds_ptr(byte_offset))

    def _inline_asm_void(operands, asm_string, constraints):
        llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)

    def _mfma_fp8(a, b, c):
        neutral = arith.constant(0x7F7F7F7F, type=T.i32)
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.f32x4,
            [a, b, _raw(c), 0, 0, 0, neutral, 0, neutral],
        )

    def _mfma_fp8_bf8(a, b, c):
        neutral = arith.constant(0x7F7F7F7F, type=T.i32)
        return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.f32x4,
            [a, b, _raw(c), 0, 1, 0, neutral, 0, neutral],
        )

    def _shuffle_xor_f32(value, offset):
        bits = _raw(ArithValue(value).bitcast(T.i32))
        peer = ArithValue(bits).shuffle_xor(offset, WARP_SIZE)
        return fx.Float32(ArithValue(peer).bitcast(T.f32))

    def _reduce_max_16(value):
        result = fx.Float32(value)
        result = fx.maxnumf(result, _shuffle_xor_f32(result, 16))
        result = fx.maxnumf(result, _shuffle_xor_f32(result, 32))
        return result

    def _reduce_sum_16(value):
        result = fx.Float32(value)
        result = result + _shuffle_xor_f32(result, 16)
        result = result + _shuffle_xor_f32(result, 32)
        return result

    def _pack_probability4(values):
        scaled = [value * probability_scale for value in values]
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32,
            _raw(scaled[0]),
            _raw(scaled[1]),
            _raw(zero_i32),
            0,
        )
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32,
            _raw(scaled[2]),
            _raw(scaled[3]),
            packed,
            1,
        )
        low = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, packed, False))
        high = Vec(rocdl.cvt_pk_f32_fp8(T.f32x2, packed, True))
        rounded = [low[0], low[1], high[0], high[1]]
        residual = [scaled[index] - rounded[index] for index in range_constexpr(4)]
        residual_packed = rocdl.cvt_pk_bf8_f32(
            T.i32,
            _raw(residual[0]),
            _raw(residual[1]),
            _raw(zero_i32),
            False,
        )
        residual_packed = rocdl.cvt_pk_bf8_f32(
            T.i32,
            _raw(residual[2]),
            _raw(residual[3]),
            residual_packed,
            True,
        )
        return packed, residual_packed

    tid = fx.Int32(gpu.thread_id("x"))
    wave = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    head = lane % 16
    lane_group = lane // 16
    worker = gpu.block_idx.x

    kv_row_base = (wave % 4) * 4 + (lane // 32) * 16 + (lane % 32) // 8
    kv_col_base = (wave // 4) * 32 + (lane % 8) * 4

    def _kv_offset(row, col):
        row_pass = row // 32
        row_in_pass = row % 32
        upper = row_in_pass // 16
        row_in_16 = row_in_pass % 16
        row_group = row_in_16 // 4
        row_in_4 = row_in_16 % 4
        block = col // KV_COL_BLOCK
        col_in_block = col % KV_COL_BLOCK
        col_strip = col_in_block // KV_SUB_COLS
        col_in_strip = col_in_block % KV_SUB_COLS
        slot = col_strip * 4 + row_group
        return (
            block * KV_BLOCK_BYTES
            + (row_pass * KV_SLOTS_PER_PASS + slot) * KV_SLOT_BYTES
            + upper * KV_SUB_ROWS * KV_SUB_COLS
            + row_in_4 * KV_SUB_COLS
            + col_in_strip
        )

    def _load_q_operand(chunk):
        words = []
        for half in range_constexpr(2):
            for word in range_constexpr(4):
                q_offset = chunk + half * 64 + lane_group * 16 + word * 4
                if const_expr(chunk == 512 and half == 1):
                    words.append(_raw(zero_i32))
                else:
                    value = zero_i32
                    if head < NUM_HEADS:
                        value = fx.Int32(_load_i32(Q_OFFSET + head * QK_DIM + q_offset))
                    words.append(_raw(value))
        return _raw(Vec.from_elements(words, fx.Int32))

    def _load_k_operand(kv_base, key_base, chunk):
        words = []
        key_row = key_base + head
        for half in range_constexpr(2):
            for word in range_constexpr(4):
                col = chunk + half * 64 + lane_group * 16 + word * 4
                if const_expr(chunk == 512 and half == 1):
                    words.append(_raw(zero_i32))
                else:
                    words.append(_load_i32(kv_base + _kv_offset(key_row, col)))
        return _raw(Vec.from_elements(words, fx.Int32))

    def _load_p_operand(base):
        words = []
        for half in range_constexpr(2):
            for word in range_constexpr(4):
                value = zero_i32
                if head < NUM_HEADS:
                    value = fx.Int32(
                        _load_i32(
                            base
                            + head * BLOCK_N
                            + half * 64
                            + lane_group * 16
                            + word * 4
                        )
                    )
                words.append(_raw(value))
        return _raw(Vec.from_elements(words, fx.Int32))

    def _load_v_operand(kv_base, fragment):
        words = []
        lane_in_group = lane % 16
        for read_index in range_constexpr(4):
            key_base = 16 * lane_group + 8 * (read_index % 2) + 64 * (read_index // 2)
            source_row = key_base + lane_in_group // 2
            source_column = 64 * wave + 16 * fragment + 8 * (lane_in_group % 2)
            source = fx.make_view(
                _lds_ptr(kv_base + _kv_offset(source_row, source_column)),
                fx.make_layout(8, 1),
            )
            loaded = Vec(
                fly.copy_atom_call_ssa(
                    [Vec.make_type(2, fx.Int32)], transpose_v_copy, source
                )
            )
            words.extend([loaded[0], loaded[1]])
        return _raw(Vec.from_elements(words, fx.Int32))

    def _resolve_kv_rows(tile_start, kv_end):
        physical_rows = []
        for pass_index in range_constexpr(KV_PASSES):
            logical_row = tile_start + pass_index * 32 + kv_row_base
            valid = logical_row < kv_end
            physical_row = fx.Int32(-1)
            if valid:
                physical_row = fx.Int32(
                    buffer_ops.buffer_load(
                        indices_rsrc, logical_row, vec_width=1, dtype=T.i32
                    )
                )
            physical_rows.append(physical_row)
        return physical_rows

    def _load_kv_blocks(kv_base, physical_rows, block_begin, block_end):
        for pass_index in range_constexpr(KV_PASSES):
            physical_row = fx.Int32(physical_rows[pass_index])
            valid = physical_row >= 0
            for block_index in range_constexpr(block_begin, block_end):
                source_byte = (
                    physical_row * QK_DIM + block_index * KV_COL_BLOCK + kv_col_base
                )
                target_base = (
                    kv_base
                    + (pass_index * KV_SLOTS_PER_PASS + wave) * KV_SLOT_BYTES
                    + block_index * KV_BLOCK_BYTES
                )
                if valid:
                    lds_base = rocdl.readfirstlane(T.i32, _raw(fx.Int32(target_base)))
                    asm_string = (
                        "s_mov_b32 m0, $0\n"
                        "s_nop 0\n"
                        "buffer_load_dword $1, $2, 0 offen lds"
                    )
                    _inline_asm_void(
                        [lds_base, _raw(fx.Int32(source_byte)), _raw(kv_rsrc)],
                        asm_string,
                        "s,v,s",
                    )
                else:
                    _store_i32(target_base + lane * 4, zero_i32)

    work_start = fx.Int32(worker)
    work_end = fx.Int32(
        buffer_ops.buffer_load(work_indptr_rsrc, 0, vec_width=1, dtype=T.i32)
    )
    work_step = fx.Int32(num_workers)

    for work_index in range(work_start, work_end, work_step):
        work_base = work_index * 8
        info_lo = Vec(
            buffer_ops.buffer_load(work_rsrc, work_base, vec_width=4, dtype=T.i32)
        )
        info_hi = Vec(
            buffer_ops.buffer_load(work_rsrc, work_base + 4, vec_width=4, dtype=T.i32)
        )
        partial_location = fx.Int32(rocdl.readfirstlane(T.i32, info_lo[1]))
        query_row = fx.Int32(rocdl.readfirstlane(T.i32, info_lo[2]))
        kv_start = fx.Int32(rocdl.readfirstlane(T.i32, info_hi[0]))
        kv_end = fx.Int32(rocdl.readfirstlane(T.i32, info_hi[1]))

        for q_iter in range_constexpr((Q_DWORDS + NUM_THREADS - 1) // NUM_THREADS):
            q_dword = tid + q_iter * NUM_THREADS
            if q_dword < Q_DWORDS:
                value = buffer_ops.buffer_load(
                    query_rsrc,
                    query_row * Q_DWORDS + q_dword,
                    vec_width=1,
                    dtype=T.i32,
                )
                _store_i32(Q_OFFSET + q_dword * 4, value)
        gpu.barrier()

        kv0_rows = _resolve_kv_rows(kv_start, kv_end)
        _load_kv_blocks(fx.Int32(KV0_OFFSET), kv0_rows, 0, KV_COL_BLOCKS)
        rocdl.asyncmark()
        fx.rocdl.s_waitcnt(vmcnt=0)
        rocdl.wait_asyncmark(0)
        gpu.barrier()

        current_scores = Vec.filled(4, 0.0, fx.Float32)
        first_next_start = kv_start + BLOCK_N
        has_first_next = first_next_start < kv_end
        first_next_rows = _resolve_kv_rows(first_next_start, kv_end)
        for chunk_index in range_constexpr(5):
            chunk = chunk_index * 128
            current_scores = Vec(
                _mfma_fp8(
                    _load_k_operand(fx.Int32(KV0_OFFSET), wave * 16, chunk),
                    _load_q_operand(chunk),
                    current_scores,
                )
            )
            if const_expr(chunk_index == 0):
                if has_first_next:
                    _load_kv_blocks(fx.Int32(KV1_OFFSET), first_next_rows, 0, 3)
                    rocdl.asyncmark()
            elif const_expr(chunk_index == 2):
                if has_first_next:
                    _load_kv_blocks(fx.Int32(KV1_OFFSET), first_next_rows, 3, 6)
                    rocdl.asyncmark()

        current_v = []
        for fragment in range_constexpr(4):
            current_v.append(_load_v_operand(fx.Int32(KV0_OFFSET), fragment))
        if has_first_next:
            _load_kv_blocks(fx.Int32(KV1_OFFSET), first_next_rows, 6, KV_COL_BLOCKS)
            rocdl.asyncmark()

        current_local_max = neg_inf
        for item in range_constexpr(4):
            key = kv_start + wave * 16 + lane_group * 4 + item
            score = fx.Float32(current_scores[item]) * qk_scale
            score = (key < kv_end).select(score, neg_inf)
            current_local_max = fx.maxnumf(current_local_max, score)
        current_local_max = _reduce_max_16(current_local_max)
        if lane_group == 0:
            if head < NUM_HEADS:
                _store_f32(
                    STATS_OFFSET + (wave * NUM_HEADS + head) * 4,
                    current_local_max,
                )

        initial_state = [neg_inf, zero_f32]
        initial_state += [Vec.filled(4, 0.0, fx.Float32)] * 4
        initial_state += [current_scores]
        initial_state += current_v

        for tile_start, state in range(
            kv_start, kv_end, fx.Int32(BLOCK_N), init=initial_state
        ):
            row_max = fx.Float32(state[0])
            row_sum = fx.Float32(state[1])
            output_acc = [Vec(state[2 + index]) for index in range_constexpr(4)]
            scores = Vec(state[6])
            v_operands = [Vec(state[7 + index]) for index in range_constexpr(4)]
            tile_start_i32 = fx.Int32(tile_start)
            tile_number = (tile_start_i32 - kv_start) // BLOCK_N
            kv_base = (tile_number & 1).select(
                fx.Int32(KV1_OFFSET), fx.Int32(KV0_OFFSET)
            )
            next_kv_base = (tile_number & 1).select(
                fx.Int32(KV0_OFFSET), fx.Int32(KV1_OFFSET)
            )
            next_tile_start = tile_start_i32 + BLOCK_N
            has_next = next_tile_start < kv_end

            if has_next:
                fx.rocdl.s_waitcnt(vmcnt=0)
                rocdl.wait_asyncmark(0)
            gpu.barrier()

            prefetch_start = next_tile_start + BLOCK_N
            has_prefetch = prefetch_start < kv_end
            prefetch_rows = _resolve_kv_rows(prefetch_start, kv_end)
            if has_prefetch:
                _load_kv_blocks(kv_base, prefetch_rows, 0, 3)
                rocdl.asyncmark()

            next_scores = Vec.filled(4, 0.0, fx.Float32)
            if has_next:
                next_scores = Vec(
                    _mfma_fp8(
                        _load_k_operand(next_kv_base, wave * 16, 0),
                        _load_q_operand(0),
                        next_scores,
                    )
                )

            tile_max = neg_inf
            if head < NUM_HEADS:
                for source_wave in range_constexpr(NUM_WARPS):
                    tile_max = fx.maxnumf(
                        tile_max,
                        _load_f32(STATS_OFFSET + (source_wave * NUM_HEADS + head) * 4),
                    )
            if has_next:
                next_scores = Vec(
                    _mfma_fp8(
                        _load_k_operand(next_kv_base, wave * 16, 128),
                        _load_q_operand(128),
                        next_scores,
                    )
                )
            next_max = fx.maxnumf(row_max, tile_max)
            rescale = fx.Float32(rocdl.exp2(T.f32, _raw((row_max - next_max) * log2e)))
            if has_prefetch:
                _load_kv_blocks(kv_base, prefetch_rows, 3, 6)
                rocdl.asyncmark()
            if has_next:
                next_scores = Vec(
                    _mfma_fp8(
                        _load_k_operand(next_kv_base, wave * 16, 256),
                        _load_q_operand(256),
                        next_scores,
                    )
                )

            probabilities = []
            local_sum = zero_f32
            for item in range_constexpr(4):
                key = tile_start_i32 + wave * 16 + lane_group * 4 + item
                score = fx.Float32(scores[item]) * qk_scale
                score = (key < kv_end).select(score, neg_inf)
                probability = fx.Float32(
                    rocdl.exp2(T.f32, _raw((score - next_max) * log2e))
                )
                probability = (key < kv_end).select(probability, zero_f32)
                probability = (head < NUM_HEADS).select(probability, zero_f32)
                probabilities.append(probability)
                local_sum = local_sum + probability
            if has_next:
                next_scores = Vec(
                    _mfma_fp8(
                        _load_k_operand(next_kv_base, wave * 16, 384),
                        _load_q_operand(384),
                        next_scores,
                    )
                )
            local_sum = _reduce_sum_16(local_sum)
            if has_next:
                next_scores = Vec(
                    _mfma_fp8(
                        _load_k_operand(next_kv_base, wave * 16, 512),
                        _load_q_operand(512),
                        next_scores,
                    )
                )
            if lane_group == 0:
                if head < NUM_HEADS:
                    _store_f32(
                        STATS_OFFSET
                        + (NUM_WARPS * NUM_HEADS + wave * NUM_HEADS + head) * 4,
                        local_sum,
                    )
            if head < NUM_HEADS:
                packed_probability, packed_residual = _pack_probability4(probabilities)
                _store_i32(
                    P_OFFSET + head * BLOCK_N + wave * 16 + lane_group * 4,
                    packed_probability,
                )
                _store_i32(
                    P_RESIDUAL_OFFSET + head * BLOCK_N + wave * 16 + lane_group * 4,
                    packed_residual,
                )
            gpu.barrier()

            tile_sum = zero_f32
            if head < NUM_HEADS:
                for source_wave in range_constexpr(NUM_WARPS):
                    tile_sum = tile_sum + _load_f32(
                        STATS_OFFSET
                        + (NUM_WARPS * NUM_HEADS + source_wave * NUM_HEADS + head) * 4
                    )
            next_sum = row_sum * rescale + tile_sum
            p_operand = _load_p_operand(P_OFFSET)
            p_residual_operand = _load_p_operand(P_RESIDUAL_OFFSET)

            next_output = []
            next_v = []
            next_local_max = neg_inf
            for fragment in range_constexpr(4):
                scaled_acc = Vec(output_acc[fragment]) * _raw(
                    Vec.filled(4, rescale, fx.Float32)
                )
                v_operand = v_operands[fragment]
                high_acc = _mfma_fp8(v_operand, p_operand, scaled_acc)
                next_output.append(
                    Vec(_mfma_fp8_bf8(v_operand, p_residual_operand, high_acc))
                )
                next_score = neg_inf
                next_v_operand = _raw(Vec.filled(8, 0, fx.Int32))
                if has_next:
                    next_v_operand = _load_v_operand(next_kv_base, fragment)
                    next_key = next_tile_start + wave * 16 + lane_group * 4 + fragment
                    next_score = fx.Float32(next_scores[fragment]) * qk_scale
                    next_score = (next_key < kv_end).select(next_score, neg_inf)
                next_v.append(next_v_operand)
                next_local_max = fx.maxnumf(next_local_max, next_score)
                if const_expr(fragment == 1):
                    if has_prefetch:
                        _load_kv_blocks(kv_base, prefetch_rows, 6, KV_COL_BLOCKS)
                        rocdl.asyncmark()
            next_local_max = _reduce_max_16(next_local_max)
            if has_next:
                if lane_group == 0:
                    if head < NUM_HEADS:
                        _store_f32(
                            STATS_OFFSET + (wave * NUM_HEADS + head) * 4,
                            next_local_max,
                        )
            results = yield (
                [next_max, next_sum] + next_output + [next_scores] + next_v
            )

        final_max = fx.Float32(results[0])
        final_sum = fx.Float32(results[1])
        reciprocal = fx.Float32(rocdl.rcp(T.f32, _raw(final_sum)))
        for fragment in range_constexpr(4):
            values = Vec(results[2 + fragment])
            value_base = wave * 64 + fragment * 16 + lane_group * 4
            if head < NUM_HEADS:
                for item in range_constexpr(4):
                    result = (
                        fx.Float32(values[item]) * reciprocal * probability_scale_recip
                    )
                    value_column = value_base + item
                    if const_expr(output_mode == 0):
                        output_offset = (
                            query_row * NUM_HEADS + head
                        ) * V_DIM + value_column
                        buffer_ops.buffer_store(
                            fx.BFloat16(result * kv_scale_value),
                            final_output_rsrc,
                            output_offset,
                        )
                    else:
                        output_offset = fx.Int32(0)
                        if partial_location < 0:
                            output_offset = fx.Int32(
                                (query_row * NUM_HEADS + head) * V_DIM + value_column
                            )
                            buffer_ops.buffer_store(
                                fx.BFloat16(result * kv_scale_value),
                                final_output_rsrc,
                                output_offset,
                            )
                        else:
                            output_offset = fx.Int32(
                                (partial_location * NUM_HEADS + head) * V_DIM
                                + value_column
                            )
                            buffer_ops.buffer_store(
                                result, split_output_rsrc, output_offset
                            )

        if const_expr(output_mode != 0):
            if (wave == 0) & (lane_group == 0) & (head < NUM_HEADS):
                write_lse = partial_location >= 0
                if write_lse:
                    lse = final_max + fx.log2(final_sum) * (1.0 / 1.4426950408889634)
                    buffer_ops.buffer_store(
                        lse,
                        lse_rsrc,
                        partial_location * NUM_HEADS + head,
                    )


@flyc.jit
def launch_mla_decode_varlen_h8(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    kv_page_indices: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    q_scale: fx.Tensor,
    kv_scale: fx.Tensor,
    softmax_scale: fx.Float32,
    output_mode: fx.Constexpr,
    num_workers: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    kn_mla_decode_varlen_h8_fp8(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        q_scale,
        kv_scale,
        softmax_scale,
        output_mode,
        num_workers,
    ).launch(
        grid=(num_workers, 1, 1),
        block=(NUM_THREADS, 1, 1),
        smem=0,
        stream=stream,
    )


@flyc.kernel(known_block_size=[256, 1, 1])
def kn_build_compact_varlen_work_h8(
    query_start_loc: fx.Tensor,
    kv_indptr: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    row_splits: fx.Tensor,
    finalize_count: fx.Tensor,
    finalize_jobs: fx.Tensor,
    num_requests: fx.Int32,
    max_rows: fx.Constexpr,
    max_splits: fx.Constexpr,
    explicit_splits: fx.Constexpr,
):
    tid = fx.Int32(gpu.thread_id("x"))
    qsl_rsrc = buffer_ops.create_buffer_resource(query_start_loc)
    indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr)
    header_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    info_rsrc = buffer_ops.create_buffer_resource(work_info_set)
    splits_rsrc = buffer_ops.create_buffer_resource(row_splits)
    finalize_count_rsrc = buffer_ops.create_buffer_resource(finalize_count)
    finalize_jobs_rsrc = buffer_ops.create_buffer_resource(finalize_jobs)
    live_rows = fx.Int32(
        buffer_ops.buffer_load(qsl_rsrc, num_requests, vec_width=1, dtype=T.i32)
    )
    if tid == 0:
        buffer_ops.buffer_store(fx.Int32(0), header_rsrc, 0)
        buffer_ops.buffer_store(live_rows, header_rsrc, 1)
        buffer_ops.buffer_store(fx.Int32(0), finalize_count_rsrc, 0)
    fx.rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()

    if const_expr(explicit_splits == 0):
        for row in range(tid, live_rows, fx.Int32(256)):
            row_i32 = fx.Int32(row)
            row_start = fx.Int32(
                buffer_ops.buffer_load(indptr_rsrc, row_i32, vec_width=1, dtype=T.i32)
            )
            row_end = fx.Int32(
                buffer_ops.buffer_load(
                    indptr_rsrc, row_i32 + fx.Int32(1), vec_width=1, dtype=T.i32
                )
            )
            tiles = (row_end - row_start + BLOCK_N - 1) // BLOCK_N
            if tiles > 0:
                atomic_add_i32(finalize_count, 1, 0, "agent")
        fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        live_nonempty = fx.Int32(
            buffer_ops.buffer_load(finalize_count_rsrc, 0, vec_width=1, dtype=T.i32)
        )
        fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            buffer_ops.buffer_store(fx.Int32(0), finalize_count_rsrc, 0)
        fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
    else:
        live_nonempty = live_rows

    for row in range(tid, live_rows, fx.Int32(256)):
        row_i32 = fx.Int32(row)
        row_start = fx.Int32(
            buffer_ops.buffer_load(indptr_rsrc, row_i32, vec_width=1, dtype=T.i32)
        )
        row_end = fx.Int32(
            buffer_ops.buffer_load(
                indptr_rsrc, row_i32 + fx.Int32(1), vec_width=1, dtype=T.i32
            )
        )
        tiles = (row_end - row_start + BLOCK_N - 1) // BLOCK_N
        if const_expr(explicit_splits > 0):
            desired_splits = fx.Int32(explicit_splits)
        else:
            desired_splits = fx.Int32(2)
            desired_splits = (live_nonempty <= 64).select(fx.Int32(4), desired_splits)
            desired_splits = (live_nonempty <= 32).select(fx.Int32(8), desired_splits)
            desired_splits = (live_nonempty <= 10).select(fx.Int32(16), desired_splits)
        desired_splits = (desired_splits < max_splits).select(
            desired_splits, fx.Int32(max_splits)
        )
        chunk_tiles = (tiles + desired_splits - 1) // desired_splits
        chunk_tiles = (chunk_tiles > 0).select(chunk_tiles, fx.Int32(1))
        actual_splits = (tiles + chunk_tiles - 1) // chunk_tiles
        buffer_ops.buffer_store(actual_splits, splits_rsrc, row_i32)

        if const_expr(explicit_splits != 1):
            if actual_splits != 1:
                finalize_job_count = atomic_add_i32(
                    finalize_count, NUM_HEADS, 0, "agent"
                )
                for finalize_head in range_constexpr(NUM_HEADS):
                    finalize_base = (finalize_job_count + finalize_head) * 2
                    buffer_ops.buffer_store(row_i32, finalize_jobs_rsrc, finalize_base)
                    buffer_ops.buffer_store(
                        fx.Int32(finalize_head), finalize_jobs_rsrc, finalize_base + 1
                    )

        task_count = atomic_add_i32(work_indptr, actual_splits, 0, "agent")
        for split in range_constexpr(max_splits):
            if split < actual_splits:
                task_start = row_start + split * chunk_tiles * BLOCK_N
                unclamped_end = task_start + chunk_tiles * BLOCK_N
                task_end = (unclamped_end < row_end).select(unclamped_end, row_end)
                partial_location = fx.Int32(
                    (actual_splits == 1).select(
                        fx.Int32(-1), fx.Int32(row * max_splits + split)
                    )
                )
                base = (task_count + split) * 8
                buffer_ops.buffer_store(fx.Int32(0), info_rsrc, base)
                buffer_ops.buffer_store(partial_location, info_rsrc, base + 1)
                buffer_ops.buffer_store(row_i32, info_rsrc, base + 2)
                buffer_ops.buffer_store(row_i32 + fx.Int32(1), info_rsrc, base + 3)
                buffer_ops.buffer_store(task_start, info_rsrc, base + 4)
                buffer_ops.buffer_store(task_end, info_rsrc, base + 5)
                buffer_ops.buffer_store(fx.Int32(0), info_rsrc, base + 6)
                buffer_ops.buffer_store(fx.Int32(0), info_rsrc, base + 7)


@flyc.jit
def launch_build_compact_varlen_work_h8(
    query_start_loc: fx.Tensor,
    kv_indptr: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    row_splits: fx.Tensor,
    finalize_count: fx.Tensor,
    finalize_jobs: fx.Tensor,
    num_requests: fx.Int32,
    max_rows: fx.Constexpr,
    max_splits: fx.Constexpr,
    explicit_splits: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    kn_build_compact_varlen_work_h8(
        query_start_loc,
        kv_indptr,
        work_indptr,
        work_info_set,
        row_splits,
        finalize_count,
        finalize_jobs,
        num_requests,
        max_rows,
        max_splits,
        explicit_splits,
    ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[256, 1, 1])
def kn_finalize_varlen_h8(
    finalize_count: fx.Tensor,
    finalize_jobs: fx.Tensor,
    row_splits: fx.Tensor,
    partial_output: fx.Tensor,
    partial_lse: fx.Tensor,
    kv_scale: fx.Tensor,
    final_output: fx.Tensor,
    max_splits: fx.Constexpr,
    num_workers: fx.Constexpr,
):
    worker = gpu.block_idx.x
    tid = fx.Int32(gpu.thread_id("x"))
    wave = tid // WARP_SIZE
    lane = tid % WARP_SIZE
    count_rsrc = buffer_ops.create_buffer_resource(finalize_count)
    jobs_rsrc = buffer_ops.create_buffer_resource(finalize_jobs)
    splits_rsrc = buffer_ops.create_buffer_resource(row_splits)
    partial_rsrc = buffer_ops.create_buffer_resource(partial_output)
    lse_rsrc = buffer_ops.create_buffer_resource(partial_lse)
    scale_rsrc = buffer_ops.create_buffer_resource(kv_scale)
    output_rsrc = buffer_ops.create_buffer_resource(final_output)
    job_count = fx.Int32(
        buffer_ops.buffer_load(count_rsrc, 0, vec_width=1, dtype=T.i32)
    )
    scale = fx.Float32(buffer_ops.buffer_load(scale_rsrc, 0, vec_width=1, dtype=T.f32))
    zero_f32 = fx.Float32(0.0)

    job_start = fx.Int32(worker * 4 + wave)
    job_step = fx.Int32(num_workers * 4)
    for job in range(job_start, job_count, job_step):
        job_info = Vec(
            buffer_ops.buffer_load(jobs_rsrc, job * 2, vec_width=2, dtype=T.i32)
        )
        row_i32 = fx.Int32(job_info[0])
        head = fx.Int32(job_info[1])
        actual_splits = fx.Int32(
            buffer_ops.buffer_load(splits_rsrc, row_i32, vec_width=1, dtype=T.i32)
        )
        output_base = (row_i32 * NUM_HEADS + head) * V_DIM
        if actual_splits == 0:
            zero_bf16 = Vec.filled(4, 0.0, fx.BFloat16)
            value_base = lane * 8
            for value_iter in range_constexpr(2):
                buffer_ops.buffer_store(
                    zero_bf16,
                    output_rsrc,
                    output_base + value_base + value_iter * 4,
                )
        elif actual_splits > 1:
            lane_lse = fx.Float32(-1.0e30)
            if lane < actual_splits:
                partial_location = row_i32 * max_splits + lane
                lane_lse = fx.Float32(
                    buffer_ops.buffer_load(
                        lse_rsrc,
                        partial_location * NUM_HEADS + head,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
            maximum = lane_lse
            for offset in (32, 16, 8, 4, 2, 1):
                maximum = fx.maxnumf(
                    maximum, fx.Float32(maximum).shuffle_xor(offset, WARP_SIZE)
                )
            lane_weight = zero_f32
            if lane < actual_splits:
                lane_weight = fx.Float32(
                    rocdl.exp2(T.f32, _raw((lane_lse - maximum) * 1.4426950408889634))
                )
            denominator = lane_weight
            for offset in (32, 16, 8, 4, 2, 1):
                denominator = denominator + fx.Float32(denominator).shuffle_xor(
                    offset, WARP_SIZE
                )
            value_base = lane * 8
            accumulator0 = Vec.filled(4, 0.0, fx.Float32)
            accumulator1 = Vec.filled(4, 0.0, fx.Float32)
            for split in range_constexpr(max_splits):
                if split < actual_splits:
                    weight = fx.Float32(
                        rocdl.readlane(T.f32, _raw(lane_weight), _raw(fx.Int32(split)))
                    )
                    partial_location = row_i32 * max_splits + split
                    partial_base = (
                        partial_location * NUM_HEADS + head
                    ) * V_DIM + value_base
                    values0 = Vec(
                        buffer_ops.buffer_load(
                            partial_rsrc,
                            partial_base,
                            vec_width=4,
                            dtype=T.f32,
                        )
                    )
                    values1 = Vec(
                        buffer_ops.buffer_load(
                            partial_rsrc,
                            partial_base + 4,
                            vec_width=4,
                            dtype=T.f32,
                        )
                    )
                    weight4 = Vec.filled(4, weight, fx.Float32)
                    accumulator0 = accumulator0 + values0 * _raw(weight4)
                    accumulator1 = accumulator1 + values1 * _raw(weight4)
            output_scale = Vec.filled(4, scale / denominator, fx.Float32)
            buffer_ops.buffer_store(
                (accumulator0 * _raw(output_scale)).to(fx.BFloat16),
                output_rsrc,
                output_base + value_base,
            )
            buffer_ops.buffer_store(
                (accumulator1 * _raw(output_scale)).to(fx.BFloat16),
                output_rsrc,
                output_base + value_base + 4,
            )


@flyc.jit
def launch_finalize_varlen_h8(
    finalize_count: fx.Tensor,
    finalize_jobs: fx.Tensor,
    row_splits: fx.Tensor,
    partial_output: fx.Tensor,
    partial_lse: fx.Tensor,
    kv_scale: fx.Tensor,
    final_output: fx.Tensor,
    max_splits: fx.Constexpr,
    num_workers: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    kn_finalize_varlen_h8(
        finalize_count,
        finalize_jobs,
        row_splits,
        partial_output,
        partial_lse,
        kv_scale,
        final_output,
        max_splits,
        num_workers,
    ).launch(
        grid=(num_workers, 1, 1),
        block=(256, 1, 1),
        stream=stream,
    )


__all__ = [
    "OCCUPANCY",
    "TOTAL_LDS_BYTES",
    "launch_build_compact_varlen_work_h8",
    "launch_finalize_varlen_h8",
    "launch_mla_decode_varlen_h8",
]
