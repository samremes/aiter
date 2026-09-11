# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-launch persistent radix merge over Stage-A local TopK bags."""

from functools import cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, rocdl as mlir_rocdl
from flydsl.expr import (
    Array,
    Float32,
    Int32,
    arith,
    as_ir_value,
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.candidate_topk_common import (
    BLOCK_THREADS,
    NUM_HIST_BINS,
    NUM_RADIX_PASSES,
    NUM_WAVES,
    RADIX_SIGN_BIT,
    block_exclusive_prefix_i32,
    f32_to_ordered_i32,
    prefix_radix_mask,
    radix_bucket,
    radix_pass_bits,
)
from aiter.ops.flydsl.kernels.kernels_common import atomic_add_i32
from aiter.ops.flydsl.kernels.split_topk_merge_layout import (
    COUNTER_ARRIVALS,
    COUNTER_OUT_ABOVE,
    COUNTER_OUT_EQUAL,
    COUNTER_PASS_DONE,
    COUNTER_STATUS,
    HIST0_OFF,
    HIST1_OFF,
    HIST2_OFF,
    ROW_STRIDE,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw, buf_copy_atom

_VEC = 4
_NEG_INF = float("-inf")


def _load_f32x4(tensor, vec_idx):
    src = fx.slice(tensor, (None, vec_idx))
    fragment = fx.make_fragment_like(src)
    fx.copy(buf_copy_atom(16, Float32), src, fragment)
    return fx.Vector(fx.memref_load_vec(fragment))


def _load_i32x4(tensor, vec_idx):
    src = fx.slice(tensor, (None, vec_idx))
    fragment = fx.make_fragment_like(src)
    fx.copy(buf_copy_atom(16, Int32), src, fragment)
    return fx.Vector(fx.memref_load_vec(fragment))


def _global_ptr(memref, offset):
    return fx.to_llvm_ptr(fx.get_iter(memref) + offset)


def _atomic_add_ordered(memref, val, offset, ordering):
    val = fx.Int32(val) if isinstance(val, int) else val
    old = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        _global_ptr(memref, offset),
        as_ir_value(val),
        ordering,
        syncscope="agent",
        alignment=4,
    ).result
    return fx.Int32(old)


def _atomic_xchg_ordered(memref, val, offset, ordering):
    val = fx.Int32(val) if isinstance(val, int) else val
    old = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.xchg,
        _global_ptr(memref, offset),
        as_ir_value(val),
        ordering,
        syncscope="agent",
        alignment=4,
    ).result
    return fx.Int32(old)


def _atomic_load_ordered(memref, offset, ordering):
    return fx.Int32(
        llvm.LoadOp(
            T.i32,
            _global_ptr(memref, offset),
            alignment=4,
            volatile_=True,
            ordering=ordering,
            syncscope="agent",
        ).result
    )


def _make_storage(k):
    @fx.struct
    class PersistentMergeStorage:
        histogram: Array[Int32, NUM_HIST_BINS, 16]
        scan: Array[Int32, NUM_WAVES + 1, 16]
        above_idxs: Array[Int32, k, 16]
        equal_idxs: Array[Int32, k, 16]
        prefix: Array[Int32, 1, 4]
        mask: Array[Int32, 1, 4]
        remaining: Array[Int32, 1, 4]
        total_live: Array[Int32, 1, 4]
        above_need: Array[Int32, 1, 4]
        equal_need: Array[Int32, 1, 4]
        above_count: Array[Int32, 1, 4]
        equal_count: Array[Int32, 1, 4]
        above_base: Array[Int32, 1, 4]
        equal_base: Array[Int32, 1, 4]

    return PersistentMergeStorage


@cache
def build_split_topk_merge_persistent(k: int, splits: int, parts: int):
    if k % _VEC:
        raise ValueError(f"k must be divisible by {_VEC}, got {k}")
    if parts < 2:
        raise ValueError(f"persistent merge requires at least 2 parts, got {parts}")
    if splits < 2:
        raise ValueError(f"persistent merge requires at least 2 splits, got {splits}")
    if splits % parts != 0:
        raise ValueError(
            f"persistent merge requires splits % parts == 0, got {splits=} {parts=}"
        )
    bags_per_part = splits // parts
    split_vectors = k // _VEC
    tiles_per_bag = (split_vectors + BLOCK_THREADS - 1) // BLOCK_THREADS
    output_steps = (k + BLOCK_THREADS - 1) // BLOCK_THREADS
    hist_steps = NUM_HIST_BINS // BLOCK_THREADS
    hist0_bins_per_part = (NUM_HIST_BINS + parts - 1) // parts
    hist0_zero_steps = (hist0_bins_per_part + BLOCK_THREADS - 1) // BLOCK_THREADS
    storage_type = _make_storage(k)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        scores: fx.Tensor,
        positions: fx.Tensor,
        counts: fx.Tensor,
        out_scores: fx.Tensor,
        out_positions: fx.Tensor,
        workspace: fx.Tensor,
    ):
        part = fx.Int32(fx.block_idx.x)
        row = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        row_base = row * fx.Int32(ROW_STRIDE)
        parts_i32 = fx.Int32(parts)
        k_i32 = fx.Int32(k)
        splits_i32 = fx.Int32(splits)

        score_buffer = fx.rocdl.make_buffer_tensor(scores, max_size=False)
        workspace_buffer = fx.rocdl.make_buffer_tensor(workspace, max_size=False)
        score_rsrc = fx.logical_divide(
            fx.slice(score_buffer, (row, None)),
            fx.make_layout(_VEC, 1),
        )
        workspace_rsrc = fx.logical_divide(
            workspace_buffer,
            fx.make_layout(_VEC, 1),
        )
        score_row = fx.slice(scores, (row, None))
        pos_row = fx.slice(positions, (row, None))
        out_score_row = fx.slice(out_scores, (row, None))
        out_pos_row = fx.slice(out_positions, (row, None))

        storage = fx.SharedAllocator().allocate(storage_type)
        histogram = storage.histogram.peek().view(fx.make_layout(NUM_HIST_BINS, 1))
        scan = storage.scan.peek().view(fx.make_layout(NUM_WAVES + 1, 1))
        s_above_idxs = storage.above_idxs.peek().view(fx.make_layout(k, 1))
        s_equal_idxs = storage.equal_idxs.peek().view(fx.make_layout(k, 1))
        s_prefix = storage.prefix.peek().view(fx.make_layout(1, 1))
        s_mask = storage.mask.peek().view(fx.make_layout(1, 1))
        s_remaining = storage.remaining.peek().view(fx.make_layout(1, 1))
        s_total = storage.total_live.peek().view(fx.make_layout(1, 1))
        s_above_need = storage.above_need.peek().view(fx.make_layout(1, 1))
        s_equal_need = storage.equal_need.peek().view(fx.make_layout(1, 1))
        s_above_count = storage.above_count.peek().view(fx.make_layout(1, 1))
        s_equal_count = storage.equal_count.peek().view(fx.make_layout(1, 1))
        s_above_base = storage.above_base.peek().view(fx.make_layout(1, 1))
        s_equal_base = storage.equal_base.peek().view(fx.make_layout(1, 1))

        if tid == 0:
            live_total = fx.Int32(0)
            invalid_count = fx.Int32(0)
            for split_idx in range_constexpr(splits):
                raw_count = counts[row, split_idx]
                too_low = raw_count < 0
                too_high = raw_count > k_i32
                invalid_count = invalid_count + (too_low | too_high).select(
                    fx.Int32(1),
                    fx.Int32(0),
                )
                clamped = too_low.select(fx.Int32(0), raw_count)
                clamped = too_high.select(k_i32, clamped)
                live_total = live_total + clamped
            selected_n = (live_total < k_i32).select(live_total, k_i32)
            s_total[0] = selected_n
            if part == 0:
                workspace[row_base + fx.Int32(COUNTER_STATUS)] = invalid_count
        gpu.barrier()
        selected_n = s_total[0]

        if part == 0:
            for step in range_constexpr(output_steps):
                slot = fx.Int32(step * BLOCK_THREADS) + tid
                pad = (slot >= selected_n) & (slot < k_i32)
                if pad:
                    out_score_row[slot] = fx.Float32(_NEG_INF)
                    out_pos_row[slot] = fx.Int32(-1)

        @flyc.jit
        def _zero_hist0_empty(
            _pred=(selected_n == 0) & (part == 0),
            _workspace=workspace,
            _row_base=row_base,
            _tid=tid,
        ):
            if _pred:
                for step in range_constexpr(hist_steps):
                    hist_bin = fx.Int32(step * BLOCK_THREADS) + _tid
                    _workspace[_row_base + fx.Int32(HIST0_OFF) + hist_bin] = 0

        _zero_hist0_empty()

        def spin_until_ge(offset, target):
            cur = _atomic_load_ordered(
                workspace,
                offset,
                llvm.AtomicOrdering.monotonic,
            )
            while cur < target:
                mlir_rocdl.s_sleep(1)
                cur = _atomic_load_ordered(
                    workspace,
                    offset,
                    llvm.AtomicOrdering.monotonic,
                )
            _atomic_load_ordered(
                workspace,
                offset,
                llvm.AtomicOrdering.acquire,
            )

        def row_barrier(token):
            gpu.barrier()

            @flyc.jit
            def _arrive(
                _pred=tid == 0,
                _token=fx.Int32(token),
                _workspace=workspace,
                _row_base=row_base,
                _parts=parts_i32,
            ):
                if _pred:
                    arrivals = _row_base + fx.Int32(COUNTER_ARRIVALS)
                    done = _row_base + fx.Int32(COUNTER_PASS_DONE)
                    prev = _atomic_add_ordered(
                        _workspace,
                        1,
                        arrivals,
                        llvm.AtomicOrdering.acq_rel,
                    )
                    last = (prev + fx.Int32(1)) == _token * _parts
                    if last:
                        _atomic_xchg_ordered(
                            _workspace,
                            _token,
                            done,
                            llvm.AtomicOrdering.release,
                        )
                    else:
                        spin_until_ge(done, _token)

            _arrive()
            gpu.barrier()

        def clear_histogram(num_bins):
            steps = num_bins // BLOCK_THREADS
            for step in range_constexpr(steps):
                histogram[fx.Int32(step * BLOCK_THREADS) + tid] = 0
            gpu.barrier()

        def copy_hist_from_workspace(hist_off, num_bins):
            vec_steps = num_bins // (BLOCK_THREADS * _VEC)
            hist_vec_base = (row_base + fx.Int32(hist_off)) // fx.Int32(_VEC)
            for step in range_constexpr(vec_steps):
                vec_idx = hist_vec_base + tid + fx.Int32(step * BLOCK_THREADS)
                vals = _load_i32x4(workspace_rsrc, vec_idx)
                bin_base = (tid + fx.Int32(step * BLOCK_THREADS)) * fx.Int32(_VEC)
                for vi in range_constexpr(_VEC):
                    histogram[bin_base + fx.Int32(vi)] = vals[vi]
            gpu.barrier()

        def flush_histogram(hist_off, num_bins):
            steps = num_bins // BLOCK_THREADS
            for step in range_constexpr(steps):
                hist_bin = fx.Int32(step * BLOCK_THREADS) + tid
                count = histogram[hist_bin]

                @flyc.jit
                def _flush(
                    _pred=(count != 0) & (hist_bin < fx.Int32(num_bins)),
                    _workspace=workspace,
                    _offset=row_base + fx.Int32(hist_off) + hist_bin,
                    _count=count,
                ):
                    if _pred:
                        atomic_add_i32(_workspace, _count, _offset, "agent")

                _flush()
            gpu.barrier()

        def select_from_histogram(pass_index, prefix, mask, remaining):
            num_bins = 1 << radix_pass_bits(pass_index)
            bins_per_thread = num_bins // BLOCK_THREADS
            xor_value = RADIX_SIGN_BIT if pass_index == 0 else 0
            selected_high = fx.Int32(num_bins - 1) - tid * fx.Int32(bins_per_thread)
            group_count = fx.Int32(0)
            bin_counts = fx.make_rmem_tensor(bins_per_thread, Int32)
            for bin_item in range_constexpr(bins_per_thread):
                value = histogram[selected_high - fx.Int32(bin_item)]
                bin_counts[bin_item] = value
                group_count = group_count + value
            before_group, _ = block_exclusive_prefix_i32(tid, group_count, scan)
            before_bin = before_group
            for bin_item in range_constexpr(bins_per_thread):
                selected = selected_high - fx.Int32(bin_item)
                bin_count = bin_counts[bin_item]

                @flyc.jit
                def _maybe_select(
                    _pred=(before_bin < remaining)
                    & (before_bin + bin_count >= remaining),
                    _selected=selected,
                    _xor=fx.Int32(xor_value),
                    _prefix=prefix,
                    _mask=mask,
                    _remaining=remaining,
                    _before=before_bin,
                    _s_prefix=s_prefix,
                    _s_mask=s_mask,
                    _s_remaining=s_remaining,
                    _pass=pass_index,
                ):
                    if _pred:
                        actual = _selected ^ _xor
                        pass_mask, shift = prefix_radix_mask(_pass)
                        _s_prefix[0] = _prefix | (actual << fx.Int32(shift))
                        _s_mask[0] = _mask | pass_mask
                        _s_remaining[0] = _remaining - _before

                _maybe_select()
                before_bin = before_bin + bin_count
            gpu.barrier()
            return s_prefix[0], s_mask[0], s_remaining[0]

        def count_matching(pass_index, prefix, mask):
            xor_value = RADIX_SIGN_BIT if pass_index == 0 else 0
            start = part * fx.Int32(bags_per_part)
            for bag_off in range_constexpr(bags_per_part):
                split_idx = start + fx.Int32(bag_off)
                active_bag = split_idx < splits_i32
                safe_split = active_bag.select(split_idx, fx.Int32(0))
                raw_count = counts[row, safe_split]
                count_s = (raw_count < 0).select(fx.Int32(0), raw_count)
                count_s = (count_s > k_i32).select(k_i32, count_s)
                count_s = active_bag.select(count_s, fx.Int32(0))
                bag_vec_base = safe_split * fx.Int32(split_vectors)
                for tile in range_constexpr(tiles_per_bag):
                    vec_in_bag = fx.Int32(tile * BLOCK_THREADS) + tid
                    in_tile = vec_in_bag < fx.Int32(split_vectors)
                    vec_idx = bag_vec_base + in_tile.select(vec_in_bag, fx.Int32(0))
                    rvals = _load_f32x4(score_rsrc, vec_idx)
                    base = vec_in_bag * fx.Int32(_VEC)
                    for vi in range_constexpr(_VEC):
                        col = base + fx.Int32(vi)
                        live = active_bag & in_tile & (col < count_s)
                        ords = f32_to_ordered_i32(rvals[vi])
                        matched = live & ((ords & mask) == prefix)

                        @flyc.jit
                        def _count(
                            _pred=matched,
                            _histogram=histogram,
                            _ords=ords,
                            _pass=pass_index,
                            _xor=xor_value,
                        ):
                            if _pred:
                                bucket = radix_bucket(_ords, _pass) ^ fx.Int32(_xor)
                                atomic_add_i32(_histogram, 1, bucket, "workgroup")

                        _count()
            gpu.barrier()

        def gather_owned_bags(threshold, above_need, equal_need, selected_n):
            start = part * fx.Int32(bags_per_part)
            for bag_off in range_constexpr(bags_per_part):
                split_idx = start + fx.Int32(bag_off)
                active_bag = split_idx < splits_i32
                safe_split = active_bag.select(split_idx, fx.Int32(0))
                raw_count = counts[row, safe_split]
                count_s = (raw_count < 0).select(fx.Int32(0), raw_count)
                count_s = (count_s > k_i32).select(k_i32, count_s)
                count_s = active_bag.select(count_s, fx.Int32(0))
                bag_vec_base = safe_split * fx.Int32(split_vectors)
                @flyc.jit
                def _reset_bag_counts(
                    _pred=tid == 0,
                    _s_above_count=s_above_count,
                    _s_equal_count=s_equal_count,
                ):
                    if _pred:
                        _s_above_count[0] = 0
                        _s_equal_count[0] = 0

                _reset_bag_counts()
                gpu.barrier()

                for tile in range_constexpr(tiles_per_bag):
                    vec_in_bag = fx.Int32(tile * BLOCK_THREADS) + tid
                    in_tile = vec_in_bag < fx.Int32(split_vectors)
                    vec_idx = bag_vec_base + in_tile.select(vec_in_bag, fx.Int32(0))
                    rvals = _load_f32x4(score_rsrc, vec_idx)
                    base = vec_in_bag * fx.Int32(_VEC)
                    for vi in range_constexpr(_VEC):
                        col = base + fx.Int32(vi)
                        live = active_bag & in_tile & (col < count_s)
                        ords = f32_to_ordered_i32(rvals[vi])
                        above = live & (ords > threshold)
                        equal = live & (ords == threshold)

                        @flyc.jit
                        def _classify_above(
                            _pred=above,
                            _col=col,
                            _s_count=s_above_count,
                            _s_idxs=s_above_idxs,
                            _k=k_i32,
                        ):
                            if _pred:
                                pos = atomic_add_i32(_s_count, 1, 0, "workgroup")
                                if pos < _k:
                                    _s_idxs[pos] = _col

                        @flyc.jit
                        def _classify_equal(
                            _pred=equal,
                            _col=col,
                            _s_count=s_equal_count,
                            _s_idxs=s_equal_idxs,
                            _k=k_i32,
                        ):
                            if _pred:
                                pos = atomic_add_i32(_s_count, 1, 0, "workgroup")
                                if pos < _k:
                                    _s_idxs[pos] = _col

                        _classify_above()
                        _classify_equal()
                gpu.barrier()

                @flyc.jit
                def _reserve(
                    _pred=(tid == 0) & active_bag,
                    _workspace=workspace,
                    _row_base=row_base,
                    _s_above_count=s_above_count,
                    _s_equal_count=s_equal_count,
                    _s_above_base=s_above_base,
                    _s_equal_base=s_equal_base,
                    _k=k_i32,
                    _above_need=above_need,
                    _equal_need=equal_need,
                ):
                    if _pred:
                        local_above = _s_above_count[0]
                        local_equal = _s_equal_count[0]
                        stored_above = (local_above < _k).select(local_above, _k)
                        stored_equal = (local_equal < _k).select(local_equal, _k)
                        old_above = _atomic_add_ordered(
                            _workspace,
                            stored_above,
                            _row_base + fx.Int32(COUNTER_OUT_ABOVE),
                            llvm.AtomicOrdering.monotonic,
                        )
                        above_room = _above_need - old_above
                        accepted_above = (above_room > 0).select(
                            (stored_above < above_room).select(
                                stored_above, above_room
                            ),
                            fx.Int32(0),
                        )
                        old_equal = _atomic_add_ordered(
                            _workspace,
                            stored_equal,
                            _row_base + fx.Int32(COUNTER_OUT_EQUAL),
                            llvm.AtomicOrdering.monotonic,
                        )
                        equal_room = _equal_need - old_equal
                        accepted_equal = (equal_room > 0).select(
                            (stored_equal < equal_room).select(
                                stored_equal, equal_room
                            ),
                            fx.Int32(0),
                        )
                        _s_above_count[0] = accepted_above
                        _s_equal_count[0] = accepted_equal
                        _s_above_base[0] = old_above
                        _s_equal_base[0] = old_equal

                _reserve()
                gpu.barrier()

                for step in range_constexpr(output_steps):
                    local_pos = fx.Int32(step * BLOCK_THREADS) + tid
                    idx_slot = (local_pos < k_i32).select(local_pos, fx.Int32(0))
                    keep_above = local_pos < s_above_count[0]
                    dest_above = s_above_base[0] + local_pos
                    keep_above = (
                        keep_above
                        & (dest_above >= 0)
                        & (dest_above < selected_n)
                        & (dest_above < k_i32)
                    )
                    keep_equal = local_pos < s_equal_count[0]
                    dest_equal = above_need + s_equal_base[0] + local_pos
                    keep_equal = (
                        keep_equal
                        & (dest_equal >= 0)
                        & (dest_equal < selected_n)
                        & (dest_equal < k_i32)
                    )

                    @flyc.jit
                    def _store_above(
                        _pred=keep_above,
                        _dest=dest_above,
                        _idx=s_above_idxs[idx_slot],
                        _split=safe_split,
                        _k=k_i32,
                        _score_row=score_row,
                        _pos_row=pos_row,
                        _out_s=out_score_row,
                        _out_p=out_pos_row,
                    ):
                        if _pred:
                            flat = _split * _k + _idx
                            _out_s[_dest] = _score_row[flat]
                            _out_p[_dest] = _pos_row[flat]

                    @flyc.jit
                    def _store_equal(
                        _pred=keep_equal,
                        _dest=dest_equal,
                        _idx=s_equal_idxs[idx_slot],
                        _split=safe_split,
                        _k=k_i32,
                        _score_row=score_row,
                        _pos_row=pos_row,
                        _out_s=out_score_row,
                        _out_p=out_pos_row,
                    ):
                        if _pred:
                            flat = _split * _k + _idx
                            _out_s[_dest] = _score_row[flat]
                            _out_p[_dest] = _pos_row[flat]

                    _store_above()
                    _store_equal()
                gpu.barrier()

        if selected_n != 0:
            copy_hist_from_workspace(HIST0_OFF, NUM_HIST_BINS)
            if tid == 0:
                s_prefix[0] = 0
                s_mask[0] = 0
                s_remaining[0] = selected_n
            gpu.barrier()
            prefix, mask, remaining = select_from_histogram(
                0,
                fx.Int32(0),
                fx.Int32(0),
                selected_n,
            )

            hist_offs = (HIST1_OFF, HIST2_OFF)
            for radix_pass in range_constexpr(1, NUM_RADIX_PASSES):
                pass_bins = 1 << radix_pass_bits(radix_pass)
                clear_histogram(pass_bins)
                count_matching(radix_pass, prefix, mask)
                flush_histogram(hist_offs[radix_pass - 1], pass_bins)
                row_barrier(radix_pass)
                if const_expr(radix_pass == 1):
                    hist0_start = part * fx.Int32(hist0_bins_per_part)
                    for step in range_constexpr(hist0_zero_steps):
                        hist_bin = (
                            hist0_start
                            + fx.Int32(step * BLOCK_THREADS)
                            + tid
                        )
                        in_part = hist_bin < (
                            hist0_start + fx.Int32(hist0_bins_per_part)
                        )
                        in_hist = hist_bin < fx.Int32(NUM_HIST_BINS)

                        @flyc.jit
                        def _zero_hist0(
                            _pred=in_part & in_hist,
                            _workspace=workspace,
                            _offset=row_base + fx.Int32(HIST0_OFF) + hist_bin,
                        ):
                            if _pred:
                                _workspace[_offset] = 0

                        _zero_hist0()
                copy_hist_from_workspace(hist_offs[radix_pass - 1], pass_bins)
                prefix, mask, remaining = select_from_histogram(
                    radix_pass,
                    prefix,
                    mask,
                    remaining,
                )

            equal_need = remaining
            above_need = selected_n - remaining
            if tid == 0:
                s_above_need[0] = above_need
                s_equal_need[0] = equal_need
                s_total[0] = selected_n
            gpu.barrier()
            gather_owned_bags(
                prefix,
                s_above_need[0],
                s_equal_need[0],
                s_total[0],
            )

    @flyc.jit
    def launch(
        scores: fx.Tensor,
        positions: fx.Tensor,
        counts: fx.Tensor,
        out_scores: fx.Tensor,
        out_positions: fx.Tensor,
        workspace: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(fx.Int32(parts)))
        gy = arith.index_cast(T.index, _to_raw(rows))
        kernel(
            scores,
            positions,
            counts,
            out_scores,
            out_positions,
            workspace,
        ).launch(
            grid=(gx, gy, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"fast_fp_math": True}
    return launch


def persistent_merge_parts(rows: int, splits: int, cu_count: int, occupancy: int = 1):
    """Largest cooperating part count that stays inside the occupancy envelope.

    ``occupancy`` is resident 256-thread blocks per CU. The shipped default is
    1, and raising it is a measured loss rather than an untested risk: at 2
    WG/CU the 16-row and 32-row cells stayed co-resident and correct but ran
    16-19% slower, because the row barrier's cost scales with the number of
    arriving parts rather than with per-part work. A cooperative launch cannot
    make this safer either -- FlyDSL's runtime has no cooperative entry point,
    so the MLIR ``cooperative`` attribute is accepted and then dropped.
    """
    if rows <= 0 or splits < 2 or cu_count <= 0:
        return 0
    envelope = cu_count * max(1, occupancy)
    max_safe = (envelope - 1) // rows + 1
    parts = min(splits, max_safe)
    while parts >= 2 and splits % parts != 0:
        parts -= 1
    if parts < 2:
        return 0
    if rows * (parts - 1) >= envelope:
        return 0
    return parts


def _validate_persistent_workspace(workspace, rows: int):
    from aiter.ops.flydsl.kernels.split_topk_merge_layout import (
        persistent_workspace_elems,
    )

    expected = persistent_workspace_elems(rows)
    if workspace.dtype != torch.int32:
        raise TypeError("persistent workspace must be int32")
    if not workspace.is_cuda:
        raise ValueError("persistent workspace must be on a CUDA/HIP device")
    if not workspace.is_contiguous():
        raise ValueError("persistent workspace must be contiguous")
    if workspace.numel() != expected:
        raise ValueError(
            f"persistent workspace must have {expected} int32 elements, "
            f"got {workspace.numel()}"
        )


def run_split_topk_merge_persistent(
    candidate_scores,
    candidate_positions,
    candidate_counts,
    workspace,
    *,
    k: int,
    parts: int,
    out_scores=None,
    out_positions=None,
):
    rows, splits, local_k = candidate_scores.shape
    if local_k != k:
        raise ValueError(f"local_k must equal k, got {local_k} and {k}")
    _validate_persistent_workspace(workspace, rows)
    if candidate_counts.dtype != torch.int32:
        raise TypeError("candidate_counts must be int32")
    if not candidate_counts.is_contiguous():
        raise ValueError("candidate_counts must be contiguous")
    scores = candidate_scores.view(rows, splits * k)
    positions = candidate_positions.view(rows, splits * k)
    if out_scores is None:
        out_scores = torch.empty(
            (rows, k),
            dtype=torch.float32,
            device=candidate_scores.device,
        )
    if out_positions is None:
        out_positions = torch.empty(
            (rows, k),
            dtype=torch.int32,
            device=candidate_scores.device,
        )
    launcher = build_split_topk_merge_persistent(k, splits, parts)
    _run_compiled(
        launcher,
        scores,
        positions,
        candidate_counts,
        out_scores,
        out_positions,
        workspace,
        rows,
        torch.cuda.current_stream(candidate_scores.device),
    )
    return out_scores, out_positions
