# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact TopK merge for score-sorted split-local runs."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Array, Int32, gpu, range_constexpr

from .candidate_topk_common import f32_to_ordered_i32

WAVE_SIZE = 64


def _make_storage(splits: int):
    @fx.struct
    class Storage:
        cuts: Array[Int32, splits, 16]
        bases: Array[Int32, splits, 16]

    return Storage


def _make_parallel_storage(splits: int, capacity: int):
    @fx.struct
    class ParallelStorage:
        keys: Array[Int32, capacity, 16]
        sources: Array[Int32, capacity, 16]
        valid: Array[Int32, capacity, 16]
        cuts: Array[Int32, splits, 16]
        bases: Array[Int32, splits, 16]
        total: Array[Int32, 1, 4]

    return ParallelStorage


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


@cache
def build_multisequence_sorted_split_topk_merge(k: int, splits: int):
    """Build blocked representative-pruning selection over sorted runs."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 2 <= splits <= 2 * WAVE_SIZE:
        raise ValueError(f"splits must be in [2, {2 * WAVE_SIZE}], got {splits}")

    block_sizes = []
    remaining = k
    while remaining > 2 * splits:
        block_size = remaining // (2 * splits)
        block_sizes.append(block_size)
        remaining -= block_size * splits
    final_extracts = remaining
    tournament_width = _next_power_of_two(min(splits, WAVE_SIZE))
    tournament_offsets = tuple(
        1 << shift for shift in range(tournament_width.bit_length() - 2, -1, -1)
    )

    @flyc.jit
    def load_head(
        candidate_scores,
        row,
        run,
        index,
        count,
        active,
    ):
        valid = active & (index < count)
        safe_run = active.select(run, fx.Int32(0))
        safe_index = valid.select(index, fx.Int32(0))
        score = candidate_scores[row, safe_run, safe_index]
        return valid, f32_to_ordered_i32(score)

    @flyc.jit
    def clamp_count(count):
        nonnegative = (count < 0).select(fx.Int32(0), count)
        return (nonnegative > k).select(fx.Int32(k), nonnegative)

    @flyc.jit
    def wave_argmax(valid, key, run):
        winner_valid = valid.select(fx.Int32(1), fx.Int32(0))
        winner_key = key
        winner_run = run
        for offset in tournament_offsets:
            peer_valid = winner_valid.shuffle_xor(offset, tournament_width)
            peer_key = winner_key.shuffle_xor(offset, tournament_width)
            peer_run = winner_run.shuffle_xor(offset, tournament_width)
            take_peer = (peer_valid != 0) & (
                (winner_valid == 0)
                | (peer_key > winner_key)
                | ((peer_key == winner_key) & (peer_run < winner_run))
            )
            winner_valid = take_peer.select(peer_valid, winner_valid)
            winner_key = take_peer.select(peer_key, winner_key)
            winner_run = take_peer.select(peer_run, winner_run)
        return winner_valid != 0, winner_run

    @flyc.kernel(
        name=f"multisequence_sorted_split_topk_merge_k{k}_s{splits}",
        known_block_size=[WAVE_SIZE, 1, 1],
    )
    def kernel(
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        selected_scores: fx.Tensor,
        selected_positions: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        lane = fx.Int32(fx.thread_idx.x)
        run0 = lane
        run1 = lane + fx.Int32(WAVE_SIZE)
        active0 = run0 < fx.Int32(splits)
        active1 = run1 < fx.Int32(splits)
        safe_run0 = active0.select(run0, fx.Int32(0))
        safe_run1 = active1.select(run1, fx.Int32(0))
        count0 = active0.select(
            clamp_count(fx.Int32(candidate_counts[row, safe_run0])),
            fx.Int32(0),
        )
        count1 = active1.select(
            clamp_count(fx.Int32(candidate_counts[row, safe_run1])),
            fx.Int32(0),
        )

        total_count = count0 + count1
        for offset in (32, 16, 8, 4, 2, 1):
            total_count = total_count + total_count.shuffle_xor(offset, WAVE_SIZE)
        need = (total_count < fx.Int32(k)).select(total_count, fx.Int32(k))

        cut0 = fx.Int32(0)
        cut1 = fx.Int32(0)
        if total_count <= fx.Int32(k):
            cut0 = count0
            cut1 = count1
        else:
            for block_size in block_sizes:
                selected0 = fx.Int32(0)
                selected1 = fx.Int32(0)
                index0 = cut0 + fx.Int32(block_size - 1)
                index1 = cut1 + fx.Int32(block_size - 1)
                valid0, key0 = load_head(
                    candidate_scores,
                    row,
                    run0,
                    index0,
                    count0,
                    active0,
                )
                valid1, key1 = load_head(
                    candidate_scores,
                    row,
                    run1,
                    index1,
                    count1,
                    active1,
                )
                for _ in range_constexpr(splits):
                    take1 = valid1 & (
                        (~valid0)
                        | (key1 > key0)
                        | ((key1 == key0) & (run1 < run0))
                    )
                    local_valid = valid0 | valid1
                    local_key = take1.select(key1, key0)
                    local_run = take1.select(run1, run0)
                    winner_valid, winner_run = wave_argmax(
                        local_valid,
                        local_key,
                        local_run,
                    )
                    if winner_valid & (winner_run == run0):
                        selected0 = selected0 + 1
                        index0 = (
                            cut0
                            + fx.Int32(block_size) * (selected0 + 1)
                            - fx.Int32(1)
                        )
                        valid0, key0 = load_head(
                            candidate_scores,
                            row,
                            run0,
                            index0,
                            count0,
                            active0,
                        )
                    if winner_valid & (winner_run == run1):
                        selected1 = selected1 + 1
                        index1 = (
                            cut1
                            + fx.Int32(block_size) * (selected1 + 1)
                            - fx.Int32(1)
                        )
                        valid1, key1 = load_head(
                            candidate_scores,
                            row,
                            run1,
                            index1,
                            count1,
                            active1,
                        )
                cut0 = cut0 + fx.Int32(block_size) * selected0
                cut1 = cut1 + fx.Int32(block_size) * selected1

            selected0 = fx.Int32(0)
            selected1 = fx.Int32(0)
            valid0, key0 = load_head(
                candidate_scores,
                row,
                run0,
                cut0,
                count0,
                active0,
            )
            valid1, key1 = load_head(
                candidate_scores,
                row,
                run1,
                cut1,
                count1,
                active1,
            )
            for _ in range_constexpr(final_extracts):
                take1 = valid1 & (
                    (~valid0)
                    | (key1 > key0)
                    | ((key1 == key0) & (run1 < run0))
                )
                local_valid = valid0 | valid1
                local_key = take1.select(key1, key0)
                local_run = take1.select(run1, run0)
                winner_valid, winner_run = wave_argmax(
                    local_valid,
                    local_key,
                    local_run,
                )
                if winner_valid & (winner_run == run0):
                    selected0 = selected0 + 1
                    valid0, key0 = load_head(
                        candidate_scores,
                        row,
                        run0,
                        cut0 + selected0,
                        count0,
                        active0,
                    )
                if winner_valid & (winner_run == run1):
                    selected1 = selected1 + 1
                    valid1, key1 = load_head(
                        candidate_scores,
                        row,
                        run1,
                        cut1 + selected1,
                        count1,
                        active1,
                    )
            cut0 = cut0 + selected0
            cut1 = cut1 + selected1

        storage = fx.SharedAllocator().allocate(_make_storage(splits))
        cuts = storage.cuts.peek().view(fx.make_layout(splits, 1))
        bases = storage.bases.peek().view(fx.make_layout(splits, 1))
        if active0:
            cuts[run0] = cut0
        if active1:
            cuts[run1] = cut1
        gpu.barrier()

        if lane == 0:
            running = fx.Int32(0)
            for split in range_constexpr(splits):
                bases[split] = running
                running = running + cuts[split]
        gpu.barrier()

        for split in range(fx.Int32(0), fx.Int32(splits), fx.Int32(1)):
            split_count = cuts[split]
            output_base = bases[split]
            for local in range(lane, split_count, fx.Int32(WAVE_SIZE)):
                output = output_base + local
                selected_scores[row, output] = candidate_scores[row, split, local]
                selected_positions[row, output] = candidate_positions[
                    row, split, local
                ]

        for output in range(
            need + lane,
            fx.Int32(k),
            fx.Int32(WAVE_SIZE),
        ):
            selected_scores[row, output] = fx.Float32(float("-inf"))
            selected_positions[row, output] = fx.Int32(-1)

    @flyc.jit
    def launch(
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        selected_scores: fx.Tensor,
        selected_positions: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            candidate_scores,
            candidate_positions,
            candidate_counts,
            selected_scores,
            selected_positions,
        ).launch(
            grid=(rows, 1, 1),
            block=(WAVE_SIZE, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1}
    return launch


@cache
def build_parallel_multisequence_sorted_split_topk_merge(k: int, splits: int):
    """Build representative pruning with parallel LDS selection rounds."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 2 <= splits <= WAVE_SIZE:
        raise ValueError(f"splits must be in [2, {WAVE_SIZE}], got {splits}")

    block_sizes = []
    remaining = k
    while remaining > 2 * splits:
        block_size = remaining // (2 * splits)
        block_sizes.append(block_size)
        remaining -= block_size * splits
    final_extracts = remaining
    round_capacity = _next_power_of_two(splits * splits)
    final_capacity = _next_power_of_two(splits * final_extracts)
    capacity = max(round_capacity, final_capacity)
    block_threads = 256

    @flyc.jit
    def clamp_count(count):
        nonnegative = (count < 0).select(fx.Int32(0), count)
        return (nonnegative > k).select(fx.Int32(k), nonnegative)

    @flyc.jit
    def bitonic_sort(keys, sources, valid, size):
        merge_size = 2
        while merge_size <= size:
            stride = merge_size // 2
            while stride > 0:
                for step in range_constexpr(
                    (size + block_threads - 1) // block_threads
                ):
                    index = (
                        fx.Int32(step * block_threads) + fx.Int32(fx.thread_idx.x)
                    )
                    partner = index ^ fx.Int32(stride)
                    if (partner > index) & (index < fx.Int32(size)):
                        a_valid = valid[index]
                        b_valid = valid[partner]
                        a_key = keys[index]
                        b_key = keys[partner]
                        a_source = sources[index]
                        b_source = sources[partner]
                        b_before_a = (b_valid > a_valid) | (
                            (b_valid == a_valid)
                            & (
                                (b_key > a_key)
                                | ((b_key == a_key) & (b_source < a_source))
                            )
                        )
                        a_before_b = (a_valid > b_valid) | (
                            (a_valid == b_valid)
                            & (
                                (a_key > b_key)
                                | ((a_key == b_key) & (a_source < b_source))
                            )
                        )
                        first_half = (index & fx.Int32(merge_size)) == 0
                        should_swap = first_half.select(b_before_a, a_before_b)
                        if should_swap:
                            valid[index] = b_valid
                            valid[partner] = a_valid
                            keys[index] = b_key
                            keys[partner] = a_key
                            sources[index] = b_source
                            sources[partner] = a_source
                gpu.barrier()
                stride //= 2
            merge_size *= 2

    @flyc.kernel(
        name=f"parallel_multisequence_sorted_split_topk_merge_k{k}_s{splits}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        selected_scores: fx.Tensor,
        selected_positions: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        storage = fx.SharedAllocator().allocate(
            _make_parallel_storage(splits, capacity)
        )
        keys = storage.keys.peek().view(fx.make_layout(capacity, 1))
        sources = storage.sources.peek().view(fx.make_layout(capacity, 1))
        valid = storage.valid.peek().view(fx.make_layout(capacity, 1))
        cuts = storage.cuts.peek().view(fx.make_layout(splits, 1))
        bases = storage.bases.peek().view(fx.make_layout(splits, 1))
        total = storage.total.peek().view(fx.make_layout(1, 1))
        round_layout = fx.make_layout((splits, splits), (splits, 1))
        final_layout = fx.make_layout(
            (splits, final_extracts),
            (final_extracts, 1),
        )

        if tid < fx.Int32(splits):
            cuts[tid] = 0
            bases[tid] = clamp_count(candidate_counts[row, tid])
        gpu.barrier()

        total_count = fx.Int32(0)
        if tid == 0:
            for split in range_constexpr(splits):
                total_count = total_count + bases[split]
            total[0] = total_count
        gpu.barrier()
        total_count = total[0]
        need = (total_count < fx.Int32(k)).select(total_count, fx.Int32(k))

        if total_count <= fx.Int32(k):
            if tid < fx.Int32(splits):
                cuts[tid] = bases[tid]
        else:
            for block_size in block_sizes:
                for step in range_constexpr(
                    (round_capacity + block_threads - 1) // block_threads
                ):
                    item = fx.Int32(step * block_threads) + tid
                    if item < fx.Int32(round_capacity):
                        is_candidate = item < fx.Int32(splits * splits)
                        safe_item = is_candidate.select(item, fx.Int32(0))
                        split, representative = fx.idx2crd(
                            safe_item,
                            round_layout,
                        ).unpack()
                        index = (
                            cuts[split]
                            + fx.Int32(block_size) * (representative + 1)
                            - 1
                        )
                        is_valid = is_candidate & (
                            index < bases[split]
                        )
                        safe_index = is_valid.select(index, fx.Int32(0))
                        keys[item] = f32_to_ordered_i32(
                            candidate_scores[row, split, safe_index]
                        )
                        sources[item] = safe_item
                        valid[item] = is_valid.select(fx.Int32(1), fx.Int32(0))
                gpu.barrier()
                bitonic_sort(keys, sources, valid, round_capacity)

                if tid < fx.Int32(splits):
                    selected_blocks = fx.Int32(0)
                    for item in range_constexpr(splits):
                        source_split, _ = fx.idx2crd(
                            sources[item],
                            round_layout,
                        ).unpack()
                        selected_blocks = selected_blocks + (
                            (valid[item] != 0) & (source_split == tid)
                        ).select(fx.Int32(1), fx.Int32(0))
                    cuts[tid] = (
                        cuts[tid] + fx.Int32(block_size) * selected_blocks
                    )
                    cuts[tid] = (cuts[tid] < bases[tid]).select(
                        cuts[tid],
                        bases[tid],
                    )
                gpu.barrier()

            for step in range_constexpr(
                (final_capacity + block_threads - 1) // block_threads
            ):
                item = fx.Int32(step * block_threads) + tid
                if item < fx.Int32(final_capacity):
                    is_candidate = item < fx.Int32(splits * final_extracts)
                    safe_item = is_candidate.select(item, fx.Int32(0))
                    split, offset = fx.idx2crd(
                        safe_item,
                        final_layout,
                    ).unpack()
                    index = cuts[split] + offset
                    is_valid = is_candidate & (index < bases[split])
                    safe_index = is_valid.select(index, fx.Int32(0))
                    keys[item] = f32_to_ordered_i32(
                        candidate_scores[row, split, safe_index]
                    )
                    sources[item] = safe_item
                    valid[item] = is_valid.select(fx.Int32(1), fx.Int32(0))
            gpu.barrier()
            bitonic_sort(keys, sources, valid, final_capacity)

            if tid < fx.Int32(splits):
                selected_items = fx.Int32(0)
                for item in range_constexpr(final_extracts):
                    source_split, _ = fx.idx2crd(
                        sources[item],
                        final_layout,
                    ).unpack()
                    selected_items = selected_items + (
                        (valid[item] != 0) & (source_split == tid)
                    ).select(fx.Int32(1), fx.Int32(0))
                cuts[tid] = cuts[tid] + selected_items
                cuts[tid] = (cuts[tid] < bases[tid]).select(
                    cuts[tid],
                    bases[tid],
                )
            gpu.barrier()

        if tid == 0:
            running = fx.Int32(0)
            for split in range_constexpr(splits):
                bases[split] = running
                running = running + cuts[split]
        gpu.barrier()

        for split in range(fx.Int32(0), fx.Int32(splits), fx.Int32(1)):
            split_count = cuts[split]
            output_base = bases[split]
            for local in range(tid, split_count, fx.Int32(block_threads)):
                output = output_base + local
                selected_scores[row, output] = candidate_scores[row, split, local]
                selected_positions[row, output] = candidate_positions[
                    row, split, local
                ]
        for output in range(
            need + tid,
            fx.Int32(k),
            fx.Int32(block_threads),
        ):
            selected_scores[row, output] = fx.Float32(float("-inf"))
            selected_positions[row, output] = fx.Int32(-1)

    @flyc.jit
    def launch(
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        selected_scores: fx.Tensor,
        selected_positions: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            candidate_scores,
            candidate_positions,
            candidate_counts,
            selected_scores,
            selected_positions,
        ).launch(
            grid=(rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1}
    return launch


@cache
def build_coordinated_rank_sorted_split_topk_merge(k: int, splits: int):
    """Build an exact fixed-round multisequence partition and prefix copy."""
    if k <= 0 or k & (k - 1):
        raise ValueError(f"k must be a positive power of two, got {k}")
    if not 2 <= splits <= 2 * WAVE_SIZE:
        raise ValueError(f"splits must be in [2, {2 * WAVE_SIZE}], got {splits}")

    tournament_width = _next_power_of_two(min(splits, WAVE_SIZE))
    tournament_offsets = tuple(
        1 << shift for shift in range(tournament_width.bit_length() - 2, -1, -1)
    )
    refinement_sizes = tuple(1 << shift for shift in range(k.bit_length() - 2, -1, -1))

    @flyc.jit
    def clamp_count(count):
        nonnegative = (count < 0).select(fx.Int32(0), count)
        return (nonnegative > k).select(fx.Int32(k), nonnegative)

    @flyc.jit
    def load_key(candidate_scores, row, run, index, count, active):
        valid = active & (index >= 0) & (index < count)
        safe_run = active.select(run, fx.Int32(0))
        safe_index = valid.select(index, fx.Int32(0))
        key = f32_to_ordered_i32(candidate_scores[row, safe_run, safe_index])
        return valid, key

    @flyc.jit
    def wave_best(valid, key, run):
        winner_valid = valid.select(fx.Int32(1), fx.Int32(0))
        winner_key = key
        winner_run = run
        for offset in tournament_offsets:
            peer_valid = winner_valid.shuffle_xor(offset, tournament_width)
            peer_key = winner_key.shuffle_xor(offset, tournament_width)
            peer_run = winner_run.shuffle_xor(offset, tournament_width)
            take_peer = (peer_valid != 0) & (
                (winner_valid == 0)
                | (peer_key > winner_key)
                | ((peer_key == winner_key) & (peer_run < winner_run))
            )
            winner_valid = take_peer.select(peer_valid, winner_valid)
            winner_key = take_peer.select(peer_key, winner_key)
            winner_run = take_peer.select(peer_run, winner_run)
        return winner_valid != 0, winner_key, winner_run

    @flyc.jit
    def wave_worst(valid, key, run):
        winner_valid = valid.select(fx.Int32(1), fx.Int32(0))
        winner_key = key
        winner_run = run
        for offset in tournament_offsets:
            peer_valid = winner_valid.shuffle_xor(offset, tournament_width)
            peer_key = winner_key.shuffle_xor(offset, tournament_width)
            peer_run = winner_run.shuffle_xor(offset, tournament_width)
            take_peer = (peer_valid != 0) & (
                (winner_valid == 0)
                | (peer_key < winner_key)
                | ((peer_key == winner_key) & (peer_run > winner_run))
            )
            winner_valid = take_peer.select(peer_valid, winner_valid)
            winner_key = take_peer.select(peer_key, winner_key)
            winner_run = take_peer.select(peer_run, winner_run)
        return winner_valid != 0, winner_key, winner_run

    @flyc.kernel(
        name=f"coordinated_rank_sorted_split_topk_merge_k{k}_s{splits}",
        known_block_size=[WAVE_SIZE, 1, 1],
    )
    def kernel(
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        selected_scores: fx.Tensor,
        selected_positions: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        lane = fx.Int32(fx.thread_idx.x)
        run0 = lane
        run1 = lane + fx.Int32(WAVE_SIZE)
        active0 = run0 < fx.Int32(splits)
        active1 = run1 < fx.Int32(splits)
        safe_run0 = active0.select(run0, fx.Int32(0))
        safe_run1 = active1.select(run1, fx.Int32(0))
        count0 = active0.select(
            clamp_count(candidate_counts[row, safe_run0]),
            fx.Int32(0),
        )
        count1 = active1.select(
            clamp_count(candidate_counts[row, safe_run1]),
            fx.Int32(0),
        )
        total_count = count0 + count1
        for offset in (32, 16, 8, 4, 2, 1):
            total_count = total_count + total_count.shuffle_xor(offset, WAVE_SIZE)
        need = (total_count < fx.Int32(k)).select(total_count, fx.Int32(k))

        a0 = fx.Int32(0)
        a1 = fx.Int32(0)
        b0 = fx.Int32(k - 1)
        b1 = fx.Int32(k - 1)
        if total_count <= fx.Int32(k):
            a0 = count0
            a1 = count1
        else:
            for h in refinement_sizes:
                left_valid0, left_key0 = load_key(
                    candidate_scores,
                    row,
                    run0,
                    a0 - 1,
                    count0,
                    active0 & (a0 > 0),
                )
                left_valid1, left_key1 = load_key(
                    candidate_scores,
                    row,
                    run1,
                    a1 - 1,
                    count1,
                    active1 & (a1 > 0),
                )
                take_left1 = left_valid1 & (
                    (~left_valid0)
                    | (left_key1 < left_key0)
                    | ((left_key1 == left_key0) & (run1 > run0))
                )
                lmax_valid, lmax_key, lmax_run = wave_worst(
                    left_valid0 | left_valid1,
                    take_left1.select(left_key1, left_key0),
                    take_left1.select(run1, run0),
                )

                middle0 = (a0 + b0) // fx.Int32(2)
                middle1 = (a1 + b1) // fx.Int32(2)
                middle_valid0, middle_key0 = load_key(
                    candidate_scores,
                    row,
                    run0,
                    middle0,
                    count0,
                    active0,
                )
                middle_valid1, middle_key1 = load_key(
                    candidate_scores,
                    row,
                    run1,
                    middle1,
                    count1,
                    active1,
                )
                middle_before0 = middle_valid0 & lmax_valid & (
                    (middle_key0 > lmax_key)
                    | ((middle_key0 == lmax_key) & (run0 < lmax_run))
                )
                middle_before1 = middle_valid1 & lmax_valid & (
                    (middle_key1 > lmax_key)
                    | ((middle_key1 == lmax_key) & (run1 < lmax_run))
                )
                next_a0 = a0 + fx.Int32(h)
                next_a0 = (next_a0 < count0).select(next_a0, count0)
                a0 = middle_before0.select(next_a0, a0)
                b0 = middle_before0.select(b0, b0 - fx.Int32(h))
                next_a1 = a1 + fx.Int32(h)
                next_a1 = (next_a1 < count1).select(next_a1, count1)
                a1 = middle_before1.select(next_a1, a1)
                b1 = middle_before1.select(b1, b1 - fx.Int32(h))

                left_blocks = a0 // fx.Int32(h) + a1 // fx.Int32(h)
                for offset in (32, 16, 8, 4, 2, 1):
                    left_blocks = left_blocks + left_blocks.shuffle_xor(
                        offset,
                        WAVE_SIZE,
                    )
                skew = fx.Int32(k // h) - left_blocks

                if skew > 0:
                    for correction in range_constexpr(splits):
                        if fx.Int32(correction) < skew:
                            frontier_valid0, frontier_key0 = load_key(
                                candidate_scores,
                                row,
                                run0,
                                b0,
                                count0,
                                active0,
                            )
                            frontier_valid1, frontier_key1 = load_key(
                                candidate_scores,
                                row,
                                run1,
                                b1,
                                count1,
                                active1,
                            )
                            take_frontier1 = frontier_valid1 & (
                                (~frontier_valid0)
                                | (frontier_key1 > frontier_key0)
                                | (
                                    (frontier_key1 == frontier_key0)
                                    & (run1 < run0)
                                )
                            )
                            winner_valid, _, winner_run = wave_best(
                                frontier_valid0 | frontier_valid1,
                                take_frontier1.select(
                                    frontier_key1,
                                    frontier_key0,
                                ),
                                take_frontier1.select(run1, run0),
                            )
                            if winner_valid & (winner_run == run0):
                                next_a0 = a0 + fx.Int32(h)
                                a0 = (next_a0 < count0).select(next_a0, count0)
                                b0 = b0 + fx.Int32(h)
                            if winner_valid & (winner_run == run1):
                                next_a1 = a1 + fx.Int32(h)
                                a1 = (next_a1 < count1).select(next_a1, count1)
                                b1 = b1 + fx.Int32(h)
                elif skew < 0:
                    for correction in range_constexpr(splits):
                        if fx.Int32(correction) < -skew:
                            boundary_valid0, boundary_key0 = load_key(
                                candidate_scores,
                                row,
                                run0,
                                a0 - 1,
                                count0,
                                active0 & (a0 > 0),
                            )
                            boundary_valid1, boundary_key1 = load_key(
                                candidate_scores,
                                row,
                                run1,
                                a1 - 1,
                                count1,
                                active1 & (a1 > 0),
                            )
                            take_boundary1 = boundary_valid1 & (
                                (~boundary_valid0)
                                | (boundary_key1 < boundary_key0)
                                | (
                                    (boundary_key1 == boundary_key0)
                                    & (run1 > run0)
                                )
                            )
                            winner_valid, _, winner_run = wave_worst(
                                boundary_valid0 | boundary_valid1,
                                take_boundary1.select(
                                    boundary_key1,
                                    boundary_key0,
                                ),
                                take_boundary1.select(run1, run0),
                            )
                            if winner_valid & (winner_run == run0):
                                a0 = a0 - fx.Int32(h)
                                b0 = b0 - fx.Int32(h)
                            if winner_valid & (winner_run == run1):
                                a1 = a1 - fx.Int32(h)
                                b1 = b1 - fx.Int32(h)

        storage = fx.SharedAllocator().allocate(_make_storage(splits))
        cuts = storage.cuts.peek().view(fx.make_layout(splits, 1))
        bases = storage.bases.peek().view(fx.make_layout(splits, 1))
        if active0:
            cuts[run0] = a0
        if active1:
            cuts[run1] = a1
        gpu.barrier()

        if lane == 0:
            running = fx.Int32(0)
            for split in range_constexpr(splits):
                bases[split] = running
                running = running + cuts[split]
        gpu.barrier()
        for split in range(fx.Int32(0), fx.Int32(splits), fx.Int32(1)):
            split_count = cuts[split]
            output_base = bases[split]
            for local in range(lane, split_count, fx.Int32(WAVE_SIZE)):
                output = output_base + local
                selected_scores[row, output] = candidate_scores[row, split, local]
                selected_positions[row, output] = candidate_positions[
                    row, split, local
                ]
        for output in range(
            need + lane,
            fx.Int32(k),
            fx.Int32(WAVE_SIZE),
        ):
            selected_scores[row, output] = fx.Float32(float("-inf"))
            selected_positions[row, output] = fx.Int32(-1)

    @flyc.jit
    def launch(
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        selected_scores: fx.Tensor,
        selected_positions: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            candidate_scores,
            candidate_positions,
            candidate_counts,
            selected_scores,
            selected_positions,
        ).launch(
            grid=(rows, 1, 1),
            block=(WAVE_SIZE, 1, 1),
            stream=stream,
        )

    launch.compile_hints = {"waves_per_eu": 1}
    return launch
