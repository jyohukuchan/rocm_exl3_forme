"""Expansion - the picked buckets read as their keys, and the window and the sink folded in after.

Two forms choose the same keys. `expand_in_order` walks the picks best first and stops once the top-k is full.
`expand_by_threshold` finds the lowest packed key at which the listed buckets fill the top-k and emits every
bucket above it in listing order, so it needs no sort. A key inside the window or the sink is not counted
toward the top-k, because attention reads it anyway, and both forms close with the same window-and-sink tail.

Both write a read list per row, `[top_k + window_size + sink_size]` int32 and -1 padded, and its length.
Under an early budget a row before its position reads the early `top_k` and `top_p` instead, chosen on the device from the
row's position, and its list stays -1 padded to the same width.
"""

import torch
import triton
import triton.language as tl

from .budget import EarlyBudget
from .keys import key_bucket


@triton.jit
def _append_window_and_sink(
    out_row_ptr: tl.tensor,
    slot_row_ptr: tl.tensor,
    position: tl.tensor,
    window_start: tl.tensor,
    filled: tl.tensor,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    LOCAL_BLOCK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
) -> tl.tensor:
    """Append the window's last positions then the sinks after `filled` keys, the one tail of a read.

    Args:
        out_row_ptr: The row's read list int32.
        slot_row_ptr: The row's positions to KV slots int32 - read only if MAP_SLOTS.
        position: The row's position.
        window_start: First position of the window.
        filled: Keys already in the read list.

    Returns:
        Keys the read list holds.
    """
    local = tl.arange(0, LOCAL_BLOCK)
    in_window = local < WINDOW_SIZE
    window_id = position + 1 - WINDOW_SIZE + local
    in_sinks = (local >= WINDOW_SIZE) & (local < WINDOW_SIZE + SINK_SIZE)
    sink_id = local - WINDOW_SIZE
    ids = tl.where(in_window, window_id, sink_id)
    valid = (in_window & (window_id >= 0)) | (in_sinks & (sink_id < window_start) & (sink_id <= position))
    at = filled + tl.cumsum(valid.to(tl.int32), 0) - 1
    if MAP_SLOTS:
        ids = tl.load(slot_row_ptr + ids, mask=valid, other=-1)
    tl.store(out_row_ptr + at, ids, mask=valid)
    return filled + tl.sum(valid.to(tl.int32), 0)


@triton.jit
def _row_budget(
    position: tl.tensor,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    EARLY_POSITION: tl.constexpr,
    EARLY_TOP_K: tl.constexpr,
    EARLY_TOP_P: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    """Return a row's top-k and top-p - the early budget's before its position, the design's from it.

    No early budget (EARLY_POSITION 0) leaves both constants.
    """
    top_k = TOP_K
    top_p = TOP_P
    if EARLY_POSITION > 0:
        early = position < EARLY_POSITION
        top_k = tl.where(early, EARLY_TOP_K, TOP_K)
        top_p = tl.where(early, EARLY_TOP_P, TOP_P)
    return top_k, top_p


@triton.jit
def _expand_in_order(
    selected_ptr: tl.tensor,
    bucket_members_ptr: tl.tensor,
    positions_ptr: tl.tensor,
    out_ptr: tl.tensor,
    lengths_ptr: tl.tensor,
    slot_table_ptr: tl.tensor,
    slot_rows_ptr: tl.tensor,
    selected_stride: tl.tensor,
    out_stride: tl.tensor,
    slot_stride: tl.tensor,
    TOP_P: tl.constexpr,
    BUCKET_CAP: tl.constexpr,
    TILE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    WIDTH: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    LOCAL_BLOCK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    EARLY_POSITION: tl.constexpr,
    EARLY_TOP_K: tl.constexpr,
    EARLY_TOP_P: tl.constexpr,
) -> None:
    """Expand one row's picks best first into its read list, stopping once the top-k is full.

    Args:
        selected_ptr: Picked cache rows [R, TOP_P] int32, best score first, -1 padded.
        bucket_members_ptr: Members of each cache row [rows, BUCKET_CAP] int32, -1 padded.
        positions_ptr: Each row's position [R] int32.
        out_ptr: Read lists [R, WIDTH] int32, -1 padded.
        lengths_ptr: Keys each read list holds [R] int32.
        slot_table_ptr: Each request's positions to KV slots [requests, context] int32 - read if MAP_SLOTS.
        slot_rows_ptr: Each row's request [R] int64 - read only if MAP_SLOTS.
    """
    row = tl.program_id(0).to(tl.int64)
    position = tl.load(positions_ptr + row)
    top_k, top_p = _row_budget(position, TOP_K, TOP_P, EARLY_POSITION, EARLY_TOP_K, EARLY_TOP_P)
    window_start = tl.maximum(0, position + 1 - WINDOW_SIZE)
    # The request's positions to KV slots, read as they are stored rather than in a pass of their own
    slot_row = 0
    if MAP_SLOTS:
        slot_row = tl.load(slot_rows_ptr + row).to(tl.int64)
    pad = tl.arange(0, PAD_BLOCK)
    tl.store(out_ptr + row * out_stride + pad, -1, mask=pad < WIDTH)
    tl.debug_barrier()
    # TILE picks at a time, best first, each a row of BUCKET_CAP members - the running count says where they land
    slot = tl.arange(0, TILE * BUCKET_CAP)
    filled = 0
    for first in range(0, TOP_P, TILE):
        pick = first + slot // BUCKET_CAP
        bucket = tl.load(
            selected_ptr + row * selected_stride + pick, mask=(pick < top_p) & (filled < top_k), other=-1
        )
        ids = tl.load(
            bucket_members_ptr + bucket.to(tl.int64) * BUCKET_CAP + slot % BUCKET_CAP,
            mask=bucket >= 0,
            other=-1,
        )
        valid = (bucket >= 0) & (ids >= SINK_SIZE) & (ids < window_start)
        at = filled + tl.cumsum(valid.to(tl.int32)) - 1
        stored = valid & (at < top_k)
        if MAP_SLOTS:
            ids = tl.load(slot_table_ptr + slot_row * slot_stride + ids, mask=stored, other=-1)
        tl.store(out_ptr + row * out_stride + at, ids, mask=stored)
        filled += tl.sum(valid.to(tl.int32))
    filled = tl.minimum(filled, top_k)
    length = _append_window_and_sink(
        out_ptr + row * out_stride,
        slot_table_ptr + slot_row * slot_stride,
        position,
        window_start,
        filled,
        WINDOW_SIZE,
        SINK_SIZE,
        LOCAL_BLOCK,
        MAP_SLOTS,
    )
    tl.store(lengths_ptr + row, length)


def expand_in_order(
    selected: torch.Tensor,
    bucket_members: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    sink_size: int,
    top_k: int,
    slot_table: torch.Tensor | None = None,
    slot_rows: torch.Tensor | None = None,
    early: EarlyBudget | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read lists from buckets of any size, walking the picks best first until the top-k is full.

    The walk takes a few picks at a time, so a wide member row costs what its members cost and the walk stops
    loading once the budget is full. A key below the sinks or inside the window is not counted, and the last
    bucket is cut short. The window then the sinks follow.

    Args:
        selected: Picked cache rows [R, top_p] int32, best score first, -1 padded.
        bucket_members: Members of each cache row [rows, bucket_cap] int32, -1 padded.
        positions: Each row's position [R] int32.
        window_size: Recent positions read whole.
        sink_size: First positions read whole.
        top_k: Bucket members read at most.
        slot_table: Each request's positions to KV slots [requests, context] int32 - when given, every position
            is stored as its slot instead, and `slot_rows` names each row's request.
        slot_rows: Each row's request [R] int64, read with `slot_table`.
        early: The budget of rows before its position - they walk its first `top_p` picks and stop at its `top_k` -
            or None to walk every pick to `top_k` throughout.

    Returns:
        Read lists [R, top_k + window_size + sink_size] int32, -1 padded, the expansion first.
        Keys each list holds [R] int32.
    """
    top_p = selected.size(1)
    bucket_cap = bucket_members.size(1)
    width = top_k + window_size + sink_size
    out = torch.empty(selected.size(0), width, device=selected.device, dtype=torch.int32)
    lengths = torch.empty(selected.size(0), device=selected.device, dtype=torch.int32)
    # A decode batch is a few rows, each a latency-bound walk - wide tiles take fewer steps; a prefill chunk's rows
    # fill the device, where narrow tiles keep more of them resident
    decoding = selected.size(0) <= 32
    tile = max(1, (4096 if decoding else 1024) // bucket_cap)
    mapped = slot_table is not None
    _expand_in_order[(selected.size(0),)](
        selected,
        bucket_members,
        positions,
        out,
        lengths,
        slot_table if slot_table is not None else bucket_members,
        slot_rows if slot_rows is not None else positions,
        selected.stride(0),
        width,
        slot_table.stride(0) if slot_table is not None else 0,
        TOP_P=top_p,
        BUCKET_CAP=bucket_cap,
        TILE=tile,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        TOP_K=top_k,
        WIDTH=width,
        PAD_BLOCK=triton.next_power_of_2(width),
        LOCAL_BLOCK=triton.next_power_of_2(window_size + sink_size),
        MAP_SLOTS=mapped,
        EARLY_POSITION=0 if early is None else early.position,
        EARLY_TOP_K=0 if early is None else early.top_k,
        EARLY_TOP_P=0 if early is None else early.top_p,
        num_warps=16 if decoding else 4,
    )
    return out, lengths
