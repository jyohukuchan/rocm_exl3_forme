"""Bottom-up spherical Ward merging on the GPU, one program a cluster block - the rounds of `lsa.core.ward`.

Each round every live bucket finds its cheapest partner under the cap, the block's Gram matrix taken a tile at a
time and never whole, and every pair that is each other's cheapest merges into its lower pool - the cheapest
pairs first in the round that would pass the block's count. The key sums, sizes and links a block works on sit
in scratch rows of its own, so a block of 2,048 pools needs no more per pool than one of 128 does.

`cluster_node_buckets` is the same merge to half the buckets under a cap of two, spread over many programs a round, for the
hierarchy's growth. `bucket_members_from_labels` turns the labels into the member table.
"""

import torch
import triton
import triton.language as tl

from .keys import ordered_bits

# The Gram matrix's dot - exact, three tf32 passes that match it, or one
PRECISION = {"ieee": 0, "tf32x3": 1, "tf32": 2}
MAX_KEY = tl.constexpr(0x7FFFFFFFFFFFFFFF)
INDEX_BITS = tl.constexpr(12)


@triton.jit
def _dot_precision(rows: tl.tensor, columns_transposed: tl.tensor, PREC: tl.constexpr) -> tl.tensor:
    """Return the dot of a tile of rows with a tile of transposed columns at the precision PREC names."""
    if PREC == 0:
        return tl.dot(rows, columns_transposed, input_precision="ieee")
    elif PREC == 1:
        return tl.dot(rows, columns_transposed, input_precision="tf32x3")
    else:
        return tl.dot(rows, columns_transposed, input_precision="tf32")


@triton.jit
def _halve(runs: tl.tensor, BLOCK: tl.constexpr, N: tl.constexpr) -> tl.tensor:
    """Add each of a row's N partial sums [BLOCK, N] to the one N / 2 along, giving [BLOCK, N / 2]."""
    lower, upper = tl.split(tl.permute(tl.reshape(runs, [BLOCK, 2, N // 2]), (0, 2, 1)))
    return lower + upper


@triton.jit
def _square_sum(vectors: tl.tensor, BLOCK: tl.constexpr, D: tl.constexpr) -> tl.tensor:
    """Return each row's squared length [BLOCK], in one rounding order whatever the Triton build.

    Near-tied merge costs turn on a square's last bit, and `tl.sum` rounds as the build lays it out.
    So runs of four dims (second squared, then the first, third and fourth fused in) are added in halves -
    the order Triton 3.7 compiled `tl.sum` to at d = 64, which the internal snapshot's reads were taken with.
    """
    first_third, second_fourth = tl.split(tl.reshape(vectors, [BLOCK, D // 4, 2, 2]))
    first, third = tl.split(first_third)
    second, fourth = tl.split(second_fourth)
    runs = tl.fma(fourth, fourth, tl.fma(third, third, tl.fma(first, first, second * second)))
    if D // 4 >= 64:
        runs = _halve(runs, BLOCK, 64)
    if D // 4 >= 32:
        runs = _halve(runs, BLOCK, 32)
    if D // 4 >= 16:
        runs = _halve(runs, BLOCK, 16)
    if D // 4 >= 8:
        runs = _halve(runs, BLOCK, 8)
    runs = _halve(_halve(runs, BLOCK, 4), BLOCK, 2)
    return tl.reshape(runs, [BLOCK])


@triton.jit
def _merge_cost(
    row_sums: tl.tensor,
    row_square: tl.tensor,
    column_sums: tl.tensor,
    column_square: tl.tensor,
    PREC: tl.constexpr,
) -> tl.tensor:
    """Return the alignment a merge loses, a tile of rows against a tile of columns."""
    dot = _dot_precision(row_sums, tl.trans(column_sums), PREC)
    merged = tl.sqrt(tl.maximum(row_square[:, None] + column_square[None, :] + 2.0 * dot, 0.0))
    row_alignment = tl.sqrt(tl.maximum(row_square, 0.0))
    column_alignment = tl.sqrt(tl.maximum(column_square, 0.0))
    return row_alignment[:, None] + column_alignment[None, :] - merged


@triton.jit
def _merge_cost_key(merge_cost: tl.tensor, index: tl.tensor) -> tl.tensor:
    """Return an fp32 merge cost as an order-preserving int64 with the index below it.

    The least key is the cheapest, and the lower index on a tie.
    """
    ordered = ordered_bits(merge_cost).to(tl.int64) + (1 << 31)
    return (ordered << INDEX_BITS) | index.to(tl.int64)


@triton.jit
def _ward_merge_kernel(
    key_sums_ptr: tl.tensor,
    sizes_ptr: tl.tensor,
    labels_ptr: tl.tensor,
    reached_ptr: tl.tensor,
    work_ptr: tl.tensor,
    square_ptr: tl.tensor,
    size_ptr: tl.tensor,
    alive_ptr: tl.tensor,
    parent_ptr: tl.tensor,
    best_ptr: tl.tensor,
    key_ptr: tl.tensor,
    mutual_ptr: tl.tensor,
    bucket_count: tl.tensor,
    bucket_cap: tl.tensor,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    PREC: tl.constexpr,
    MAX_ROUNDS: tl.constexpr,
    JUMPS: tl.constexpr,
) -> None:
    """Merge one cluster block's pools down to `bucket_count` buckets, one program a block.

    Args:
        key_sums_ptr: Pool key sums [G, P, D] fp32.
        sizes_ptr: Keys each pool holds [G, P] fp32.
        labels_ptr: Each pool's bucket [G, P] int64.
        reached_ptr: Whether each block reached its count [G] int32.
        work_ptr: Scratch for the merged key sums [G, P, D] fp32.
        square_ptr: Scratch for the squared lengths [G, P] fp32.
        size_ptr: Scratch for the sizes [G, P] fp32.
        alive_ptr: Scratch for the live pools [G, P] int32.
        parent_ptr: Scratch for the pool each merged into [G, P] int32.
        best_ptr: Scratch for the cheapest partners [G, P] int32.
        key_ptr: Scratch for the cheapest partners' keys [G, P] int64.
        mutual_ptr: Scratch for the keys of mutual pairs [G, P] int64.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.
    """
    block = tl.program_id(0).to(tl.int64)
    base = block * P
    dims = tl.arange(0, D)
    lane = tl.arange(0, BLOCK)
    index_mask = (1 << INDEX_BITS) - 1

    for tile in range(0, P // BLOCK):
        rows = tile * BLOCK + lane
        at = (base + rows)[:, None] * D + dims[None, :]
        pools = tl.load(key_sums_ptr + at)
        tl.store(work_ptr + at, pools)
        tl.store(square_ptr + base + rows, _square_sum(pools, BLOCK, D))
        tl.store(size_ptr + base + rows, tl.load(sizes_ptr + base + rows))
        tl.store(alive_ptr + base + rows, tl.full([BLOCK], 1, tl.int32))
        tl.store(parent_ptr + base + rows, rows)
    tl.debug_barrier()

    alive_count = P
    applied = 1
    rounds = 0
    while (alive_count > bucket_count) & (applied > 0) & (rounds < MAX_ROUNDS):
        # Every live bucket's cheapest partner the cap allows, a tile of rows against every tile of columns
        for tile in range(0, P // BLOCK):
            rows = tile * BLOCK + lane
            row_sums = tl.load(work_ptr + (base + rows)[:, None] * D + dims[None, :])
            row_square = tl.load(square_ptr + base + rows)
            row_size = tl.load(size_ptr + base + rows)
            row_alive = tl.load(alive_ptr + base + rows) > 0
            best = tl.full([BLOCK], MAX_KEY, tl.int64)
            for column_tile in range(0, P // BLOCK):
                columns = column_tile * BLOCK + lane
                column_sums = tl.load(work_ptr + (base + columns)[:, None] * D + dims[None, :])
                column_square = tl.load(square_ptr + base + columns)
                column_size = tl.load(size_ptr + base + columns)
                column_alive = tl.load(alive_ptr + base + columns) > 0
                merge_cost = _merge_cost(row_sums, row_square, column_sums, column_square, PREC)
                allowed = (
                    row_alive[:, None]
                    & column_alive[None, :]
                    & (rows[:, None] != columns[None, :])
                    & (row_size[:, None] + column_size[None, :] <= bucket_cap)
                )
                key = tl.where(allowed, _merge_cost_key(merge_cost, columns[None, :]), MAX_KEY)
                best = tl.minimum(best, tl.min(key, axis=1))
            tl.store(key_ptr + base + rows, best)
            tl.store(best_ptr + base + rows, tl.where(best < MAX_KEY, (best & index_mask).to(tl.int32), -1))
        tl.debug_barrier()

        # A pair that is each other's cheapest, named by its lower pool
        mutual_count = 0
        for tile in range(0, P // BLOCK):
            rows = tile * BLOCK + lane
            partner = tl.load(best_ptr + base + rows)
            back = tl.load(best_ptr + base + partner, mask=partner >= 0, other=-1)
            mutual = (partner >= 0) & (back == rows) & (rows < partner)
            tl.store(mutual_ptr + base + rows, tl.where(mutual, tl.load(key_ptr + base + rows), MAX_KEY))
            mutual_count += tl.sum(mutual.to(tl.int32), axis=0)
        tl.debug_barrier()

        # The cheapest pairs the block's count leaves room for merge, the lower pool taking the higher
        merges_left = alive_count - bucket_count
        applied = tl.minimum(mutual_count, merges_left)
        for tile in range(0, P // BLOCK):
            rows = tile * BLOCK + lane
            mine = tl.load(mutual_ptr + base + rows)
            keep = mine < MAX_KEY
            if mutual_count > merges_left:
                rank = tl.zeros([BLOCK], dtype=tl.int32)
                for column_tile in range(0, P // BLOCK):
                    theirs = tl.load(mutual_ptr + base + column_tile * BLOCK + lane)
                    rank += tl.sum((theirs[None, :] < mine[:, None]).to(tl.int32), axis=1)
                keep = keep & (rank < merges_left)
            partner = tl.load(best_ptr + base + rows)
            row_at = (base + rows)[:, None] * D + dims[None, :]
            partner_at = (base + partner)[:, None] * D + dims[None, :]
            joined = tl.load(work_ptr + row_at, mask=keep[:, None], other=0.0) + tl.load(
                work_ptr + partner_at, mask=keep[:, None], other=0.0
            )
            tl.store(work_ptr + row_at, joined, mask=keep[:, None])
            tl.store(square_ptr + base + rows, _square_sum(joined, BLOCK, D), mask=keep)
            partner_size = tl.load(size_ptr + base + partner, mask=keep, other=0.0)
            tl.store(size_ptr + base + rows, tl.load(size_ptr + base + rows) + partner_size, mask=keep)
            tl.store(alive_ptr + base + partner, tl.zeros([BLOCK], dtype=tl.int32), mask=keep)
            tl.store(parent_ptr + base + partner, rows, mask=keep)
        tl.debug_barrier()
        alive_count -= applied
        rounds += 1

    # Every merged pool points at the pool it merged into - jump along the chain to the surviving one
    for _jump in range(JUMPS):
        for tile in range(0, P // BLOCK):
            rows = tile * BLOCK + lane
            parent = tl.load(parent_ptr + base + rows)
            tl.store(parent_ptr + base + rows, tl.load(parent_ptr + base + parent))
        tl.debug_barrier()
    # A surviving pool's bucket is numbered by how many survive before it
    seen = 0
    for tile in range(0, P // BLOCK):
        rows = tile * BLOCK + lane
        alive = tl.load(alive_ptr + base + rows)
        tl.store(best_ptr + base + rows, tl.cumsum(alive, axis=0) + seen - 1)
        seen += tl.sum(alive, axis=0)
    tl.debug_barrier()
    for tile in range(0, P // BLOCK):
        rows = tile * BLOCK + lane
        root = tl.load(parent_ptr + base + rows)
        tl.store(labels_ptr + base + rows, tl.load(best_ptr + base + root).to(tl.int64))
    tl.store(reached_ptr + block, (alive_count <= bucket_count).to(tl.int32))


def check_pools(pool_keys: torch.Tensor) -> None:
    """Check the pool keys a Ward merge takes.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim].

    Raises:
        ValueError: If P is not a power of two from 16 to 4,096, or index_dim is not a power of two from 16 to 256.
    """
    _, pools, index_dim = pool_keys.shape
    if pools & (pools - 1) or not 16 <= pools <= 1 << 12:
        raise ValueError(f"a block's pools must be a power of two from 16 to 4,096 - got {pools}")
    if index_dim & (index_dim - 1) or not 16 <= index_dim <= 256:
        raise ValueError(f"the index dim must be a power of two from 16 to 256 - got {index_dim}")


def ward_merge(
    pool_keys: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    precision: str = "tf32x3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each cluster block's pools down to `bucket_count` buckets on the GPU.

    The same contract as `lsa.core.ward.ward_merge`, except that a block the cap stops short is reported in
    the second return rather than raised, since a launch cannot raise from the device.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32, P a power of two from 16 to 4,096, index_dim a power of two from 16
            to 256.
        pool_sizes: Keys each pool holds [G, P] fp32.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.
        precision: The Gram matrix's dot - `ieee`, `tf32x3` or `tf32`.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.

    Raises:
        ValueError: If P is not a power of two from 16 to 4,096, or index_dim is not a power of two from 16 to 256.
    """
    check_pools(pool_keys)
    blocks, pools, index_dim = pool_keys.shape
    device = pool_keys.device
    key_sums = pool_keys.float().contiguous()
    labels = torch.empty(blocks, pools, dtype=torch.int64, device=device)
    reached = torch.empty(blocks, dtype=torch.int32, device=device)
    work = torch.empty_like(key_sums)
    square, size = (torch.empty(blocks, pools, device=device) for _ in range(2))
    alive, parent, best = (torch.empty(blocks, pools, dtype=torch.int32, device=device) for _ in range(3))
    key, mutual = (torch.empty(blocks, pools, dtype=torch.int64, device=device) for _ in range(2))
    _ward_merge_kernel[(blocks,)](
        key_sums,
        pool_sizes.float().contiguous(),
        labels,
        reached,
        work,
        square,
        size,
        alive,
        parent,
        best,
        key,
        mutual,
        bucket_count,
        float(bucket_cap),
        P=pools,
        D=index_dim,
        BLOCK=min(pools, 64),
        PREC=PRECISION[precision],
        MAX_ROUNDS=pools,
        JUMPS=max(1, (pools - 1).bit_length()),
        num_warps=4,
    )
    return labels, reached.bool()


def bucket_members_from_labels(labels: torch.Tensor, bucket_count: int, bucket_cap: int) -> torch.Tensor:
    """Turn each block's labels into its member table - the keys of every bucket, in position order.

    Args:
        labels: The bucket of each key [G, T] int64, from 0 to `bucket_count` - 1, at most `bucket_cap` keys a
            bucket. Block-local: key `i` of block `g` is position `i`.
        bucket_count: Buckets a block holds.
        bucket_cap: Row width, the most keys a bucket holds.

    Returns:
        Bucket members [G, bucket_count, bucket_cap] int32 - block-local positions, -1 past a bucket's size.
    """
    blocks, key_count = labels.shape
    # [G, T] int64 block-local positions grouped stably by bucket label
    positions_by_bucket = torch.argsort(labels, dim=1, stable=True)
    # [G, bucket_count] int64 each bucket's size and where its run starts in the sorted positions
    sizes = torch.zeros(blocks, bucket_count, dtype=torch.long, device=labels.device)
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    starts = sizes.cumsum(1) - sizes
    slot = torch.arange(bucket_cap, device=labels.device)
    runs = (starts[..., None] + slot).clamp_max(key_count - 1).view(blocks, -1)
    members = positions_by_bucket.gather(1, runs).view(blocks, bucket_count, bucket_cap)
    return torch.where(slot < sizes[..., None], members, -1).to(torch.int32)
