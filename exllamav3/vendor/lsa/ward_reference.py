"""Ward clustering of a cluster block's pools, in plain torch - the design's algorithm and the kernels' oracle.

A bucket is its key sum, and its centroid is that sum over its alignment, the key sum's norm. Merging two buckets
costs the sum of their alignments less the alignment of their key sums added, the alignment the union loses. Each round every bucket finds its
cheapest partner among those the cap allows, a tie going to the lower index, and every pair that is each other's
cheapest merges - the cheapest pairs first in the round that reaches the block's count. There is no seed and no
assignment pass: the merges are the partition.
"""

import math

import torch


def pool_keys(index_keys: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Sum each run of `pool_size` consecutive index keys into a pool key.

    Args:
        index_keys: A cluster block's index keys [G, T, index_dim] fp32, T a multiple of `pool_size`.
        pool_size: Keys in a pool.

    Returns:
        Pool keys [G, T / pool_size, index_dim] fp32.
    """
    blocks, key_count, index_dim = index_keys.shape
    return index_keys.reshape(blocks, key_count // pool_size, pool_size, index_dim).sum(2)


def ward_rounds(
    pool_keys: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    max_rounds: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge each block's pools down to `bucket_count` buckets and label every pool with its bucket.

    This is the merge itself. It stops short when the cap leaves no pair to merge, and says so in its second
    return - `ward_merge` is the form that refuses instead.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32.
        pool_sizes: Keys each pool holds [G, P] fp32.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.
        max_rounds: Rounds of mutual-cheapest merges at most.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.
        Whether each block reached its count [G] bool.
    """
    blocks, pools, _ = pool_keys.shape
    device = pool_keys.device
    key_sums = pool_keys.clone()
    bucket_sizes = pool_sizes.clone()
    alive = torch.ones(blocks, pools, dtype=torch.bool, device=device)
    parent = torch.arange(pools, device=device).expand(blocks, pools).clone()
    index = torch.arange(pools, device=device).expand(blocks, pools)
    diagonal = torch.eye(pools, dtype=torch.bool, device=device)
    for _ in range(max_rounds):
        merges_left = alive.sum(1) - bucket_count
        if not (merges_left > 0).any():
            break
        gram = torch.bmm(key_sums, key_sums.transpose(1, 2))
        squares = gram.diagonal(dim1=1, dim2=2)
        merged_alignments = (squares[:, :, None] + squares[:, None, :] + 2 * gram).clamp_min(0).sqrt()
        alignments = squares.clamp_min(0).sqrt()
        merge_costs = alignments[:, :, None] + alignments[:, None, :] - merged_alignments
        allowed = alive[:, :, None] & alive[:, None, :] & ~diagonal
        allowed &= bucket_sizes[:, :, None] + bucket_sizes[:, None, :] <= bucket_cap
        best_merge_costs, best = merge_costs.masked_fill(~allowed, float("inf")).min(2)
        del gram, merged_alignments, merge_costs, allowed
        # A pair that is each other's cheapest merges into its lower pool
        mutual = (best_merge_costs < float("inf")) & (best.gather(1, best) == index) & (index < best)
        # A block merges no further than its count, its cheapest pairs first
        rank = best_merge_costs.masked_fill(~mutual, float("inf")).argsort(1).argsort(1)
        mutual &= rank < merges_left.clamp_min(0)[:, None]
        if not mutual.any():
            break
        block_ids, kept = mutual.nonzero(as_tuple=True)
        absorbed = best[block_ids, kept]
        key_sums[block_ids, kept] += key_sums[block_ids, absorbed]
        bucket_sizes[block_ids, kept] += bucket_sizes[block_ids, absorbed]
        alive[block_ids, absorbed] = False
        parent[block_ids, absorbed] = kept
    # Every absorbed pool points at the pool it merged into - follow the chain to the surviving one
    for _ in range(math.ceil(math.log2(max(pools, 2))) + 1):
        parent = parent.gather(1, parent)
    number = torch.cumsum(alive.long(), 1) - 1
    return number.gather(1, parent), alive.sum(1) <= bucket_count


def ward_merge(
    pool_keys: torch.Tensor, pool_sizes: torch.Tensor, bucket_count: int, bucket_cap: int
) -> torch.Tensor:
    """Merge each block's pools down to exactly `bucket_count` buckets, or refuse.

    Ward never stops short at a cap of twice the bucket size: while more than `bucket_count` buckets remain
    their mean size is under the average bucket size, and the two smallest sum to at most twice it, so an allowed
    pair is always left. A cap under that can leave a block with none, and that is an error rather than a
    fallback.

    Args:
        pool_keys: Each block's pool keys [G, P, index_dim] fp32.
        pool_sizes: Keys each pool holds [G, P] fp32.
        bucket_count: Buckets a block ends with.
        bucket_cap: Most keys a merged bucket may hold.

    Returns:
        Labels [G, P] int64 - each pool's bucket, numbered from 0 within its block in the order of its first pool.

    Raises:
        ValueError: If the cap is under twice the average bucket size, or a block stops short of its count.
    """
    bucket_size = pool_sizes.sum(1).max().item() / bucket_count
    if bucket_cap < 2 * bucket_size:
        raise ValueError(
            f"the bucket cap must be at least twice the bucket size, {2 * bucket_size:g} - got {bucket_cap}"
        )
    labels, reached = ward_rounds(pool_keys, pool_sizes, bucket_count, bucket_cap)
    if not reached.all():
        raise ValueError(
            f"{int((~reached).sum())} blocks stopped short of {bucket_count} buckets under the cap"
        )
    return labels


def members_from_labels(labels: torch.Tensor, bucket_count: int, bucket_cap: int) -> torch.Tensor:
    """List each bucket's keys, in position order, from the bucket of every key.

    Args:
        labels: The bucket of each key [T] int64, from 0 to `bucket_count` - 1.
        bucket_count: Buckets in the block.
        bucket_cap: Row width, the most keys a bucket holds.

    Returns:
        Bucket members [bucket_count, bucket_cap] int32 - block-local positions, -1 past a bucket's size.
    """
    members = torch.full((bucket_count, bucket_cap), -1, dtype=torch.int32)
    for bucket in range(bucket_count):
        keys = (labels == bucket).nonzero().flatten()
        members[bucket, : keys.numel()] = keys.to(torch.int32)
    return members
