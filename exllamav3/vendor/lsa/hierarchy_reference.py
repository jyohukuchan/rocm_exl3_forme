"""The hierarchy in plain torch - node bucket levels over the buckets, and a beam down them - correct on a CPU.

The specification of fixed and tree, as `reference` is of flat, and the oracle the served hierarchy and its
beam are tested against. `build_hierarchy` pairs every full node cluster block of every level: a block's buckets
paired by the Ward merge, then the pairs, into node buckets of exactly `node_bucket_size` children, whose centroid is
the normalized sum of every index key beneath them. A level's node buckets are numbered block after block, each
block's in the order of its first child, and a level above pairs them in that order.

`read_hierarchy` reads each query over the levels as they stood when its page was read. A server grows them at page
ends only, and only once a growth's worth of buckets waits, so a query sees the node buckets of the blocks taken by
then - every one of its visible buckets past them is scored directly, as are the node buckets no level above covers
yet. The beam starts at the start level, keeps the best `beam_width` a level and picks the `top_p` best buckets
among the last beam's children and the uncovered ones, and the read expands them as flat's does.
Node bucket shapes and fp8 node centroids are not specified here.

Two research options are not the design's and no server runs them, so a grouping and a score can be compared on one
code path. `node_cluster_algorithm` `position` pairs a block's adjacent node buckets in their order, where Ward pairs
the closest - with `build_buckets`' `cluster_algorithm` `position` that is a pyramid of positional blocks. And
`node_bucket_score` and `bucket_score` `lse` rank a node bucket by the LogSumExp of the query's scores of its
children's mean keys and a bucket by that of its own keys, read at the map's loss temperature `tau`, where the design
ranks both by the centroid.
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .reference import CLUSTER_ALGORITHMS, Buckets, check_unit_rows, expand_buckets, read
from .ward_reference import ward_rounds

# What ranks a node bucket or a bucket - its centroid, or a LogSumExp over what it holds as a research option
BUCKET_SCORES = ("centroid", "lse")


class Hierarchy(NamedTuple):
    """The node bucket levels over a cache's buckets, every full node cluster block of every level paired."""

    # Each level's node centroids [N, index_dim] fp32, level 1 first
    node_centroids: list[torch.Tensor]
    # Each level's children [N, node_bucket_size] int64 - bucket rows under level 1, the level below's rows above
    children: list[torch.Tensor]
    # Each level's mean index key [N, index_dim] fp32 - the key sum over the keys beneath
    node_means: list[torch.Tensor]
    # Each bucket's mean index key [rows, index_dim] fp32
    bucket_means: torch.Tensor
    # Children a node bucket holds, at every level
    node_bucket_size: int
    # Level - 1 buckets node buckets are formed over, a level
    node_cluster_block_sizes: tuple[int, ...]


class Growth(NamedTuple):
    """What growing at every page end up to some bucket count leaves - the counts of a hierarchy as a query sees it."""

    # Buckets under a level 1 node bucket, the first ones
    covered_buckets: int
    # Node buckets made a level, the first ones of each
    made: list[int]
    # Node buckets a level that a node bucket above covers, the first ones of each
    closed: list[int]


def per_level(value: int | Sequence[int], top_level: int) -> tuple[int, ...]:
    """Return a knob's value at each level from 1 to `top_level` - a scalar is every level's."""
    values = (value,) * top_level if isinstance(value, int) else tuple(value)
    if len(values) != top_level:
        raise ValueError(f"a per-level knob takes one value or {top_level} - got {len(values)}")
    return values


def pair_into_node_buckets(
    key_sums: torch.Tensor, node_bucket_size: int, node_cluster_algorithm: str = "ward"
) -> torch.Tensor:
    """Each bucket's node bucket [G, P] int64 - the buckets paired by the Ward merge, then the pairs, to exactly the size.

    Args:
        key_sums: Each block's buckets' key sums [G, P, index_dim] fp32, P a multiple of `node_bucket_size`.
        node_bucket_size: Children a node bucket holds, a power of two.
        node_cluster_algorithm: `ward`, or `position` to take each run of `node_bucket_size` adjacent buckets instead.

    Returns:
        Each bucket's node bucket within its block, numbered from 0 in the order of its first child.
    """
    blocks, buckets, index_dim = key_sums.shape
    labels = torch.arange(buckets).expand(blocks, buckets)
    if node_cluster_algorithm == "position":
        return labels // node_bucket_size
    for _ in range(node_bucket_size.bit_length() - 1):
        count = key_sums.shape[1]
        # Pairs - a merge to half the buckets, at most two a bucket
        pairs = ward_rounds(key_sums, torch.ones(blocks, count), count // 2, 2)[0]
        labels = pairs.gather(1, labels)
        key_sums = torch.zeros(blocks, count // 2, index_dim).scatter_add_(
            1, pairs[..., None].expand_as(key_sums), key_sums
        )
    return labels


def build_hierarchy(
    index_keys: torch.Tensor,
    buckets: Buckets,
    top_level: int,
    node_bucket_size: int,
    node_cluster_block_size: int | Sequence[int],
    node_cluster_algorithm: str = "ward",
) -> Hierarchy:
    """Pair every full node cluster block of every level into node buckets, from the buckets up to `top_level`.

    Args:
        index_keys: The cache's index keys [T, index_dim] fp32 or bf16, unit rows, the buckets' own.
        buckets: The buckets, from `build_buckets`.
        top_level: The highest level, 1 for fixed.
        node_bucket_size: Children a node bucket holds, a power of two from 2.
        node_cluster_block_size: Level - 1 buckets node buckets are formed over - one for every level or one a level,
            each a multiple of `node_bucket_size`.
        node_cluster_algorithm: `ward`, the design's, or `position` to pair adjacent node buckets, a research option.

    Returns:
        The levels that fill at least one node cluster block, level 1 first.

    Raises:
        ValueError: If the keys are not unit rows, the top level is under 1, the node bucket size is not a power of
            two from 2, a node cluster block size is not a multiple of it, or the node cluster algorithm is unknown.
    """
    check_unit_rows("index_keys", index_keys)
    block_sizes = per_level(node_cluster_block_size, top_level)
    if top_level < 1:
        raise ValueError(f"top_level must be 1 or more - got {top_level}")
    if node_bucket_size < 2 or node_bucket_size & (node_bucket_size - 1):
        raise ValueError(f"node_bucket_size must be a power of two from 2 - got {node_bucket_size}")
    if any(block % node_bucket_size for block in block_sizes):
        raise ValueError(
            f"every node_cluster_block_size must be a multiple of {node_bucket_size} - got {block_sizes}"
        )
    if node_cluster_algorithm not in CLUSTER_ALGORITHMS:
        raise ValueError(
            f"node_cluster_algorithm must be one of {CLUSTER_ALGORITHMS} - got {node_cluster_algorithm!r}"
        )
    keys = index_keys.float()
    members = buckets.bucket_members.long()
    key_sums = (keys[members.clamp_min(0)] * (members >= 0)[..., None]).sum(1)
    key_counts = (members >= 0).sum(1).float()
    bucket_means = key_sums / key_counts.clamp_min(1)[:, None]
    node_centroids, children, node_means = [], [], []
    for block in block_sizes:
        blocks = key_sums.shape[0] // block
        if not blocks:
            break
        count = blocks * block
        labels = pair_into_node_buckets(
            key_sums[:count].view(blocks, block, -1), node_bucket_size, node_cluster_algorithm
        )
        # Each block's rows in the order of their node buckets, then block after block
        order = (torch.argsort(labels, dim=1, stable=True) + block * torch.arange(blocks)[:, None]).flatten()
        key_sums = key_sums[order].view(-1, node_bucket_size, key_sums.shape[1]).sum(1)
        key_counts = key_counts[order].view(-1, node_bucket_size).sum(1)
        node_centroids.append(F.normalize(key_sums, dim=-1))
        node_means.append(key_sums / key_counts.clamp_min(1)[:, None])
        children.append(order.view(-1, node_bucket_size))
    return Hierarchy(node_centroids, children, node_means, bucket_means, node_bucket_size, block_sizes)


def growth_at(hierarchy: Hierarchy, closed_buckets: int, page_buckets: int, growth_buckets: int) -> Growth:
    """Replay growing at every page end up to `closed_buckets`, from the counts alone.

    At a page end a level takes its full node cluster blocks of uncovered node buckets - the buckets at level 1, once
    `growth_buckets` of them wait - and each level above does the same over the node buckets made below it.

    Args:
        hierarchy: The levels, from `build_hierarchy`.
        closed_buckets: Buckets closed when the query's page was read, a page end's.
        page_buckets: Buckets a page holds.
        growth_buckets: Buckets level 1 takes at once at least.

    Returns:
        The covered buckets and each level's node buckets made and closed.
    """
    top_level = len(hierarchy.node_cluster_block_sizes)
    size, first_block = hierarchy.node_bucket_size, hierarchy.node_cluster_block_sizes[0]
    covered, made, closed = 0, [0] * top_level, [0] * top_level
    for end in range(page_buckets, closed_buckets + 1, page_buckets):
        blocks = (end - covered) // first_block
        if blocks < max(1, growth_buckets // first_block):
            continue
        covered += blocks * first_block
        made[0] += blocks * first_block // size
        for level in range(1, top_level):
            block = hierarchy.node_cluster_block_sizes[level]
            taken = (made[level - 1] - closed[level - 1]) // block * block
            if not taken:
                break
            closed[level - 1] += taken
            made[level] += taken // size
    return Growth(covered, made, closed)


def read_hierarchy(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    beam_width: int | Sequence[int],
    start_nodes: int | None,
    top_p: int,
    top_k: int,
    sink_size: int,
    window_size: int,
    page_size: int,
    growth_buckets: int,
    node_bucket_score: str = "centroid",
    bucket_score: str = "centroid",
    tau: float = 1.0,
    index_keys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the keys each query reads, as `read_hierarchy_counted` does, without the count of vectors it scored.

    Args:
        index_queries: One index query a row [Tq, index_dim] fp32 or bf16, unit rows.
        positions: Each query's position [Tq] int32 or int64.
        buckets: The buckets of the cache, from `build_buckets`.
        hierarchy: Their levels, from `build_hierarchy`.
        beam_width: Node buckets a query keeps - one for every level or one a level.
        start_nodes: The most node buckets of the level a beam starts at, or None to start at the highest.
        top_p: Buckets a query picks.
        top_k: Long-range keys a query reads at most.
        sink_size: First positions always read.
        window_size: Recent positions always read, the query's own included.
        page_size: Keys a page holds - levels grow at its ends, and a query reads them as they stood at its page's start.
        growth_buckets: Buckets level 1 takes at once at least.
        node_bucket_score: `centroid`, the design's, or `lse` over the children's mean keys, a research option.
        bucket_score: `centroid`, the design's, or `lse` over the bucket's own keys, a research option.
        tau: The map's loss temperature, which an `lse` score divides the query's scores by before the LogSumExp.
        index_keys: The cache's index keys [T, index_dim], the buckets' own - needed by the `lse` bucket score only.

    Returns:
        Selected key ids [Tq, top_k + window_size + sink_size] int32 - the expansion, then the window, then the
        sinks, -1 padded.
        Keys each row holds [Tq] int32.
    """
    selected, lengths, _ = read_hierarchy_counted(
        index_queries,
        positions,
        buckets,
        hierarchy,
        beam_width,
        start_nodes,
        top_p,
        top_k,
        sink_size,
        window_size,
        page_size,
        growth_buckets,
        node_bucket_score,
        bucket_score,
        tau,
        index_keys,
    )
    return selected, lengths


def read_hierarchy_counted(
    index_queries: torch.Tensor,
    positions: torch.Tensor,
    buckets: Buckets,
    hierarchy: Hierarchy,
    beam_width: int | Sequence[int],
    start_nodes: int | None,
    top_p: int,
    top_k: int,
    sink_size: int,
    window_size: int,
    page_size: int,
    growth_buckets: int,
    node_bucket_score: str = "centroid",
    bucket_score: str = "centroid",
    tau: float = 1.0,
    index_keys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the keys each query reads - a beam down the levels its page saw, its buckets picked and expanded.

    A query before any node bucket reads flat. Otherwise the beam starts at the lowest level of at most `start_nodes`
    node buckets, scoring every one of them, or at the highest level grown, scoring its node buckets. Each level
    below scores its node buckets no level above covers beside the children of the node buckets kept above, and
    keeps the best `beam_width`. Level 0 scores the query's visible buckets no node bucket covers beside the last
    beam's children, picks the `top_p` best - a tie going to the lower bucket - and expands them as flat's read does.

    Args:
        index_queries: One index query a row [Tq, index_dim] fp32 or bf16, unit rows.
        positions: Each query's position [Tq] int32 or int64.
        buckets: The buckets of the cache, from `build_buckets`.
        hierarchy: Their levels, from `build_hierarchy`.
        beam_width: Node buckets a query keeps - one for every level or one a level.
        start_nodes: The most node buckets of the level a beam starts at, or None to start at the highest.
        top_p: Buckets a query picks.
        top_k: Long-range keys a query reads at most.
        sink_size: First positions always read.
        window_size: Recent positions always read, the query's own included.
        page_size: Keys a page holds - levels grow at its ends, and a query reads them as they stood at its page's start.
        growth_buckets: Buckets level 1 takes at once at least.
        node_bucket_score: `centroid`, the design's, or `lse` over the children's mean keys, a research option.
        bucket_score: `centroid`, the design's, or `lse` over the bucket's own keys, a research option.
        tau: The map's loss temperature, which an `lse` score divides the query's scores by before the LogSumExp.
        index_keys: The cache's index keys [T, index_dim], the buckets' own - needed by the `lse` bucket score only.

    Returns:
        Selected key ids [Tq, top_k + window_size + sink_size] int32 - the expansion, then the window, then the
        sinks, -1 padded.
        Keys each row holds [Tq] int32.
        Vectors each row scored [Tq] int64 - a centroid or a node bucket's mean key a node bucket or bucket scored by
        the centroid, and each child's mean key or member key an `lse` score read.

    Raises:
        ValueError: As `read` does, if the page is not a whole number of cluster blocks, a score is unknown, the
            `tau` is not positive, or the `lse` bucket score has no index keys.
    """
    selected, lengths = read(index_queries, positions, buckets, top_p, top_k, sink_size, window_size)
    if page_size % buckets.cluster_block_size:
        raise ValueError(
            f"page_size {page_size} must be a multiple of cluster_block_size {buckets.cluster_block_size}"
        )
    if node_bucket_score not in BUCKET_SCORES or bucket_score not in BUCKET_SCORES:
        raise ValueError(
            f"node_bucket_score and bucket_score must be one of {BUCKET_SCORES} - got {node_bucket_score!r}, {bucket_score!r}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be positive - got {tau}")
    if bucket_score == "lse" and index_keys is None:
        raise ValueError("the lse bucket score reads the buckets' keys - pass index_keys")
    scorer = Scorer(buckets, hierarchy, node_bucket_score, bucket_score, tau, index_keys)
    beam_widths = per_level(beam_width, len(hierarchy.node_cluster_block_sizes))
    page_buckets = page_size // buckets.cluster_block_size * buckets.block_bucket_count
    queries = index_queries.float()
    scored = torch.zeros(len(positions), dtype=torch.int64)
    for row, position in enumerate(positions.tolist()):
        scorer.scored = 0
        growth = growth_at(hierarchy, position // page_size * page_buckets, page_buckets, growth_buckets)
        picked = beam_pick(queries[row], position, buckets, growth, beam_widths, start_nodes, top_p, scorer)
        keys = expand_buckets(picked, position, buckets, top_k, sink_size, window_size)
        selected[row] = -1
        selected[row, : keys.numel()] = keys
        lengths[row] = keys.numel()
        scored[row] = scorer.scored
    return selected, lengths, scored


class Scorer:
    """One query's score of a level's node buckets or of buckets, by the centroid or by a LogSumExp."""

    def __init__(
        self,
        buckets: Buckets,
        hierarchy: Hierarchy,
        node_bucket_score: str,
        bucket_score: str,
        tau: float,
        index_keys: torch.Tensor | None,
    ) -> None:
        """Hold what the scores read.

        Args:
            buckets: The buckets of the cache.
            hierarchy: Their levels.
            node_bucket_score: `centroid` or `lse`.
            bucket_score: `centroid` or `lse`.
            tau: The map's loss temperature, which an `lse` score divides the query's scores by.
            index_keys: The cache's index keys [T, index_dim], or None when no bucket is scored by `lse`.
        """
        self.buckets = buckets
        self.hierarchy = hierarchy
        self.node_bucket_score = node_bucket_score
        self.bucket_score = bucket_score
        self.tau = tau
        self.index_keys = None if index_keys is None else index_keys.float()
        # Vectors scored since the last reset - one query's cost
        self.scored = 0

    def node_buckets(self, query: torch.Tensor, level: int, rows: torch.Tensor) -> torch.Tensor:
        """Return the query's score of level-`level` node buckets [N] fp32 - query [index_dim] fp32, rows [N] int64."""
        if self.node_bucket_score == "centroid":
            self.scored += len(rows)
            return self.hierarchy.node_centroids[level - 1][rows] @ query
        children = self.hierarchy.children[level - 1][rows]
        self.scored += children.numel()
        means = self.hierarchy.bucket_means if level == 1 else self.hierarchy.node_means[level - 2]
        return torch.logsumexp(means[children] @ query / self.tau, dim=1)

    def bucket_rows(self, query: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Return the query's score of buckets [N] fp32 - query [index_dim] fp32, rows [N] int64."""
        if self.bucket_score == "centroid" or self.index_keys is None:
            self.scored += len(rows)
            return self.buckets.bucket_centroids[rows] @ query
        members = self.buckets.bucket_members[rows].long()
        self.scored += int((members >= 0).sum())
        logits = (self.index_keys[members.clamp_min(0)] @ query / self.tau).masked_fill(
            members < 0, -torch.inf
        )
        return torch.logsumexp(logits, dim=1)


def beam_pick(
    query: torch.Tensor,
    position: int,
    buckets: Buckets,
    growth: Growth,
    beam_widths: tuple[int, ...],
    start_nodes: int | None,
    top_p: int,
    scorer: Scorer,
) -> torch.Tensor:
    """Pick one query's buckets by a beam down the levels its page saw - every visible bucket before any level.

    Args:
        query: The query's index query [index_dim] fp32.
        position: The query's position.
        buckets: The buckets of the cache.
        growth: The levels as the query's page saw them.
        beam_widths: Node buckets kept a level.
        start_nodes: The most node buckets of the start level, or None.
        top_p: Buckets the query picks.
        scorer: The node bucket and bucket scores.

    Returns:
        The picked bucket rows [<= top_p] int64, best first.
    """
    levels = sum(1 for count in growth.made if count)
    start = levels
    if start_nodes is not None:
        start = next(
            (level for level in range(1, levels + 1) if growth.made[level - 1] <= start_nodes), levels
        )
    carried = torch.empty(0, dtype=torch.int64)
    for level in range(start, 0, -1):
        made, closed = growth.made[level - 1], growth.closed[level - 1]
        # The start level whole when a level above covers some of it, else its open node buckets and the carried
        first_open = 0 if level == start and start_nodes is not None else closed
        candidates = torch.cat([carried, torch.arange(first_open, made)]).sort().values
        kept = best(scorer.node_buckets(query, level, candidates), candidates, beam_widths[level - 1])
        carried = scorer.hierarchy.children[level - 1][kept].flatten()
    visible = min(
        position // buckets.cluster_block_size * buckets.block_bucket_count, len(buckets.bucket_sizes)
    )
    uncovered = torch.arange(growth.covered_buckets, max(visible, growth.covered_buckets))
    candidates = torch.cat([carried, uncovered]).sort().values
    return best(scorer.bucket_rows(query, candidates), candidates, top_p)


def best(scores: torch.Tensor, rows: torch.Tensor, count: int) -> torch.Tensor:
    """Return the `count` rows of the highest scores [<= count] int64, best first, a tie to the lower row."""
    return rows[torch.argsort(scores, descending=True, stable=True)[:count]]
