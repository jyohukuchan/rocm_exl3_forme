"""Bucket shapes in plain torch - a bucket's spread about its centroid, and the estimated best key it scores by.

A shape is the covariance of a bucket's index keys about its centroid, kept as its `rank` leading directions with
their variances and one floor - the rest of the trace spread evenly over the directions left out. It is exact on
the trace and approximate only in how it splits the variance past `rank`, so a score reads `1 + rank` products a
bucket and no key.

A shaped bucket is scored by its estimated best key, `q . c + z sqrt(q^T Sigma q)`. With a unit query, `||q||^2 = 1`
splits between the span of the directions and the rest, which carries the floor:

    q^T Sigma q = sum_j lambda_j (q . u_j)^2 + floor (1 - sum_j (q . u_j)^2)

The served shape keeps one sign bit a dimension of each direction, `+-1 / sqrt(d)`, which is unit length whatever
the direction was, and orders buckets no worse than the exact directions do. A direction's sign is arbitrary, and the
score reads only its square, so a direction and its negation score alike.

This is the oracle the shape kernels are tested against.
"""

from typing import NamedTuple

import torch


class Shapes(NamedTuple):
    """The shapes of some buckets, one row each, aligned with their centroids."""

    # Sign directions [m, rank, index_dim] fp32, every entry +-1 / sqrt(d), leading first
    basis: torch.Tensor
    # Variance along each kept direction [m, rank] fp32, descending
    variances: torch.Tensor
    # Average variance of the index_dim - rank directions left out [m] fp32
    floor: torch.Tensor


def covariances(
    index_keys: torch.Tensor, centroids: torch.Tensor, members: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each bucket's covariance of its keys about its centroid, over its own size.

    Args:
        index_keys: Every index key [T, index_dim] fp32 or bf16, unit rows.
        centroids: The buckets' centroids [m, index_dim] fp32.
        members: Each bucket's key positions [m, slots] int32 or int64, -1 past its size.

    Returns:
        Covariances [m, index_dim, index_dim] fp32 - all zero for an empty bucket.
        Keys each bucket holds [m] fp32, at least 1.
    """
    held = members >= 0
    keys = index_keys.float()[members.clamp_min(0).long()]
    residuals = (keys - centroids.float()[:, None]) * held[..., None]
    sizes = held.sum(1).clamp_min(1).float()
    return residuals.transpose(1, 2) @ residuals / sizes[:, None, None], sizes


def leading_directions(covariance: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each covariance's `rank` leading eigen-directions and their variances, by an exact eigendecomposition.

    Args:
        covariance: Symmetric covariances [m, index_dim, index_dim] fp32.
        rank: Directions kept.

    Returns:
        Unit directions [m, rank, index_dim] fp32, leading first.
        Their variances [m, rank] fp32, descending and never negative.
    """
    values, vectors = torch.linalg.eigh(covariance.double())
    kept = values[:, values.shape[1] - rank :].flip(-1).clamp_min(0)
    directions = vectors[:, :, vectors.shape[2] - rank :].flip(-1).transpose(1, 2)
    return directions.float(), kept.float()


def sign_directions(directions: torch.Tensor) -> torch.Tensor:
    """Round directions to one sign bit a dimension, `+-1 / sqrt(d)`, a zero entry to the negative side.

    Args:
        directions: Directions [..., index_dim] fp32.

    Returns:
        The sign directions [..., index_dim] fp32, unit rows.
    """
    return ((directions > 0).float() * 2 - 1) / directions.shape[-1] ** 0.5


def build_shapes(
    index_keys: torch.Tensor, centroids: torch.Tensor, members: torch.Tensor, rank: int
) -> Shapes:
    """Shape every bucket of a table - its covariance's leading directions as signs, their variances and a floor.

    Args:
        index_keys: Every index key [T, index_dim] fp32 or bf16, unit rows.
        centroids: The buckets' centroids [m, index_dim] fp32, the normalized sum of their keys.
        members: Each bucket's key positions [m, slots], -1 past its size.
        rank: Directions kept a bucket, at most index_dim.

    Returns:
        The buckets' shapes - an empty bucket's all zero but its signs.
    """
    covariance, _ = covariances(index_keys, centroids, members)
    index_dim = covariance.shape[-1]
    trace = covariance.diagonal(dim1=1, dim2=2).sum(1)
    directions, variances = leading_directions(covariance, rank)
    floor = ((trace - variances.sum(1)) / max(index_dim - rank, 1)).clamp_min(0)
    return Shapes(sign_directions(directions), variances, floor)


def spread(index_queries: torch.Tensor, shapes: Shapes) -> torch.Tensor:
    """Estimate the standard deviation of each bucket's key scores under each query.

    Args:
        index_queries: Index queries [R, index_dim] fp32, unit rows.
        shapes: The shapes of `m` buckets.

    Returns:
        Estimated standard deviations [R, m] fp32, never negative.
    """
    projections = torch.einsum("rd,mkd->rmk", index_queries.float(), shapes.basis).square()
    variance = (projections * shapes.variances[None]).sum(-1) + shapes.floor[None] * (1 - projections.sum(-1))
    return variance.clamp_min(0).sqrt()


def shaped_scores(
    index_queries: torch.Tensor, centroids: torch.Tensor, shapes: Shapes, deviations: float
) -> torch.Tensor:
    """Score each bucket by its estimated best key, `q . c + z sqrt(q^T Sigma q)`.

    Args:
        index_queries: Index queries [R, index_dim] fp32, unit rows.
        centroids: The buckets' centroids [m, index_dim] fp32.
        shapes: Their shapes.
        deviations: Standard deviations added, $z$.

    Returns:
        Scores [R, m] fp32.
    """
    return index_queries.float() @ centroids.float().T + deviations * spread(index_queries, shapes)


def pack_shape_basis(basis: torch.Tensor) -> torch.Tensor:
    """Pack a basis into one bit a dimension, set where the entry is positive.

    Bit `j` of byte `i` holds dimension `8 i + j`, so the bytes read as little-endian words put dimension `32 w + j`
    at bit `j` of word `w` - the order the kernels read.

    Args:
        basis: Directions [m, rank, index_dim] fp32, `index_dim` a multiple of 8.

    Returns:
        The packed basis [m, rank, index_dim / 8] uint8.
    """
    index_dim = basis.shape[-1]
    weights = 2 ** torch.arange(8, device=basis.device, dtype=torch.int32)
    grouped = (basis > 0).view(*basis.shape[:-1], index_dim // 8, 8).int()
    return (grouped * weights).sum(-1).to(torch.uint8)


def unpack_shape_basis(packed: torch.Tensor) -> torch.Tensor:
    """Expand a packed basis back to its sign directions.

    Args:
        packed: The packed basis [m, rank, index_dim / 8] uint8.

    Returns:
        Sign directions [m, rank, index_dim] fp32, every entry `+-1 / sqrt(d)`.
    """
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits = (packed.int()[..., None] >> shifts) & 1
    signs = bits.flatten(-2).float() * 2 - 1
    return signs / signs.shape[-1] ** 0.5
