"""A score packed with its id in one int64, so that one sort or top-k orders by score and breaks a tie by id.

The score's order-preserving bits ride in the high word and `2^31 - 1 - id` in the low, so the greater key is the
greater score and, at an equal score, the lower id. The result does not depend on the order a set of keys was
produced in. The empty key is the least int64.

This is the one place the float-to-ordered-int conversion is written - the kernels call `ordered_bits` and the
torch side calls `ordered_int32`, which are the same map.
"""

import torch
import triton
import triton.language as tl

EMPTY = torch.iinfo(torch.int64).min


@triton.jit
def ordered_bits(score: tl.tensor) -> tl.tensor:
    """An fp32 score as an int32 of the same order - negatives flip their magnitude bits."""
    bits = score.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def pack_key(score: tl.tensor, bucket: tl.tensor) -> tl.tensor:
    """A score and its bucket as one int64 key - the score in the high word, the bucket reversed in the low."""
    return (ordered_bits(score).to(tl.int64) << 32) | (0x7FFFFFFF - bucket).to(tl.int64)


@triton.jit
def key_bucket(key: tl.tensor) -> tl.tensor:
    """The bucket of a packed key as an int32 - undefined for the empty key."""
    return 0x7FFFFFFF - (key & 0xFFFFFFFF).to(tl.int32)


def ordered_int32(scores: torch.Tensor) -> torch.Tensor:
    """Map fp32 scores to int32 of the same order, as `ordered_bits` does on the device.

    Args:
        scores: Scores [...] fp32.

    Returns:
        Their order-preserving bits [...] int32.
    """
    bits = scores.contiguous().view(torch.int32)
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def pack_keys(scores: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Pack scores with their ids so that one int64 orders by score and breaks a tie toward the lower id.

    Args:
        scores: Scores [...] fp32.
        ids: Their ids [...] int32, -1 for none.

    Returns:
        Packed keys [...] int64, the empty key where an id is -1.
    """
    keys = (ordered_int32(scores).to(torch.int64) << 32) | (0x7FFFFFFF - ids.to(torch.int64))
    return torch.where(ids >= 0, keys, EMPTY)


def unpack_scores(keys: torch.Tensor) -> torch.Tensor:
    """Unpack the scores of packed keys - the empty key comes back as some score, to be masked.

    Args:
        keys: Packed keys [...] int64.

    Returns:
        Their scores [...] fp32.
    """
    ordered = (keys >> 32).to(torch.int32)
    bits = torch.where(ordered < 0, ordered ^ 0x7FFFFFFF, ordered)
    return bits.view(torch.float32)


def unpack_ids(keys: torch.Tensor) -> torch.Tensor:
    """Unpack the ids of packed keys.

    Args:
        keys: Packed keys [...] int64.

    Returns:
        Their ids [...] int32, -1 for the empty key.
    """
    ids = (0x7FFFFFFF - (keys & 0xFFFFFFFF)).to(torch.int32)
    return torch.where(keys == EMPTY, -1, ids)
