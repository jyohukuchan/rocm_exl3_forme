"""A design's smaller read budget for its early queries.

This module imports nothing else of the repo's, so the settings, the reference and the kernels share one type.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class EarlyBudget:
    """The `top_k` and `top_p` a query reads while its position is below `position` - the design's own are the late queries'."""

    # First position that reads the design's own budget
    position: int
    # Long-range keys an early query reads at most
    top_k: int
    # Buckets an early query picks for expansion
    top_p: int

    def __post_init__(self) -> None:
        """Reject a field that is not a positive whole number.

        Raises:
            ValueError: `position`, `top_k` or `top_p` is not an integer of at least 1.
        """
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"early_budget {item.name} must be an integer - got {value!r}")
            if value < 1:
                raise ValueError(f"early_budget {item.name} must be at least 1 - got {value}")

    @classmethod
    def read(cls, value: "EarlyBudget | Mapping[str, Any] | None") -> "EarlyBudget | None":
        """Build the knob from what a block holds - a mapping of `position`, `top_k` and `top_p`, or none.

        Args:
            value: What arrived.

        Returns:
            The budget, or None for none.

        Raises:
            ValueError: The value is neither a budget, a mapping nor None, or lacks a key or names one a budget does not have.
        """
        if value is None or isinstance(value, EarlyBudget):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                f"early_budget is a mapping of position, top_k and top_p, or null - got {value!r}"
            )
        names = [item.name for item in fields(cls)]
        if sorted(value) != sorted(names):
            raise ValueError(f"early_budget has the keys {names} - got {sorted(value)}")
        return cls(**value)


def budget_at(position: int, top_k: int, top_p: int, early: EarlyBudget | None) -> tuple[int, int]:
    """Return the `top_k` and `top_p` a query at a position reads.

    Args:
        position: The query's position.
        top_k: The design's own long-range keys.
        top_p: The design's own buckets.
        early: The budget of queries before its position, or None for none.

    Returns:
        The query's `top_k` and `top_p`.
    """
    if early is not None and position < early.position:
        return early.top_k, early.top_p
    return top_k, top_p
