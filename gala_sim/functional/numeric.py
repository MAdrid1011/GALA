"""Small, auditable FP32 helpers shared by functional replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NumericConfig:
    dtype: str
    rounding: str
    flush_subnormal_input: bool
    flush_subnormal_output: bool
    exp_max_relative_error: float
    log_max_absolute_error: float
    rcp_max_relative_error: float
    sqrt_max_relative_error: float

    def __post_init__(self) -> None:
        if self.dtype != "fp32" or self.rounding != "round_to_nearest_even":
            raise ValueError("only the frozen FP32 round-to-nearest-even semantics are supported")
        if min(self.exp_max_relative_error, self.log_max_absolute_error,
               self.rcp_max_relative_error, self.sqrt_max_relative_error) < 0:
            raise ValueError("numeric error bounds must be non-negative")


@dataclass(frozen=True)
class ReductionOrder:
    event_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("reduction order is not a permutation")


def flush_subnormal(value: np.float32) -> np.float32:
    smallest_normal = np.float32(np.finfo(np.float32).tiny)
    if np.abs(value) < smallest_normal:
        return np.float32(0.0)
    return np.float32(value)


def replay_fp32_sum(values: Iterable[np.float32], order: ReductionOrder | None = None,
                    *, flush_input: bool = True, flush_output: bool = True) -> np.float32:
    values = tuple(np.float32(value) for value in values)
    if order is not None and (len(order.event_ids) != len(values)
                              or set(order.event_ids) != set(range(len(values)))):
        raise ValueError("reduction order must be a permutation of contribution indices")
    ordered = values if order is None else tuple(values[index] for index in order.event_ids)
    total = np.float32(0.0)
    for value in ordered:
        term = flush_subnormal(value) if flush_input else value
        total = np.float32(total + term)
        if flush_output:
            total = flush_subnormal(total)
    return total
