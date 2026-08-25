from __future__ import annotations

import numpy as np
import pytest

from gala_sim.functional import ReductionOrder, flush_subnormal, replay_fp32_sum


def test_fp32_reduction_order_is_explicit() -> None:
    values = (np.float32(1.0), np.float32(2.0), np.float32(3.0))
    assert replay_fp32_sum(values, ReductionOrder((2, 0, 1))) == np.float32(6.0)
    with pytest.raises(ValueError, match="permutation"):
        replay_fp32_sum(values, ReductionOrder((0, 0, 1)))


def test_subnormal_flush_is_deterministic() -> None:
    assert flush_subnormal(np.float32(1e-40)) == np.float32(0.0)
    assert flush_subnormal(np.float32(1e-3)) == np.float32(1e-3)
