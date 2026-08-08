"""apcs 公共工具测试。"""
from __future__ import annotations

from apcs.utils import percentile, set_seed


def test_percentile_basic():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.0) == 1.0
    assert percentile([1, 2, 3, 4], 1.0) == 4.0
    assert percentile([], 0.5) == 0.0


def test_set_seed_runs():
    # 只验证不抛异常 + numpy 种子生效
    set_seed(42)
    import numpy as np

    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert (a == b).all()