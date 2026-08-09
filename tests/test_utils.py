"""apcs 公共工具测试（utils.percentile 分位数语义 + set_seed 种子可复现性）。"""
from __future__ import annotations

from apcs.utils import percentile, set_seed


def test_percentile_basic():
    """percentile：线性插值语义（0.5→中位均值），空列表安全返回 0。"""
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.0) == 1.0
    assert percentile([1, 2, 3, 4], 1.0) == 4.0
    assert percentile([], 0.5) == 0.0


def test_set_seed_runs():
    """set_seed：同 seed 复位后 numpy 随机序列逐位一致（§51 可复现性）。"""
    # 只验证不抛异常 + numpy 种子生效
    set_seed(42)
    import numpy as np

    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert (a == b).all()