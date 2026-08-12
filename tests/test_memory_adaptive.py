"""测试 apcs.utils.memory 自适应默认值。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apcs.utils.memory import (
    _available_memory_bytes,
    memory_status,
    safe_calibration_defaults,
)


def test_available_memory_returns_positive():
    """无论哪个平台，available_memory_bytes 都应当 > 0。"""
    assert _available_memory_bytes() > 0


def test_memory_status_keys():
    assert set(memory_status().keys()) == {"available_mb", "budget_mb"}
    assert memory_status()["available_mb"] > 0


def test_safe_defaults_in_range_at_low_memory():
    """模拟 1 GB 可用：n_calib 应处于 [8, 128] 的安全区间。"""
    info = safe_calibration_defaults(n_t=36, n_s=28, H=8, D=128, seq=512)
    assert 8 <= info["n_calib_estimate"] <= 128
    assert info["seq_estimate"] >= 128


def test_safe_defaults_at_high_memory():
    """模拟 1 TB 可用：n_calib 应立刻达到 128 上限。"""
    # 直接调用，避免真的用 1 TB
    global _available_memory_bytes
    orig = _available_memory_bytes
    try:
        from apcs.utils import memory as mem
        mem._available_memory_bytes = lambda: 1 << 40  # 1 TB
        info = safe_calibration_defaults(n_t=4, n_s=4, H=2, D=8, seq=64)
        assert info["n_calib_estimate"] == 128, "充裕内存应直接给上限"
    finally:
        mem._available_memory_bytes = orig


def test_safe_defaults_monotonic_in_mem():
    """内存越大，n_calib 估计越大（单调不减）。"""
    from apcs.utils import memory as mem
    orig = mem._available_memory_bytes
    small = safe_calibration_defaults_with(n=1 << 30)
    big = safe_calibration_defaults_with(n=1 << 32)
    assert big["n_calib_estimate"] >= small["n_calib_estimate"]
    mem._available_memory_bytes = orig


def safe_calibration_defaults_with(n: int) -> dict:
    """测试辅助：指定具体可用内存跑一次。"""
    from apcs.utils import memory as mem
    orig = mem._available_memory_bytes
    mem._available_memory_bytes = lambda: n
    try:
        return safe_calibration_defaults(n_t=4, n_s=4, H=2, D=8, seq=64)
    finally:
        mem._available_memory_bytes = orig
