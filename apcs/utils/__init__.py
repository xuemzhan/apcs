"""公共工具：seed 设置、计时、上下文管理（design.md §49）。

═══════════════════════════════════════════════════════════════════════════════
§49 计时规范：
    1. warmup
    2. torch.cuda.synchronize()
    3. ≥10 次正式重复
    4. P50 / P95

§51 统计规范：
    - 3 Seeds
    - Mean / Std
    - 95% CI
    - Paired Bootstrap / Permutation

本模块提供：
    - set_seed: 固定 Python / NumPy / Torch 随机种子
    - cuda_sync: context manager，包装 torch.cuda.synchronize（CPU 时 no-op）
    - time_block: 计时 context manager，返回 elapsed 秒
    - percentile: 轻量 percentile（线性插值）
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import contextlib
import os
import random
import time
from typing import Iterator

import numpy as np


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / (可选) PyTorch 随机种子。

    设计要点：
        - PYTHONHASHSEED 也固定（避免 dict 顺序抖动）
        - 若 CUDA 可用，同时固定所有 GPU 种子
        - 若 torch 不可用，跳过对应步骤
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def cuda_sync() -> Iterator[None]:
    """§49 timing sync：若 torch+CUDA 可用则做 sync；否则 no-op。"""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    yield


class _TimeBlock:
    """§49 timing block：进入时 sync，退出时 sync 并计算 elapsed。"""

    def __init__(self, sync: bool) -> None:
        self.sync = sync
        self.elapsed: float = 0.0

    def __enter__(self) -> "_TimeBlock":
        if self.sync:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.sync:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self.elapsed = time.perf_counter() - self._t0


def time_block(sync: bool = True):
    """返回一个 §49 风格的计时 context manager。"""
    return _TimeBlock(sync)


def percentile(values: list[float], p: float) -> float:
    """轻量 percentile（线性插值，§49 报告 P50 / P95）。

    边界：
        - 空列表 → 0.0
        - p=0.0 → 最小值；p=1.0 → 最大值
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)