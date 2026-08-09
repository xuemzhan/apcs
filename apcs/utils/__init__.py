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

说明：配置加载 / ${...} 占位符展开（load_config）与 Run 产物落盘
（write_json / write_text）位于 `apcs.io` 包；§64/§65 的版本与
commit 等元数据管理见 `apcs.io.metadata`，均不在本模块。
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
    random.seed(seed)          # Python 内建随机
    np.random.seed(seed)       # NumPy 随机（数值实验主路径）
    # 固定 hash 种子：dict/set 迭代顺序也变为确定性（见 docstring）
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # 多卡统一种子（覆盖所有 GPU）
    except ImportError:
        pass  # 无 torch 环境（CI / 纯 numpy）静默跳过，不破坏可移植性


@contextlib.contextmanager
def cuda_sync() -> Iterator[None]:
    """§49 timing sync：若 torch+CUDA 可用则做 sync；否则 no-op。"""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 排空 GPU 队列：进入 with 块时同步一次
    except ImportError:
        pass
    yield  # 退出时不做二次同步；需要精确前后边界的场景请用 _TimeBlock


class _TimeBlock:
    """§49 timing block：进入时 sync，退出时 sync 并计算 elapsed。"""

    def __init__(self, sync: bool) -> None:
        self.sync = sync        # 是否在进入/退出时做 CUDA sync（§49 要求）
        self.elapsed: float = 0.0

    def __enter__(self) -> "_TimeBlock":
        # §49 计时起点：先 sync 再取时间，把 GPU 排队延迟从计时中排除
        if self.sync:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self._t0 = time.perf_counter()  # perf_counter 单调时钟，不受系统时间调整影响
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # §49 计时终点：先 sync（确保 GPU kernel 真正执行完）再取时间
        if self.sync:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        # 即便 with 块抛异常，__exit__ 也必然执行 → elapsed 照常可用
        self.elapsed = time.perf_counter() - self._t0


def time_block(sync: bool = True):
    """返回一个 §49 风格的计时 context manager。

    `with time_block() as tb: ...` 后读 `tb.elapsed` 得到秒数；
    默认 sync=True（GPU 实验需要前后边界同步）；
    纯 CPU 场景可传 sync=False，省去逐次 import torch 的检查开销。
    """
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
    # 线性插值：k 是 [0, n-1] 上的浮点索引；f 是其整数部分（下取整）
    k = (len(s) - 1) * p
    f = int(k)
    # c 是上取整邻居；min 防止 p=1.0 时越界
    c = min(f + 1, len(s) - 1)
    if f == c:  # 恰好落在某个数据点上（含 p=0.0 / p=1.0 端点）
        return s[f]
    # 两点之间线性插值：s[f] + frac * (s[c] - s[f])
    return s[f] + (s[c] - s[f]) * (k - f)