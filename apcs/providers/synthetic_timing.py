"""Synthetic TimingProvider —— §38 系统耗时/显存的离线估算。

T10 runner 内部原先的 `_simulate_timings` 提炼到 provider 层。
设计.md §38 明确禁止把理论估算当论文结果，故 provider 测试同步打
offline_demo 标记。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SyntheticTimingProvider:
    """合成 TimingProvider（线性公式，与 system/runner.py::_simulate_timings 一致）。"""
    kind: str = "synthetic"

    def measure(self, ctx: int, seed: int) -> dict[str, float]:
        """返回各组件耗时（ms）。

        斜率与常数项均与原 `_simulate_timings` 保持一致（基线回归
        应当无误），便于接入真实 provider 后直接对比。
        """
        return {
            "teacher_prefill": 8.0 + ctx * 0.012,
            "student_prefill": 3.0 + ctx * 0.006,
            "map": 0.5 + ctx * 0.0008,
            "load": 0.3 + ctx * 0.0005,
            "query": 0.4 + ctx * 0.001,
        }

    def measure_vram(self) -> int:
        """VRAM 量级估算（MB），与 system/runner.py 中公式一致。"""
        # 实际实现调用 torch.cuda.max_memory_allocated；此处仅给占位值
        return 0

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "synthetic_timing_linear",
            "note": (
                "线性公式：teacher_prefill 与(student_prefill) 与 ctx 长度成正比。"
                "真实实验应替换为 HFTimingProvider（§49 warmup + ≥10 repeats + P50/P95）。"
            ),
        }


__all__ = ["SyntheticTimingProvider"]
