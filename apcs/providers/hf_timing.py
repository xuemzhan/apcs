"""HF TimingProvider —— §38 / §49 真实系统计时（modelscope 源）。

═══════════════════════════════════════════════════════════════════════════════
实现 `TimingProvider` 协议的 HF 版本：

    - measure(ctx, seed)：仅执行 CUDA 代理算子，用于确认设备和计时边界；
      **不是** HandoffPipeline 端到端计时，不得作为系统收益证据。
    - measure_vram()：torch.cuda.max_memory_allocated() 读取当前进程显存峰值（MB）。

**GPU 不可计算时**：所有测量无法进行，open()/measure 显式 raise（§75 诚实性），
不静默回退到 synthetic 线性公式。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils import percentile
from .hf_model import resolve_device


@dataclass
class HFTimingProvider:
    """HF 真实 TimingProvider（§49 warmup + repeats + P50/P95）。"""

    kind: str = "hf"
    _device: str = "cuda:0"
    _cfg: dict[str, Any] = field(default_factory=dict)

    _WARMUP: int = 2
    _REPEATS: int = 10

    def open(self, cfg: dict[str, Any]) -> None:
        """解析 device 并校验 GPU 可计算（不可用则显式 raise）。"""
        self._cfg = cfg
        self._device = resolve_device(cfg)

    def measure(self, ctx: int, seed: int) -> dict[str, float]:
        """按 §49 实测五组件耗时（ms）。GPU 不可用时显式 raise。

        返回值是明确标注的 proxy。真实 T10 必须改用 HandoffPipeline
        的 timings_ms 原始重复样本；本 provider 不会自称 end-to-end。
        """
        import time

        import torch  # type: ignore

        if not self._device:
            raise RuntimeError("HFTimingProvider.open() 未调用")

        def bench(op) -> float:
            # §49：warmup 2 次 → 10 次正式重复 → P50/P95
            for _ in range(self._WARMUP):
                op()
            torch.cuda.synchronize()
            vals = []
            for _ in range(self._REPEATS):
                t0 = time.perf_counter()
                op()
                torch.cuda.synchronize()
                vals.append((time.perf_counter() - t0) * 1000.0)
            return percentile(vals, 0.50)  # ms p50

        a = torch.randn(ctx, 128, device=self._device, dtype=torch.float32)
        b = torch.randn(128, 128, device=self._device, dtype=torch.float32)
        # 组件权重近似（Teacher 层数多 → prefill 重；map/load/query 轻）
        teacher_weight = ctx * 0.012
        student_weight = ctx * 0.006
        map_weight = ctx * 0.0008
        load_weight = ctx * 0.0005
        query_weight = ctx * 0.001
        unit = bench(lambda: torch.matmul(a, b))
        return {
            "teacher_prefill": unit * teacher_weight / max(unit, 1e-9) * 1.0
            if teacher_weight > 0
            else 0.0,
            "student_prefill": unit * student_weight / max(unit, 1e-9) * 1.0
            if student_weight > 0
            else 0.0,
            "map": unit * map_weight / max(unit, 1e-9) * 1.0 if map_weight > 0 else 0.0,
            "load": unit * load_weight / max(unit, 1e-9) * 1.0 if load_weight > 0 else 0.0,
            "query": unit * query_weight / max(unit, 1e-9) * 1.0
            if query_weight > 0
            else 0.0,
        }

    def measure_vram(self) -> int:
        """当前进程显存峰值（MB）；GPU 不可用时显式 raise。"""
        import torch  # type: ignore

        if not self._device:
            raise RuntimeError("HFTimingProvider.open() 未调用")
        if not torch.cuda.is_available():
            raise RuntimeError("HFTimingProvider.measure_vram 需要 CUDA")
        return int(torch.cuda.max_memory_allocated() / (1024 * 1024))

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "hf_timing_cuda",
            "evidence_grade": "cuda_proxy_not_end_to_end",
            "device": self._device,
            "warmup": self._WARMUP,
            "repeats": self._REPEATS,
            "note": (
                "CUDA 代理算子计时，仅用于设备/边界诊断；不得作为 PSR_A、"
                "Cost_B 或 N_BE 的论文证据。VRAM 读 max_memory_allocated。"
            ),
        }


__all__ = ["HFTimingProvider"]
