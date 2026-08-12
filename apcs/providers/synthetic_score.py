"""Synthetic ScoreProvider —— §37 7 仿真方法的得分/决策数据源。

把 `apcs/capability/main.py::` 内部的 `_simulate_scores` / `_simulate_decision`
提炼到 provider 层，runner 不再直接调用仿真函数，便于未来替换为真实 LLM 评分。

数据来源：
    §37 各方法期望分表（teacher=0.80 > full_apcs=0.70 > base_plus_adv=0.66 >
    base_only=0.60 > ridge=0.58 > text=0.55 > student=0.50）。
    评分 = 期望分 + seed 偏移 + 样本噪声。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# 与 capability/main.py 保持一致的偏移幅度（seed 波动量级）。
_OFFSET_SPAN = {
    "student": 0.01,
    "teacher": 0.05,
    "text": 0.01,
    "ridge": 0.01,
    "base_only": 0.01,
    "base_plus_adv": 0.02,
    "full_apcs": 0.02,
}
_BASES = {
    "student": 0.50,
    "teacher": 0.80,
    "text": 0.55,
    "ridge": 0.58,
    "base_only": 0.60,
    "base_plus_adv": 0.66,
    "full_apcs": 0.70,
}
_DECISION_P = {
    "student": 0.55,
    "teacher": 0.85,
    "text": 0.58,
    "ridge": 0.60,
    "base_only": 0.62,
    "base_plus_adv": 0.68,
    "full_apcs": 0.72,
}


def _stable_seed(seed: int, kind: str, sample_id: str = "") -> int:
    """crc32 派生（与 main.py::_stable_seed 同源，保跨进程可复现）。"""
    import zlib

    return zlib.crc32(f"{seed}:{kind}:{sample_id}".encode("utf-8")) & 0x7FFFFFFF


@dataclass
class SyntheticScoreProvider:
    """合成 ScoreProvider（§37 7 方法 + §45 决策）。"""
    kind: str = "synthetic"

    def score(self, method: str, sample_id: str, seed: int) -> float:
        """`method` 在 `sample_id` 上的得分（clip 到 [0,1]）。"""
        if method not in _BASES:
            raise KeyError(f"Unknown method {method!r}")
        offset_rng = np.random.default_rng(_stable_seed(seed, f"offset:{method}", sample_id))
        noise_rng = np.random.default_rng(_stable_seed(seed, f"noise:{method}", sample_id))
        offset = (offset_rng.random() * 2 - 1) * _OFFSET_SPAN[method]
        return float(np.clip(_BASES[method] + offset + noise_rng.normal(0, 0.05), 0, 1))

    def decision(self, method: str, sample_id: str, seed: int) -> int:
        """`method` 在 `sample_id` 上的决策 ID（0/1 伯努利）。"""
        if method not in _DECISION_P:
            raise KeyError(f"Unknown method {method!r}")
        rng = np.random.default_rng(_stable_seed(seed, method, sample_id))
        return int(rng.random() < _DECISION_P[method])

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "synthetic_score",
            "note": (
                "合成得分：表驱动 baseline + seed 偏移 + 噪声。"
                "CHG > 0 由 base 表 (teacher > student) 构造，**非测量结果**。"
            ),
            "bases": dict(_BASES),
            "offsets": dict(_OFFSET_SPAN),
        }


__all__ = ["SyntheticScoreProvider"]
