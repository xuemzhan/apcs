"""K‖V 联合布局适配器：打通 TorchBackend (L,S,H,2D) 与 per-kind mapper。

════════════════════════════════════════════════════════════════════════════════
◆ D2 修复背景：TorchBackend.forward_prefill 返回 (L, S, H, 2D)（K‖V 末维拼接），
而 RidgePerHeadMapper 等按 (L, S, H, D) 单 kind 设计 —— 直接把联合 KV 喂给
per-kind mapper 会在 einsum 处形状崩溃。此前只有 evaluator 的私有路径
（手动拆 K/V、各建一个 mapper）能跑通，HandoffPipeline + TorchBackend 组合
实际不可运行。

本适配器实现 §22 的 K/V 独立参数化语义在联合布局下的等价形式：
    - fit: 拆 K/V → 分别用 (kv_kind="K"/"V") 拟合独立参数
    - transform: 拆 K/V → 分别变换 → 重新拼接为 (L_s, S, H, 2D)
de-RoPE 只作用于 K（RoPE 不作用于 V，§23）：de_rope_fn 透传给 K mapper，
V mapper 恒传 None。
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class JointKVMapper:
    """(L, S, H, 2D) 联合 KV ↔ per-kind mapper 的适配器。

    参数：
        mapper_k: K 通道 mapper（如 RidgeMapper / RidgePerHeadMapper）
        mapper_v: V 通道 mapper（独立参数，§22）
    """

    def __init__(self, mapper_k: Any, mapper_v: Any):
        self.mapper_k = mapper_k
        self.mapper_v = mapper_v

    @staticmethod
    def _split(kv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L, S, H, 2D) → ((L,S,H,D)_K, (L,S,H,D)_V)。末维必须为偶数。"""
        kv = np.asarray(kv)
        if kv.ndim != 4 or kv.shape[-1] % 2 != 0:
            raise ValueError(
                f"JointKVMapper 期望 (L, S, H, 2D) 联合 KV，got {kv.shape}"
            )
        dt = kv.shape[-1] // 2
        return kv[..., :dt], kv[..., dt:]

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> None:
        """联合 KV 配对拟合：K/V 各自独立参数（§22 禁止共享）。"""
        kt, kv_ = self._split(kv_t)
        st, sv = self._split(kv_s)
        # de-RoPE 只作用于 K（§23）；V 不旋转，恒传 None
        self.mapper_k.fit(kt, st, layer_map, kv_kind="K",
                          positions=positions, de_rope_fn=de_rope_fn)
        self.mapper_v.fit(kv_, sv, layer_map, kv_kind="V",
                          positions=positions, de_rope_fn=None)

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """联合 KV 变换 → (L_s, S, H, 2D)。"""
        kt, kv_ = self._split(kv_t)
        mk = self.mapper_k.transform(kt, layer_map, kv_kind="K",
                                     positions=positions, de_rope_fn=de_rope_fn)
        mv = self.mapper_v.transform(kv_, layer_map, kv_kind="V",
                                     positions=positions, de_rope_fn=None)
        return np.concatenate([mk, mv], axis=-1)

    @property
    def n_params(self) -> int:
        return int(self.mapper_k.n_params + self.mapper_v.n_params)


__all__ = ["JointKVMapper"]
