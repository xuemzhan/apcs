"""MLP mapper：per-head 非线性映射（2 层 MLP + GELU），回答"映射是否非线性"。

设计动机（v1.3 审稿发现）：
    线性/仿射 mapper 的天花板已被规模阶梯双边夹逼（gold 0.26–0.37），
    但没有测试过非线性映射。本模块用每 (层,头) 一个小型 MLP
    （D→hidden→D, GELU）做梯度训练，直接回答"映射是否非线性"。

与 RAT 的区别：RAT 用架构解析核 + 闭式修正（无线性假设之外的自由度）；
MLP mapper 是纯数据驱动的非线性函数逼近器，不使用任何架构先验——
如果 MLP 也无法超越 affine，则非线性不是瓶颈。

训练：校准数据上最小化 ||MLP(V_t) − V_s||²（Adam，全批量或小批量）。
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class MLPMapper:
    """Per-head 2-layer MLP mapper（非线性函数逼近器，梯度训练）。

    参数：
        hidden: MLP 隐层维度（默认 64，约 D/2）
        epochs: 训练 epoch 数
        lr: Adam 学习率
        weight_decay: AdamW 权重衰减（防过拟合）
    """

    def __init__(
        self,
        lam: float = 1e-4,
        hidden: int = 64,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        batch_size: int = 256,
        seed: int = 0,
    ):
        self.lam = lam
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.W: dict[tuple[str, int, int], np.ndarray] = {}  # 兼容接口
        self._models: dict[tuple[str, int, int], Any] = {}

    @property
    def n_params(self) -> int:
        import torch
        return int(sum(sum(p.numel() for p in m.parameters()) for m in self._models.values()))

    def fit_batch(
        self,
        samples: list[tuple[np.ndarray, np.ndarray]],
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> None:
        """梯度训练每 (层,头) 的 MLP。"""
        import torch
        import torch.nn as nn

        if not samples:
            raise ValueError("MLPMapper.fit_batch: samples 不能为空")
        kv_s0 = samples[0][1]
        L_s, _, H, D = kv_s0.shape
        L_t = samples[0][0].shape[0]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        for s in range(L_s):
            teachers = layer_map[s] if s < len(layer_map) else [min(s, L_t - 1)]
            for h in range(H):
                src_all, tgt_all = [], []
                for kv_t, kv_s in samples:
                    S_i = min(kv_t.shape[1], kv_s.shape[1])
                    pos = (
                        np.arange(S_i, dtype=np.float64)
                        if positions is None else positions[:S_i]
                    )
                    for t in teachers:
                        k_raw = kv_t[t, :S_i]
                        if de_rope_fn is not None and kv_kind == "K":
                            from .math import _apply_or_skip
                            k_raw = _apply_or_skip(de_rope_fn, k_raw, pos)
                        src_all.append(k_raw[:, h, :].astype(np.float32))
                        tgt_all.append(kv_s[s, :S_i, h, :].astype(np.float32))

                x = np.concatenate(src_all)
                y = np.concatenate(tgt_all)
                x_t = torch.tensor(x, device=device)
                y_t = torch.tensor(y, device=device)

                torch.manual_seed(self.seed * 1_000_003 + s * 131 + h)
                model = nn.Sequential(
                    nn.Linear(D, self.hidden), nn.GELU(),
                    nn.Linear(self.hidden, self.hidden), nn.GELU(),
                    nn.Linear(self.hidden, D),
                ).to(device)
                opt = torch.optim.AdamW(
                    model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                loss_fn = nn.MSELoss()

                model.train()
                n = x_t.shape[0]
                for epoch in range(self.epochs):
                    if n <= self.batch_size:
                        pred = model(x_t)
                        loss = loss_fn(pred, y_t)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    else:
                        perm = torch.randperm(n, device=device)[: self.batch_size]
                        pred = model(x_t[perm])
                        loss = loss_fn(pred, y_t[perm])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                model.eval()
                self._models[(kv_kind, s, h)] = model
                with torch.no_grad():
                    self.W[(kv_kind, s, h)] = model(x_t[:1]).cpu().numpy()[0]  # 占位

        logger.info("[MLP] fit complete for %s: %d heads, %d params total",
                    kv_kind, L_s * H, self.n_params)

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        from .math import _require_kv_kind, _apply_or_skip

        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "MLPMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)

        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        out = np.zeros((L_s, S, H, D), dtype=np.float32)

        for s in range(L_s):
            teachers = layer_map[s] if s < len(layer_map) else [min(s, L_t - 1)]
            acc = np.zeros((S, H, D), dtype=np.float64)
            for t in teachers:
                k_raw = kv_t[t]
                if de_rope_fn is not None and kv_kind == "K":
                    k_raw = _apply_or_skip(de_rope_fn, k_raw, positions)
                for h in range(H):
                    key = (kv_kind, s, h)
                    if key not in self._models:
                        continue
                    x_t = torch.tensor(
                        k_raw[:, h, :].astype(np.float32), device=device
                    )
                    with torch.no_grad():
                        pred = self._models[key](x_t).cpu().numpy()
                    acc[:, h, :] += pred
            out[s] = (acc / len(teachers)).astype(np.float32)
        return out
