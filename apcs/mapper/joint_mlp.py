"""Joint (cross-layer, cross-head) MLP translator — over-parameterized nonlinear audit mapper.

Motivation (round-2/round-3 review): the audited ladder covers per-head linear /
affine maps and a per-head MLP, but not a *jointly structured* nonlinear path
that can read several teacher layers and all heads at once. A skeptical reviewer
can therefore argue that the negative result is an artifact of the restricted
function class. This module closes that objection with a deliberately
over-parameterized translator: for each student layer, one MLP maps the
concatenation of the aligned teacher layers and all KV heads at a position to
all student heads at that position.

The path is trained by gradient descent on the calibration split only and scored
on the disjoint held-out split, under the same audit protocol as every other
family (Section~\ref{sec:method}). It is an oracle-style upper bound within the
audited zero-re-prefill regime, not a deployable translator.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class JointMLPMapper:
    """Cross-layer, cross-head MLP mapper (over-parameterized nonlinear family).

    For each student layer ``s`` and KV kind, the input is the concatenation of
    the de-RoPE'd teacher states over all teacher layers in ``layer_map[s]`` and
    all heads at a given position; the output is the corresponding student state
    over all heads. One fresh MLP is trained per student layer.
    """

    def __init__(
        self,
        lam: float = 1e-4,
        hidden: int = 256,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        n_hidden_layers: int = 2,
        seed: int = 0,
    ):
        self.lam = lam
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.n_hidden_layers = int(n_hidden_layers)
        self.seed = int(seed)
        self.W: dict[tuple[str, int, int], np.ndarray] = {}
        self._models: dict[tuple[str, int], Any] = {}

    @property
    def n_params(self) -> int:
        return int(
            sum(sum(p.numel() for p in m.parameters()) for m in self._models.values())
        )

    @staticmethod
    def _teacher_layers(layer_map: list[list[int]], s: int, L_t: int) -> list[int]:
        teachers = layer_map[s] if s < len(layer_map) else [min(s, L_t - 1)]
        return [min(int(t), L_t - 1) for t in (teachers or [min(s, L_t - 1)])]

    def fit_batch(
        self,
        samples: list[tuple[np.ndarray, np.ndarray]],
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> None:
        import torch
        import torch.nn as nn
        from .math import _apply_or_skip

        if not samples:
            raise ValueError("JointMLPMapper.fit_batch: samples 不能为空")
        L_s, _, H, D = samples[0][1].shape
        L_t = samples[0][0].shape[0]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        for s in range(L_s):
            teachers = self._teacher_layers(layer_map, s, L_t)
            x_list: list[np.ndarray] = []
            y_list: list[np.ndarray] = []
            for kv_t, kv_s in samples:
                S_i = min(kv_t.shape[1], kv_s.shape[1])
                pos = (
                    np.arange(S_i, dtype=np.float64)
                    if positions is None
                    else positions[:S_i]
                )
                feats = []
                for t in teachers:
                    raw = kv_t[t, :S_i]
                    if de_rope_fn is not None and kv_kind == "K":
                        raw = _apply_or_skip(de_rope_fn, raw, pos)
                    feats.append(raw.reshape(S_i, H * D))
                x_list.append(np.concatenate(feats, axis=1).astype(np.float32))
                y_list.append(kv_s[s, :S_i].reshape(S_i, H * D).astype(np.float32))

            x = np.concatenate(x_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            in_dim = x.shape[1]
            out_dim = H * D
            x_t = torch.tensor(x, device=device)
            y_t = torch.tensor(y, device=device)

            layers: list[nn.Module] = [nn.Linear(in_dim, self.hidden), nn.GELU()]
            for _ in range(max(0, self.n_hidden_layers - 1)):
                layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
            layers += [nn.Linear(self.hidden, out_dim)]
            torch.manual_seed(self.seed * 1_000_003 + s)
            model = nn.Sequential(*layers).to(device)
            opt = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            loss_fn = nn.MSELoss()

            model.train()
            n = x_t.shape[0]
            for _ in range(self.epochs):
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
            self._models[(kv_kind, s)] = model
            with torch.no_grad():
                self.W[(kv_kind, s, 0)] = model(x_t[:1]).cpu().numpy()[0]

        logger.info(
            "[JointMLP] fit complete for %s: %d student layers, %d params total",
            kv_kind, L_s, self.n_params,
        )

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        import torch
        from .math import _apply_or_skip, _require_kv_kind

        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "JointMLPMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        out = np.zeros((L_s, S, H, D), dtype=np.float32)

        for s in range(L_s):
            key = (kv_kind, s)
            if key not in self._models:
                continue
            teachers = self._teacher_layers(layer_map, s, L_t)
            feats = []
            for t in teachers:
                raw = kv_t[t]
                if de_rope_fn is not None and kv_kind == "K":
                    raw = _apply_or_skip(de_rope_fn, raw, positions)
                feats.append(raw.reshape(S, H * D).astype(np.float32))
            x = np.concatenate(feats, axis=1)
            with torch.no_grad():
                pred = self._models[key](torch.tensor(x, device=device)).cpu().numpy()
            out[s] = pred.reshape(S, H, D)
        return out
