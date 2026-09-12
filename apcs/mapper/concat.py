"""Concatenation-based per-head ridge mapper (V2 of the Heo-style baseline).

The Heo-style baseline in the audit combines the selected top-k source layers by
*averaging* them (the ``layer_map[s] = [t1..tk]`` semantics of the existing
mappers). The reference design instead concatenates the selected layers' KV
before the closed-form linear map. This mapper implements that combination: for
each (student layer, head) it fits ``W: R^{k*D_t} -> R^{D_s}`` (with intercept)
on the concatenation of the de-RoPE'd source layers.

It shares the ``W``/``bias`` parameter layout with ``AffineMapper`` so the
existing persistence path can round-trip it.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class ConcatRidgeMapper:
    def __init__(self, lam: float = 1e-3):
        self.lam = float(lam)
        self.W: dict[tuple[str, int, int], np.ndarray] = {}
        self.bias: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def n_params(self) -> int:
        return int(
            sum(v.size for v in self.W.values())
            + sum(v.size for v in self.bias.values())
        )

    @staticmethod
    def _teachers(layer_map: list[list[int]], s: int, L_t: int) -> list[int]:
        ts = layer_map[s] if s < len(layer_map) else [min(s, L_t - 1)]
        return [min(int(t), L_t - 1) for t in (ts or [min(s, L_t - 1)])]

    def fit_batch(
        self,
        samples: list[tuple[np.ndarray, np.ndarray]],
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> None:
        from .math import _apply_or_skip

        if not samples:
            raise ValueError("ConcatRidgeMapper.fit_batch: samples 不能为空")
        L_s, _, H, D_s = samples[0][1].shape
        L_t, _, _, D_t = samples[0][0].shape

        # Precompute de-RoPE'd teacher layers once: T[t] = [(S_i, H, D_t), ...]
        T: list[list[np.ndarray]] = [[] for _ in range(L_t)]
        for kv_t, kv_s in samples:
            S_i = int(min(kv_t.shape[1], kv_s.shape[1]))
            pos = np.arange(S_i, dtype=np.float64) if positions is None else positions[:S_i]
            for t in range(L_t):
                T[t].append(_apply_or_skip(de_rope_fn, kv_t[t, :S_i], pos).astype(np.float64))

        for s in range(L_s):
            teachers = self._teachers(layer_map, s, L_t)
            k = len(teachers)
            for h in range(H):
                xs: list[np.ndarray] = []
                ys: list[np.ndarray] = []
                for i, (_kv_t, kv_s) in enumerate(samples):
                    S_i = T[teachers[0]][i].shape[0]
                    xs.append(np.concatenate([T[t][i][:, h, :] for t in teachers], axis=-1))
                    ys.append(kv_s[s, :S_i, h, :].astype(np.float64))
                X = np.concatenate(xs, axis=0)
                Y = np.concatenate(ys, axis=0)
                xm = X.mean(axis=0, keepdims=True)
                ym = Y.mean(axis=0, keepdims=True)
                Xc = X - xm
                Yc = Y - ym
                W = np.linalg.solve(
                    Xc.T @ Xc + self.lam * np.eye(k * D_t, dtype=np.float64),
                    Xc.T @ Yc,
                )
                self.W[(kv_kind, s, h)] = W
                self.bias[(kv_kind, s, h)] = (ym - xm @ W).ravel()

        logger.info(
            "[ConcatRidge] fit %s: D_t=%d -> D_s=%d, k=%s, %d params",
            kv_kind, D_t, D_s,
            {len(self._teachers(layer_map, s, L_t)) for s in range(L_s)}, self.n_params,
        )

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        from .math import _apply_or_skip, _require_kv_kind

        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "ConcatRidgeMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D_t = kv_t.shape
        D_s = self.W[(kv_kind, 0, 0)].shape[1]
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H, D_s), dtype=np.float64)
        for s in range(L_s):
            teachers = self._teachers(layer_map, s, L_t)
            feats = [
                _apply_or_skip(de_rope_fn, kv_t[t], positions).astype(np.float64)
                for t in teachers
            ]
            x = np.concatenate(feats, axis=-1)  # (S, H, k*D_t)
            for h in range(H):
                W = self.W[(kv_kind, s, h)]
                b = self.bias[(kv_kind, s, h)]
                out[s, :, h, :] = x[:, h, :] @ W + b
        return out.astype(np.float32)
