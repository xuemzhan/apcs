"""Rectangular per-head affine mapper for cross-family cache translation (P2-1).

The within-family ``AffineMapper`` assumes the teacher and student share a head
dimension ``D``. A cross-family pair (e.g. Qwen3-4B, ``D=128`` -> Llama-3.2-1B,
``D=64``) violates that assumption, and the audit pipeline previously discarded
such calibration pairs. This mapper generalizes the per-head affine fit to a
rectangular map ``W: R^{D_t} -> R^{D_s}`` with an intercept, keeping the same
``fit_batch`` / ``transform`` interface as the other families. It still requires
the two models to expose the same number of KV heads; the head mapping itself is
the identity.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class RectAffineMapper:
    """Per-head rectangular affine map ``x @ W + b`` from teacher to student."""

    def __init__(self, lam: float = 1e-3):
        self.lam = float(lam)
        self.W: dict[tuple[str, int, int], np.ndarray] = {}
        self.bias: dict[tuple[str, int, int], np.ndarray] = {}
        self._in_dim: dict[str, int] = {}
        self._out_dim: dict[str, int] = {}

    @property
    def n_params(self) -> int:
        return int(
            sum(v.size for v in self.W.values())
            + sum(v.size for v in self.bias.values())
        )

    @staticmethod
    def _head_groups(H_t: int, H_s: int) -> list[list[int]]:
        """Map each student KV head to a contiguous group of teacher KV heads.

        A teacher with more KV heads is mean-pooled within each group; a teacher
        with fewer is repeated. This mirrors the design's mean ``p_h`` strategy
        for GQA ratios that do not match across families.
        """
        groups: list[list[int]] = []
        for h in range(H_s):
            lo = (h * H_t) // H_s
            hi = ((h + 1) * H_t) // H_s
            if hi <= lo:
                hi = lo + 1
            groups.append(list(range(lo, min(hi, H_t))))
        return groups

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
        from .math import _apply_or_skip

        if not samples:
            raise ValueError("RectAffineMapper.fit_batch: samples 不能为空")
        kv_s0 = samples[0][1]
        L_s, _, H_s, D_s = kv_s0.shape
        L_t = samples[0][0].shape[0]
        H_t = samples[0][0].shape[2]
        D_t = samples[0][0].shape[3]
        groups = self._head_groups(H_t, H_s)
        self._in_dim[kv_kind] = D_t
        self._out_dim[kv_kind] = D_s

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
                src = np.stack(
                    [_apply_or_skip(de_rope_fn, kv_t[t, :S_i], pos) for t in teachers],
                    axis=0,
                ).mean(axis=0).astype(np.float64)  # (S_i, H_t, D_t)
                x_list.append(src)
                y_list.append(kv_s[s, :S_i].astype(np.float64))  # (S_i, H_s, D_s)

            x_cat = np.concatenate(x_list, axis=0)  # (N, H_t, D_t)
            y_cat = np.concatenate(y_list, axis=0)  # (N, H_s, D_s)
            for h in range(H_s):
                x = x_cat[:, groups[h], :].mean(axis=1)  # (N, D_t)
                y = y_cat[:, h, :]  # (N, D_s)
                x_mean = x.mean(axis=0, keepdims=True)
                y_mean = y.mean(axis=0, keepdims=True)
                x_c = x - x_mean
                y_c = y - y_mean
                W = np.linalg.solve(
                    x_c.T @ x_c + self.lam * np.eye(D_t, dtype=np.float64),
                    x_c.T @ y_c,
                )  # (D_t, D_s)
                self.W[(kv_kind, s, h)] = W
                self.bias[(kv_kind, s, h)] = (y_mean - x_mean @ W).ravel()

        logger.info(
            "[RectAffine] fit %s: D_t=%d -> D_s=%d, H_t=%d -> H_s=%d, %d params",
            kv_kind, D_t, D_s, H_t, H_s, self.n_params,
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

        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "RectAffineMapper.transform")
        L_s = len(layer_map)
        L_t, S, H_t, _D_t = kv_t.shape
        D_s = self._out_dim[kv_kind]
        H_s = max(key[2] for key in self.W if key[0] == kv_kind) + 1
        groups = self._head_groups(H_t, H_s)
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H_s, D_s), dtype=np.float64)
        for s in range(L_s):
            teachers = self._teacher_layers(layer_map, s, L_t)
            acc = np.zeros((S, H_s, D_s), dtype=np.float64)
            for t in teachers:
                src = _apply_or_skip(de_rope_fn, kv_t[t], positions).astype(np.float64)
                for h in range(H_s):
                    x = src[:, groups[h], :].mean(axis=1)  # (S, D_t)
                    W = self.W[(kv_kind, s, h)]
                    b = self.bias[(kv_kind, s, h)]
                    acc[:, h, :] += x @ W + b
            out[s] = acc / len(teachers)
        return out.astype(np.float32)
