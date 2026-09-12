"""Data-driven top-k cross-layer source selection for the Heo-style baseline.

The within-family mapper ladder uses *proportional* layer correspondence. Heo et
al. instead select, per target layer, the top-k source layers by a closed-form
ridge reconstruction score. This module implements that selection: for every
(student layer s, teacher layer t) pair it fits a per-head ridge on a fit split
of the calibration samples and scores held-out R^2 on the rest, then keeps the
top-k teacher layers per student layer.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def select_topk_layer_map(
    calib: list[tuple[np.ndarray, np.ndarray]],
    n_t: int,
    n_s: int,
    k: int,
    de_rope_fn: Callable | None = None,
    lam: float = 1e-3,
    fit_frac: float = 0.7,
) -> list[list[int]]:
    """Return ``layer_map`` mapping each student layer to its top-k teacher layers."""
    if not calib:
        raise ValueError("top-k layer selection requires real calibration data")
    from ..mapper.math import _apply_or_skip
    from ..metrics import r2

    n = len(calib)
    n_fit = max(1, int(round(n * fit_frac)))
    fit_idx = list(range(n_fit))
    eval_idx = list(range(n_fit, n))

    T: list[list[np.ndarray]] = [[] for _ in range(n_t)]
    S: list[list[np.ndarray]] = [[] for _ in range(n_s)]
    for kv_t, kv_s in calib:
        S_i = int(min(kv_t.shape[1], kv_s.shape[1]))
        pos = np.arange(S_i, dtype=np.float64)
        for t in range(min(n_t, kv_t.shape[0])):
            raw = _apply_or_skip(de_rope_fn, kv_t[t, :S_i], pos)
            T[t].append(raw.astype(np.float64))
        for s in range(min(n_s, kv_s.shape[0])):
            S[s].append(kv_s[s, :S_i].astype(np.float64))

    def r2_pair(s: int, t: int) -> float:
        H = S[s][0].shape[1]
        rs: list[float] = []
        for h in range(H):
            xf = np.concatenate([T[t][i][:, h, :] for i in fit_idx], axis=0)
            yf = np.concatenate([S[s][i][:, h, :] for i in fit_idx], axis=0)
            xe = (np.concatenate([T[t][i][:, h, :] for i in eval_idx], axis=0)
                  if eval_idx else xf)
            ye = (np.concatenate([S[s][i][:, h, :] for i in eval_idx], axis=0)
                  if eval_idx else yf)
            dx = xf.shape[1]
            W = np.linalg.solve(xf.T @ xf + lam * np.eye(dx), xf.T @ yf)
            rs.append(float(r2(ye, xe @ W)))
        return float(np.mean(rs))

    layer_map: list[list[int]] = []
    for s in range(n_s):
        scores = [(r2_pair(s, t), t) for t in range(n_t)]
        scores.sort(key=lambda x: (-x[0], x[1]))
        chosen = sorted(t for _, t in scores[: max(1, k)])
        layer_map.append(chosen)
        logger.info("[topk] student layer %d -> teachers %s (top scores %s)",
                    s, chosen, [round(sc, 3) for sc, _ in scores[: max(1, k)]])
    return layer_map
