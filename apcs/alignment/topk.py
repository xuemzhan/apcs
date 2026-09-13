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
    """Return ``layer_map`` mapping each student layer to its top-k teacher layers.

    The per-(layer, head) design matrices and the Gram matrix ``X^T X`` are
    materialized once and reused across student layers: the ridge fit for a
    (student, teacher, head) triple only needs ``X_t^T X_t`` and ``X_t^T Y_s``.
    This is algebraically identical to refitting each pair but avoids the
    redundant concatenation/matmul that made the naive loop scale poorly.
    """
    if not calib:
        raise ValueError("top-k layer selection requires real calibration data")
    from ..mapper.math import _apply_or_skip
    from ..metrics import r2

    n = len(calib)
    n_fit = max(1, int(round(n * fit_frac)))
    has_eval = n_fit < n

    # Per-(layer, head) fit/eval matrices, accumulated in float64.
    t_fit: dict[tuple[int, int], list[np.ndarray]] = {}
    t_eval: dict[tuple[int, int], list[np.ndarray]] = {}
    s_fit: dict[tuple[int, int], list[np.ndarray]] = {}
    s_eval: dict[tuple[int, int], list[np.ndarray]] = {}
    H: int | None = None
    n_t_seen = min(n_t, int(calib[0][0].shape[0]))
    n_s_seen = min(n_s, int(calib[0][1].shape[0]))
    for i, (kv_t, kv_s) in enumerate(calib):
        S_i = int(min(kv_t.shape[1], kv_s.shape[1]))
        pos = np.arange(S_i, dtype=np.float64)
        if H is None:
            H = int(kv_s.shape[2])
        for t in range(min(n_t, kv_t.shape[0])):
            raw = _apply_or_skip(de_rope_fn, kv_t[t, :S_i], pos).astype(np.float64)
            dst = t_fit if i < n_fit else t_eval
            for h in range(H):
                dst.setdefault((t, h), []).append(raw[:, h, :])
        for s in range(min(n_s, kv_s.shape[0])):
            raw = kv_s[s, :S_i].astype(np.float64)
            dst = s_fit if i < n_fit else s_eval
            for h in range(H):
                dst.setdefault((s, h), []).append(raw[:, h, :])

    def _cat(d):  # noqa: ANN001
        return {kk: np.concatenate(v, axis=0) for kk, v in d.items()}

    t_fit = _cat(t_fit)
    s_fit = _cat(s_fit)
    if has_eval:
        t_eval = _cat(t_eval)
        s_eval = _cat(s_eval)

    # X^T X depends only on (teacher layer, head) — factorize once and reuse
    # across all student layers (the naive per-pair solve dominated runtime).
    ainv: dict[tuple[int, int], np.ndarray] = {}
    xt_fit: dict[tuple[int, int], np.ndarray] = {}
    for kk, X in t_fit.items():
        dx = X.shape[1]
        ainv[kk] = np.linalg.inv(X.T @ X + lam * np.eye(dx))
        xt_fit[kk] = X.T

    def r2_pair(s: int, t: int) -> float:
        rs: list[float] = []
        for h in range(H):  # type: ignore[arg-type]
            Xt = xt_fit[(t, h)]
            Y = s_fit[(s, h)]
            W = ainv[(t, h)] @ (Xt @ Y)
            if has_eval:
                Xe, Ye = t_eval[(t, h)], s_eval[(s, h)]
            else:
                Xe, Ye = Xt.T, Y
            rs.append(float(r2(Ye, Xe @ W)))
        return float(np.mean(rs))

    layer_map: list[list[int]] = []
    for s in range(n_s_seen):
        scores = [(r2_pair(s, t), t) for t in range(n_t_seen)]
        scores.sort(key=lambda x: (-x[0], x[1]))
        chosen = sorted(t for _, t in scores[: max(1, k)])
        layer_map.append(chosen)
        logger.info("[topk] student layer %d -> teachers %s (top scores %s)",
                    s, chosen, [round(sc, 3) for sc, _ in scores[: max(1, k)]])
    return layer_map
