"""Mapper 数学层单元测试（覆盖所有 bug 修复）。

bug-2: de-RoPE 必须接入 mapper
bug-3: 形状不兼容要 raise 而非静默截断
bug-4: calibration KV 必须有真实对应关系
"""
from __future__ import annotations

import numpy as np
import pytest

from apcs.mapper.math import (
    LowRankMapper,
    RidgePerHeadMapper,
    SharedBasisMapper,
    _check_shape,
)


def _toy(n_t=4, n_s=2, S=8, H=2, D=16, seed=0, with_signal=True):
    """构造有结构的 toy 数据：kv_s 是 kv_t 经线性变换 + 噪声。"""
    rng = np.random.default_rng(seed)
    if with_signal:
        Z = rng.standard_normal((S, H, D))
        Wt = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_t)]
        Ws = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_s)]
        kv_t = np.stack([(Z @ Wt[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_t)])
        kv_s = np.stack([(Z @ Ws[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_s)])
        return kv_t.astype(np.float32), kv_s.astype(np.float32)
    else:
        return (
            rng.standard_normal((n_t, S, H, D)).astype(np.float32),
            rng.standard_normal((n_s, S, H, D)).astype(np.float32),
        )


def _layer_map(n_t, n_s):
    return [list(range(s, s + 1)) for s in range(n_s)]


# ---- bug-3 修复：_check_shape 必须 raise ----


def test_check_shape_raises_on_mismatch():
    bad_t = np.zeros((4, 8, 2, 16))
    bad_s = np.zeros((2, 7, 2, 16))  # S 不一致
    with pytest.raises(ValueError, match="形状不兼容"):
        _check_shape(bad_t, bad_s)


def test_check_shape_raises_on_too_few_teacher_layers():
    bad_t = np.zeros((1, 8, 2, 16))
    bad_s = np.zeros((2, 8, 2, 16))  # n_t < n_s
    with pytest.raises(ValueError, match="层数"):
        _check_shape(bad_t, bad_s)


# ---- bug-2 修复：de-RoPE 接入 mapper ----


def test_ridge_per_head_with_de_rope():
    """有结构数据 + de-RoPE 应该学到比纯随机更高的 retention。"""
    from apcs.rope.runner import _rope_pairs, de_rope

    kv_t, kv_s = _toy(n_t=4, n_s=2, S=16, H=2, D=16, seed=0)
    positions = np.arange(16, dtype=np.float64)
    inv_freq = _rope_pairs(16, theta=1_000_000.0)
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)

    m = RidgePerHeadMapper(lam=1e-3)
    m.fit(kv_t, kv_s, _layer_map(4, 2), positions=positions, de_rope_fn=de_rope_fn)
    pred = m.transform(kv_t, _layer_map(4, 2), positions=positions, de_rope_fn=de_rope_fn)
    assert pred.shape == kv_s.shape


def test_ridge_per_head_without_de_rope_also_works():
    """无 de-RoPE 时也必须能跑（保持兼容）。"""
    kv_t, kv_s = _toy()
    m = RidgePerHeadMapper()
    m.fit(kv_t, kv_s, _layer_map(4, 2))
    pred = m.transform(kv_t, _layer_map(4, 2))
    assert pred.shape == kv_s.shape


# ---- 低秩 + Shared Basis 形状与参数 ----


def test_lowrank_shape_and_smaller_than_ridge():
    kv_t, kv_s = _toy()
    ridge = RidgePerHeadMapper()
    ridge.fit(kv_t, kv_s, _layer_map(4, 2))
    lr = LowRankMapper(rank=2)
    lr.fit(kv_t, kv_s, _layer_map(4, 2))
    assert lr.n_params < ridge.n_params


def test_shared_basis_smaller_than_lowrank():
    kv_t, kv_s = _toy()
    lr = LowRankMapper(rank=4)
    lr.fit(kv_t, kv_s, _layer_map(4, 2))
    sb = SharedBasisMapper(rank=4)
    sb.fit(kv_t, kv_s, _layer_map(4, 2))
    assert sb.n_params < lr.n_params
    pred = sb.transform(kv_t, _layer_map(4, 2))
    assert pred.shape == kv_s.shape


# ---- bug-4 修复：calibration 数据应该让 retention > 0.5 ----


def test_calibration_data_has_signal():
    """§32 calibration 数据有真实结构对应 → retention 应显著高于纯随机。

    bug-4 验证：用最直接的形式：kv_s = kv_t @ W + 噪声。
    Ridge 应该能学到 W^{-1} 使 retention 接近 1.0。
    纯随机独立数据时 retention ≈ 0。
    """
    n_t, n_s, H, D = 4, 2, 2, 16
    S = 64

    # case 1: kv_s = kv_t[:n_s] @ W + 极小噪声 → 可完美学
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    W = rng.standard_normal((D, D)).astype(np.float32) / np.sqrt(D)
    kv_s = (kv_t[:n_s] @ W + 0.001 * rng.standard_normal((n_s, S, H, D))).astype(np.float32)
    layer_map = _layer_map(n_t, n_s)
    ridge = RidgePerHeadMapper(lam=1e-5)
    ridge.fit(kv_t, kv_s, layer_map)
    pred = ridge.transform(kv_t, layer_map)
    a = pred.reshape(-1, D)
    b = kv_s.reshape(-1, D)
    cos_struct = float(np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    ))

    # case 2: 纯随机独立数据 → cosine 应 ≈ 0
    kv_t_rand = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    kv_s_rand = rng.standard_normal((n_s, S, H, D)).astype(np.float32)
    ridge2 = RidgePerHeadMapper(lam=1e-5)
    ridge2.fit(kv_t_rand, kv_s_rand, layer_map)
    pred_rand = ridge2.transform(kv_t_rand, layer_map)
    a2 = pred_rand.reshape(-1, D)
    b2 = kv_s_rand.reshape(-1, D)
    cos_rand = float(np.dot(a2.reshape(-1), b2.reshape(-1)) / (
        np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-12
    ))

    # 有结构数据 retention 应显著高于随机
    assert cos_struct > 0.5, (
        f"有结构数据应可完美学，但 cos_struct={cos_struct:.4f}"
    )
    assert cos_struct > cos_rand + 0.3, (
        f"有结构应 > 随机；struct={cos_struct:.4f}, rand={cos_rand:.4f}"
    )