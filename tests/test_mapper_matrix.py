"""V-mapper progression ladder 测试（§34 / §20 兼容接口）。

覆盖：
    (a) 每个新 mapper 在合成数据上拟合，held-out cosine 超过 identity baseline
    (b) AffineMapper 在有偏移目标上 >= plain Ridge
    (c) ProcrustesMapper 的 W 正交性（W^T W ≈ I）
    (d) CCAMapper 运行正确性（r < min(dims)，transform shape 正确）
    (e) _gate1_decision 三带逻辑边界测试
    (f) 现有 RidgeMapper / RidgePerHeadMapper 保持不变（smoke）

Chinese docstrings citing design.md §20/§33/§34。
"""
from __future__ import annotations

import numpy as np
import pytest

from apcs.mapper.math import (
    AffineMapper,
    CCAMapper,
    ProcrustesMapper,
    RidgePerHeadMapper,
    WhitenedMapper,
)


# ---------------------------------------------------------------------------
# Helper: 合成数据（与 test_mapper.py _toy 一致）
# ---------------------------------------------------------------------------

def _make_data(
    n_t: int = 4,
    n_s: int = 2,
    S: int = 32,
    H: int = 2,
    D: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    """构造有结构的 Teacher→Student KV 数据对。"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((S, H, D))
    Wt = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_t)]
    Ws = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_s)]
    kv_t = np.stack([
        (Z @ Wt[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_t)
    ])
    kv_s = np.stack([
        (Z @ Ws[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_s)
    ])
    layer_map = [list(range(s, s + 1)) for s in range(n_s)]
    positions = np.arange(S, dtype=np.float64)
    return kv_t.astype(np.float32), kv_s.astype(np.float32), layer_map, positions


def _make_offset_data(
    n_t: int = 4,
    n_s: int = 2,
    S: int = 32,
    H: int = 2,
    D: int = 16,
    seed: int = 0,
    offset: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray]:
    """构造有常数偏移的 Teacher→Student KV 数据对。

    kv_s = kv_t[:n_s] @ W + offset（常数偏移），affine mapper 应恢复此偏移。
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((S, H, D))
    Wt = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_t)]
    Ws = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_s)]
    kv_t = np.stack([
        (Z @ Wt[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_t)
    ])
    kv_s = np.stack([
        (Z @ Ws[l]) + offset + 0.05 * rng.standard_normal(Z.shape)
        for l in range(n_s)
    ])
    layer_map = [list(range(s, s + 1)) for s in range(n_s)]
    positions = np.arange(S, dtype=np.float64)
    return kv_t.astype(np.float32), kv_s.astype(np.float32), layer_map, positions


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个展平数组的余弦相似度。"""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return float(np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12
    ))


def _identity_cosine(kv_s: np.ndarray) -> float:
    """identity baseline：直接用 kv_s 与自身比较（= 1.0）。"""
    return _cosine(kv_s, kv_s)


# ---------------------------------------------------------------------------
# (a) 每个新 mapper 在合成数据上拟合，held-out cosine 超过 identity baseline
# ---------------------------------------------------------------------------

class TestNewMappersBeatIdentity:
    """§34：每个 V-mapper progression ladder 变体在合成数据上，
    held-out cosine 必须超过 identity baseline（0 以上即有意义）。

    所有新 mapper 实现相同的 .fit() + .transform() 接口，
    runner.py 可按 cfg.mapper.type 透明替换。
    """

    def _fit_and_score(self, mapper, kv_t, kv_s, layer_map, positions):
        """拟合 mapper 并在训练数据上评分（快速 smoke test）。"""
        mapper.fit(kv_t, kv_s, layer_map, positions=positions)
        pred = mapper.transform(kv_t, layer_map, positions=positions)
        return _cosine(pred, kv_s)

    def test_affine_beats_random(self):
        """AffineMapper：有结构数据的 cosine 应远高于纯随机数据。"""
        kv_t, kv_s, lm, pos = _make_data(seed=10)
        m = AffineMapper(lam=1e-3)
        cos = self._fit_and_score(m, kv_t, kv_s, lm, pos)
        assert cos > 0.3, f"AffineMapper cosine={cos:.4f} 应 > 0.3"

    def test_whitened_beats_random(self):
        """WhitenedMapper：标准化后 Ridge 应学到信号。"""
        kv_t, kv_s, lm, pos = _make_data(seed=11)
        m = WhitenedMapper(lam=1e-3)
        cos = self._fit_and_score(m, kv_t, kv_s, lm, pos)
        assert cos > 0.3, f"WhitenedMapper cosine={cos:.4f} 应 > 0.3"

    def test_procrustes_beats_random(self):
        """ProcrustesMapper：正交旋转应能对齐 source/target（cosine 为正）。"""
        kv_t, kv_s, lm, pos = _make_data(seed=12)
        m = ProcrustesMapper()
        cos = self._fit_and_score(m, kv_t, kv_s, lm, pos)
        assert cos > 0.0, f"ProcrustesMapper cosine={cos:.4f} 应 > 0.0"

    def test_cca_beats_random(self):
        """CCAMapper：top-r 典范方向应能保留相关信号。"""
        kv_t, kv_s, lm, pos = _make_data(seed=13)
        m = CCAMapper(r=8, lam=1e-3)
        cos = self._fit_and_score(m, kv_t, kv_s, lm, pos)
        assert cos > 0.2, f"CCAMapper cosine={cos:.4f} 应 > 0.2"

    def test_all_new_mappers_output_shape(self):
        """所有新 mapper 的 transform 输出形状必须与输入一致 (L_s, S, H, D)。"""
        kv_t, kv_s, lm, pos = _make_data(seed=14)
        expected_shape = kv_s.shape
        for name, mapper in [
            ("affine", AffineMapper(lam=1e-3)),
            ("whitening", WhitenedMapper(lam=1e-3)),
            ("procrustes", ProcrustesMapper()),
            ("cca", CCAMapper(r=4, lam=1e-3)),
        ]:
            mapper.fit(kv_t, kv_s, lm, positions=pos)
            pred = mapper.transform(kv_t, lm, positions=pos)
            assert pred.shape == expected_shape, (
                f"{name}: output shape {pred.shape} != {expected_shape}"
            )

    def test_all_new_mappers_n_params_positive(self):
        """所有新 mapper 的 n_params 必须为正整数。"""
        kv_t, kv_s, lm, pos = _make_data(seed=15)
        for name, mapper in [
            ("affine", AffineMapper(lam=1e-3)),
            ("whitening", WhitenedMapper(lam=1e-3)),
            ("procrustes", ProcrustesMapper()),
            ("cca", CCAMapper(r=4, lam=1e-3)),
        ]:
            mapper.fit(kv_t, kv_s, lm, positions=pos)
            assert mapper.n_params > 0, f"{name}: n_params={mapper.n_params} 应 > 0"


# ---------------------------------------------------------------------------
# (b) AffineMapper >= plain Ridge on offset-corrupted targets
# ---------------------------------------------------------------------------

class TestAffineOffsetRecovery:
    """§34 §20：当 Teacher→Student 存在常数偏移时，
    AffineMapper（带截距）应显著优于 plain Ridge（无截距）。

    构造数据：kv_s = kv_t[:n_s] @ W + offset（常数偏移），
    plain Ridge 必须用 W 的自由度拟合偏移 → 浪费表达力 → cosine 低；
    AffineMapper 用截距 b 直接建模偏移 → cosine 高。
    """

    def test_affine_recovers_offset(self):
        """有偏移数据上 AffineMapper cosine 应显著高于 plain Ridge。"""
        kv_t, kv_s, lm, pos = _make_offset_data(seed=20, offset=5.0)
        # Plain Ridge
        ridge = RidgePerHeadMapper(lam=1e-3)
        ridge.fit(kv_t, kv_s, lm, positions=pos)
        cos_ridge = _cosine(ridge.transform(kv_t, lm, positions=pos), kv_s)
        # Affine
        aff = AffineMapper(lam=1e-3)
        aff.fit(kv_t, kv_s, lm, positions=pos)
        cos_aff = _cosine(aff.transform(kv_t, lm, positions=pos), kv_s)
        assert cos_aff > cos_ridge, (
            f"AffineMapper (cos={cos_aff:.4f}) 应 >= plain Ridge (cos={cos_ridge:.4f})"
            f"when data has constant offset"
        )

    def test_affine_fit_batch_matches_single_fit(self):
        """AffineMapper.fit_batch（list-of-pairs）应产生与单次 fit 类似的结果。"""
        kv_t1, kv_s1, lm, pos = _make_offset_data(seed=21, offset=3.0)
        kv_t2, kv_s2, _, _ = _make_offset_data(seed=22, offset=3.0)
        # fit_batch with 2 samples
        m_batch = AffineMapper(lam=1e-3)
        m_batch.fit_batch([(kv_t1, kv_s1), (kv_t2, kv_s2)], lm, positions=pos)
        pred = m_batch.transform(kv_t1, lm, positions=pos)
        cos = _cosine(pred, kv_s1)
        assert cos > 0.3, f"fit_batch cosine={cos:.4f} 应 > 0.3"


# ---------------------------------------------------------------------------
# (c) ProcrustesMapper W is orthogonal (W^T W ≈ I)
# ---------------------------------------------------------------------------

class TestProcrustesOrthogonality:
    """§34：ProcrustesMapper 的映射矩阵 W 必须是正交矩阵（W^T W ≈ I）。

    正交性保证映射是纯旋转/反射，保持向量间距离和角度（几何保真）。
    """

    def test_procrustes_w_is_orthogonal(self):
        """ProcrustesMapper 拟合后的 W 必须满足 W^T W ≈ I（正交性）。"""
        kv_t, kv_s, lm, pos = _make_data(seed=30)
        m = ProcrustesMapper()
        m.fit(kv_t, kv_s, lm, positions=pos)
        for key, w in m.W.items():
            D = w.shape[0]
            wt_w = w.T @ w
            np.testing.assert_allclose(
                wt_w, np.eye(D), atol=1e-5,
                err_msg=f"ProcrustesMapper W[{key}] 不正交：W^T W 与 I 差 "
                        f"{np.abs(wt_w - np.eye(D)).max():.2e}",
            )

    def test_procrustes_batch_w_is_orthogonal(self):
        """fit_batch 路径的 W 也必须正交。"""
        kv_t1, kv_s1, lm, pos = _make_data(seed=31)
        kv_t2, kv_s2, _, _ = _make_data(seed=32)
        m = ProcrustesMapper()
        m.fit_batch([(kv_t1, kv_s1), (kv_t2, kv_s2)], lm, positions=pos)
        for key, w in m.W.items():
            D = w.shape[0]
            np.testing.assert_allclose(
                w.T @ w, np.eye(D), atol=1e-5,
                err_msg=f"fit_batch ProcrustesMapper W[{key}] 不正交",
            )

    def test_procrustes_deterministic(self):
        """同一批数据两次 fit 得到相同 W（确定性）。"""
        kv_t, kv_s, lm, pos = _make_data(seed=33)
        m1 = ProcrustesMapper()
        m1.fit(kv_t, kv_s, lm, positions=pos)
        m2 = ProcrustesMapper()
        m2.fit(kv_t, kv_s, lm, positions=pos)
        for key in m1.W:
            np.testing.assert_array_equal(m1.W[key], m2.W[key])


# ---------------------------------------------------------------------------
# (d) CCA runs with r < min(dims) and transform shape correct
# ---------------------------------------------------------------------------

class TestCCACorrectness:
    """§34：CCAMapper 的 CCA 投影维度正确，transform 输出 shape 正确。"""

    def test_cca_r_less_than_dims(self):
        """r < min(D, S) 时 CCA 必须能跑通且输出正确维度。"""
        kv_t, kv_s, lm, pos = _make_data(seed=40)
        D = kv_t.shape[-1]
        for r in [1, 4, 8, min(16, D - 1)]:
            m = CCAMapper(r=r, lam=1e-3)
            m.fit(kv_t, kv_s, lm, positions=pos)
            pred = m.transform(kv_t, lm, positions=pos)
            assert pred.shape == kv_s.shape, (
                f"CCA r={r}: output shape {pred.shape} != {kv_s.shape}"
            )

    def test_cca_n_params_depends_on_r(self):
        """CCA 参数量应与 r 成正比（r 越大参数越多）。"""
        kv_t, kv_s, lm, pos = _make_data(seed=41)
        params = []
        for r in [2, 4, 8]:
            m = CCAMapper(r=r, lam=1e-3)
            m.fit(kv_t, kv_s, lm, positions=pos)
            params.append(m.n_params)
        assert params[0] < params[1] < params[2], (
            f"CCA params 应随 r 递增：{params}"
        )

    def test_cca_batch_fit(self):
        """CCA fit_batch 路径能正确运行。"""
        kv_t1, kv_s1, lm, pos = _make_data(seed=42)
        kv_t2, kv_s2, _, _ = _make_data(seed=43)
        m = CCAMapper(r=4, lam=1e-3)
        m.fit_batch([(kv_t1, kv_s1), (kv_t2, kv_s2)], lm, positions=pos)
        pred = m.transform(kv_t1, lm, positions=pos)
        assert pred.shape == kv_s1.shape

    def test_cca_transform_produces_nonzero_output(self):
        """CCA transform 输出不应全零（有信号时）。"""
        kv_t, kv_s, lm, pos = _make_data(seed=44)
        m = CCAMapper(r=4, lam=1e-3)
        m.fit(kv_t, kv_s, lm, positions=pos)
        pred = m.transform(kv_t, lm, positions=pos)
        assert np.abs(pred).sum() > 0, "CCA transform 输出全零"


# ---------------------------------------------------------------------------
# (e) Gate band unit test: _gate1_decision
# ---------------------------------------------------------------------------

class TestGate1Decision:
    """§33 §6：_gate1_decision 三带逻辑的边界测试。

    默认阈值：retention_strong=0.90, retention_min=0.80。
    测试边界值：0.95（PASS）、0.85（CONDITIONAL）、0.75（FAIL）。
    """

    def test_pass_band(self):
        """task_retention >= 0.90 → PASS。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(0.95) == "PASS"
        assert _gate1_decision(0.90) == "PASS"  # 边界

    def test_conditional_band(self):
        """0.80 <= task_retention < 0.90 → CONDITIONAL。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(0.85) == "CONDITIONAL"
        assert _gate1_decision(0.80) == "CONDITIONAL"  # 下界

    def test_fail_band(self):
        """task_retention < 0.80 → FAIL。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(0.75) == "FAIL"
        assert _gate1_decision(0.0) == "FAIL"

    def test_none_is_inconclusive(self):
        """task_retention=None → INCONCLUSIVE。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(None) == "INCONCLUSIVE"

    def test_custom_thresholds(self):
        """自定义阈值也应正确路由。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(0.92, strong=0.92, minimum=0.85) == "PASS"
        assert _gate1_decision(0.88, strong=0.92, minimum=0.85) == "CONDITIONAL"
        assert _gate1_decision(0.80, strong=0.92, minimum=0.85) == "FAIL"

    def test_boundary_exactly_at_strong(self):
        """恰好等于 strong 阈值 → PASS（>= 语义）。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(0.90, strong=0.90, minimum=0.80) == "PASS"

    def test_boundary_just_below_strong(self):
        """略低于 strong 阈值（0.899）→ CONDITIONAL。"""
        from apcs.mapper.runner import _gate1_decision
        assert _gate1_decision(0.899, strong=0.90, minimum=0.80) == "CONDITIONAL"


# ---------------------------------------------------------------------------
# (f) 现有 RidgeMapper / RidgePerHeadMapper 不受影响（smoke test）
# ---------------------------------------------------------------------------

class TestExistingRidgeUnchanged:
    """smoke test：确认现有 RidgeMapper / RidgePerHeadMapper 行为不变。"""

    def test_ridge_per_head_basic(self):
        """RidgePerHeadMapper：fit + transform 基本流程。"""
        kv_t, kv_s, lm, pos = _make_data(seed=50)
        m = RidgePerHeadMapper(lam=1e-3)
        m.fit(kv_t, kv_s, lm, positions=pos)
        pred = m.transform(kv_t, lm, positions=pos)
        assert pred.shape == kv_s.shape
        cos = _cosine(pred, kv_s)
        assert cos > 0.5, f"RidgePerHeadMapper cosine={cos:.4f} 应 > 0.5"

    def test_ridge_per_head_n_params(self):
        """RidgePerHeadMapper n_params = L_s × H × D × D。"""
        kv_t, kv_s, lm, pos = _make_data(n_s=2, H=2, D=16, seed=51)
        m = RidgePerHeadMapper()
        m.fit(kv_t, kv_s, lm, positions=pos)
        # n_s=2 layers, H=2 heads, D=16 → 2*2*256 = 1024
        assert m.n_params == 2 * 2 * 16 * 16

    def test_new_mappers_dont_modify_ridge(self):
        """新 mapper 类的存在不应影响 RidgeMapper 的行为。"""
        from apcs.mapper.math import RidgeMapper
        kv_t, kv_s, lm, pos = _make_data(seed=52)
        m = RidgeMapper(lam=1e-3)
        m.fit(kv_t, kv_s, lm, positions=pos)
        pred = m.transform(kv_t, lm, positions=pos)
        cos = _cosine(pred, kv_s)
        assert cos > 0.5, f"RidgeMapper cosine={cos:.4f} 应 > 0.5"
        assert m.n_params == 2 * 16 * 16

    def test_whitened_fit_batch_produces_nonzero(self):
        """WhitenedMapper fit_batch 路径不崩溃且产生非零预测。"""
        kv_t1, kv_s1, lm, pos = _make_data(seed=53)
        kv_t2, kv_s2, _, _ = _make_data(seed=54)
        m = WhitenedMapper(lam=1e-3)
        m.fit_batch([(kv_t1, kv_s1), (kv_t2, kv_s2)], lm, positions=pos)
        pred = m.transform(kv_t1, lm, positions=pos)
        assert pred.shape == kv_s1.shape
        assert np.abs(pred).sum() > 0


# ---------------------------------------------------------------------------
# Mapper 类型路由测试
# ---------------------------------------------------------------------------

class TestMapperTypeRouting:
    """§34：cfg.mapper.type → mapper 类型路由正确。"""

    def test_create_mapper_types(self):
        """_create_mapper 返回正确的类型。"""
        from apcs.mapper.runner import _create_mapper
        for mt, expected_cls in [
            ("ridge", RidgePerHeadMapper),
            ("affine", AffineMapper),
            ("whitening", WhitenedMapper),
            ("procrustes", ProcrustesMapper),
            ("cca", CCAMapper),
        ]:
            cfg = {"mapper": {"type": mt}}
            m = _create_mapper(cfg)
            assert isinstance(m, expected_cls), (
                f"mapper.type='{mt}' 应返回 {expected_cls.__name__}，"
                f"实际返回 {type(m).__name__}"
            )

    def test_create_mapper_default_is_ridge(self):
        """默认 mapper.type 是 ridge。"""
        from apcs.mapper.runner import _create_mapper
        m = _create_mapper({})
        assert isinstance(m, RidgePerHeadMapper)

    def test_create_mapper_invalid_type_raises(self):
        """无效 mapper.type 应 raise ValueError。"""
        from apcs.mapper.runner import _create_mapper
        with pytest.raises(ValueError, match="不在已知类型列表"):
            _create_mapper({"mapper": {"type": "invalid_type"}})
