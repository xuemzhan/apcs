"""metrics 模块单元测试（§3/§4/§45/§48/§51 指标数学正确性）。

覆盖：Retention / CHG / TGRR / PCR、PSR_A（含分母为 0 的 safe_div 边界）、
      Cosine / R² / JCR / KL / RMS、effective rank、principal angle、
      线性 CKA、bootstrap CI（§51）。
"""
from __future__ import annotations

import numpy as np

from apcs.metrics import (
    bootstrap_ci,
    chg,
    cosine,
    effective_rank,
    jcr,
    kl_divergence,
    linear_cka,
    pcr,
    principal_angle,
    psr_a,
    r2,
    retention,
    rms_norm,
    tgrr,
)


def test_retention_chg_tgrr_pcr_basic():
    """§3 四大核心指标基础公式：retention/chg/tgrr/pcr 数值精确断言。"""
    assert abs(retention(0.8, 0.5) - 1.6) < 1e-12
    assert abs(chg(0.8, 0.5) - 0.3) < 1e-12
    assert abs(tgrr(0.8, 0.5, 0.9) - 0.75) < 1e-12
    assert abs(pcr(100, 200) - 0.5) < 1e-12


def test_psr_a():
    """§4 PSR_A：正常公式 + 全 0 时分母为 0 → safe_div 返回 0 → PSR_A=1.0。"""
    assert abs(psr_a(1, 1, 1, 10) - (1 - 0.3)) < 1e-12
    # 全 0 时分母为 0 → safe_div 返回 0 → 1 - 0 = 1.0
    assert psr_a(0, 0, 0, 0) == 1.0


def test_cosine_and_r2():
    """Cosine：共线向量应接近 1；R²：b=2a 时为 -6、完全相等时为 1。"""
    a = np.array([1.0, 2.0, 3.0])
    b = a * 2.0
    assert cosine(a, b) > 0.9999
    # b = 2a：b - a = a，ss_res=14；ss_tot=2 → R² = 1 - 7 = -6
    assert r2(a, b) == -6.0
    # 完全相等的 R² = 1
    assert r2(a, a) > 0.9999


def test_jcr():
    """§48 JCR（Judged Correctness Ratio）：逐位置比较，空列表安全返回 0。"""
    assert jcr([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    assert jcr([], []) == 0.0


def test_kl_zero_for_equal():
    """KL 散度自对称性：同分布时结果 ≈ 0。"""
    p = np.array([0.25, 0.75])
    assert kl_divergence(p, p) < 1e-6


def test_rms_norm():
    """RMS 归一化：sqrt(mean(x²)) 数值验证，单元素返回 |x|。"""
    x = np.array([3.0, 4.0])
    # RMS = sqrt(mean(x²)) = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
    assert abs(rms_norm(x) - np.sqrt(12.5)) < 1e-9
    # 单元素 RMS = |x|
    assert abs(rms_norm(np.array([5.0])) - 5.0) < 1e-12


def test_effective_rank():
    """effective rank：一维向量为 1，各向同性 50×50 矩阵接近 min(n,m)（约 39）。"""
    # 一维向量 effective rank = 1
    assert abs(effective_rank(np.array([1.0, 2.0])) - 1.0) < 1e-6
    # 各向同性随机矩阵 effective rank 接近 min(n,m)（对 50x50 实际值约 39）
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 50))
    er = effective_rank(x)
    assert 35 < er < 42


def test_principal_angle_identical():
    """§40 主角度：子空间与自身的主角度应 ≈ 0。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 16))
    assert principal_angle(a, a) < 1e-3


def test_linear_cka_self():
    """§40 线性 CKA：特征与自身比较应 ≈ 1（最大相似）。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 32))
    assert linear_cka(a, a) > 0.999


def test_bootstrap_ci():
    """§51 bootstrap CI：常量样本 point=0.5，且 lo ≤ point ≤ hi。"""
    vals = [0.5] * 100
    point, lo, hi = bootstrap_ci(vals, n_boot=500, ci=0.95)
    assert abs(point - 0.5) < 1e-6
    assert lo <= point <= hi