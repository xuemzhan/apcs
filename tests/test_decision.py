"""Decision runner 测试 + bootstrap/permutation 测试（bug-5 修复）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apcs.capability.main import _paired_permutation_test, run_main_capability
from apcs.decision.runner import decide


def test_decide_a_when_all_positive():
    assert decide(retention=0.95, chg=0.05, tgrr=0.5, psr_a=0.3) == "A_RUNTIME_CAPABILITY_TRANSFER"


def test_decide_b_when_chg_zero_psr_positive():
    assert decide(retention=0.92, chg=-0.01, tgrr=-0.1, psr_a=0.4) == "B_EFFICIENT_STATE_HANDOFF"


def test_decide_c_mechanism_boundary():
    """retention ≥0.90, chg ≤0, psr_a ≤0 → C mechanism boundary。"""
    assert decide(retention=0.92, chg=-0.01, tgrr=-0.1, psr_a=-0.1) == "C_MECHANISM_BOUNDARY"


def test_decide_d_when_low_retention():
    assert decide(retention=0.5, chg=0.0, tgrr=0.0, psr_a=0.0) == "D_STOP_REPLACEABILITY_UNSTABLE"


def test_permutation_p_value_under_h0():
    """a == b 时 p-value 应接近 1.0（极端对称情形）。"""
    import numpy as np

    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = a.copy()
    p = _paired_permutation_test(a, b, n_perm=2000)
    # 严格 H0 下，observed = 0，permuted_stat >= 0 几乎都满足 → p 接近 1
    assert p > 0.9, f"H0 下 p 应接近 1，实测 {p}"


def test_permutation_p_value_small_under_h1():
    """a 显著大于 b 时 p-value 应远小于 0.05。"""
    import numpy as np

    a = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    b = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    p = _paired_permutation_test(a, b, n_perm=2000)
    assert p < 0.05, f"H1 下 p 应很小，实测 {p}"


def test_t09_runs_and_emits_gate2a(tmp_path):
    """T09 端到端：bootstrap CI + permutation + Gate 2A 判定。"""
    cfg = {
        "seeds": [0, 1, 2],
        "t09_n_samples": 8,
        "loss": {"lambda_task": 1.0},
    }
    res = run_main_capability(cfg, tmp_path)
    m = res["metrics"]
    assert "chg_bootstrap" in m
    assert m["chg_bootstrap"]["n_samples"] == 8
    assert "chg_permutation" in m
    assert "gap_strata" in m
    # gate2a 字段必须存在
    assert "gate2a" in m
    # 至少 report 写出来了
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "summary.md").exists()