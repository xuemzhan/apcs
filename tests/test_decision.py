"""Decision runner 测试 + bootstrap/permutation 测试（bug-5 修复）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apcs.capability.main import (
    _paired_permutation_test,
    _simulate_scores,
    run_main_capability,
)
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


# ── bug-9 科学诚实性：跨进程稳定 seed 派生 + offline demo 标注 ──────────────


def test_simulate_deterministic_with_same_seed():
    """固定 seed 可复现：同 seed 两次调用结果完全一致（跨进程稳定）。"""
    v1 = _simulate_scores(7, "base_plus_adv")
    v2 = _simulate_scores(7, "base_plus_adv")
    assert v1 == v2, f"同 seed 应完全一致，实测 {v1} vs {v2}"


def test_simulate_seed_causes_different_offsets():
    """不同 seed 应产生不同偏移：存在 s1 != s2 使 _simulate_scores 结果不同。"""
    vals = {_simulate_scores(s, "base_plus_adv") for s in range(10)}
    assert len(vals) > 1, f"10 个不同 seed 的 base_plus_adv 得分恒同: {sorted(vals)}"


def test_no_hash_in_capability():
    """禁止内置 hash()：受 PYTHONHASHSEED 加盐，跨进程不可复现。"""
    src = Path(__file__).resolve().parents[1] / "apcs" / "capability" / "main.py"
    text = src.read_text(encoding="utf-8")
    assert "hash(" not in text, "main.py 中残留内置 hash() 调用，跨进程不可复现"


def test_t09_emits_offline_demo(tmp_path):
    """T09 结果必须自标注为 offline demo（合成数据），不得冒充真实实验。"""
    cfg = {"seeds": [0, 1, 2], "t09_n_samples": 8, "loss": {"lambda_task": 1.0}}
    res = run_main_capability(cfg, tmp_path)
    m = res["metrics"]
    assert m.get("offline_demo") is True
    assert "note" in m
    summary_text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "offline demo" in summary_text