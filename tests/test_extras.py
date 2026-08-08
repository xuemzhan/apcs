"""T08 / T10 / T12 / ablation / multiturn / prereg / orchestrator 测试。"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from apcs.advantage.runner import BoundedAlpha, LowRankResidual, SourceLayerMixer, calibrate_rms
from apcs.multiturn.runner import run_multi_turn
from apcs.orchestrator import TASK_ORDER, dependencies, next_allowed, write_task_report
from apcs.prereg import generate_prereg
from apcs.system.runner import _cache_bytes, run_system_cost


def test_calibrate_rms_caps_ratio():
    """§24 max_ratio 限制放大。"""
    import numpy as np

    base = np.ones((100,)) * 1.0
    residual = np.ones((100,)) * 100.0  # RMS=100，远大于 base
    out = calibrate_rms(base, residual, max_ratio=1.5)
    # 缩放后 RMS(R) ≤ 1.5 * RMS(base) = 1.5
    assert np.sqrt(np.mean(out**2)) <= 1.5 + 1e-6


def test_bounded_alpha_tanh_range():
    """§25 α ∈ [-α_max, α_max]。"""
    b = BoundedAlpha(alpha_max=0.5, n_layers=4)
    a = b.get()
    assert (a >= -0.5).all() and (a <= 0.5).all()


def test_low_rank_residual_separate_kv():
    """§22 K / V 独立参数化。"""
    rk = LowRankResidual(d_in=16, rank=4, kind="K")
    rv = LowRankResidual(d_in=16, rank=4, kind="V")
    assert rk.A.shape == (16, 4)
    assert rk.B.shape == (4, 16)
    # 独立 → K 的 A 与 V 的 A 不应相同
    assert not (rk.A == rv.A).all()


def test_source_layer_mixer_normalized_weights():
    """§21 weights 应均匀初始化（sum=1）。"""
    m = SourceLayerMixer(n_t=10, n_s=4, n_heads=2, top_k=2)
    assert m.w.shape == (4, 2, 2)
    # 每个 (s, h) 的 k 个权重和 = 1（均匀初始化）
    assert (m.w.sum(axis=1) == 1.0).all()


def test_cache_bytes_formula():
    """§38 cache_bytes = L × S × H × D × dtype × 2 (K+V)。"""
    # L=4, S=16, H=2, D=8, dtype=2 → 4*16*2*8*2*2 = 4096
    assert _cache_bytes(4, 16, 2, 8, dtype_bytes=2) == 4096


def test_t10_runs_and_emits_system_json(tmp_path):
    cfg = {
        "context_lengths_extended": [1024, 4096],
        "seeds": [0, 1],
        "timing": {"repeats": 5, "warmup": 1},
        "teacher": {"num_layers": 4, "num_kv_heads": 2, "head_dim": 8},
        "student": {"num_layers": 2, "num_kv_heads": 2, "head_dim": 8},
    }
    res = run_system_cost(cfg, tmp_path)
    assert res["system"]["task"] == "T10"
    assert "per_context" in res["system"]
    # §38 cache_bytes 字段必须出现
    assert all("teacher_cache_bytes" in r for r in res["system"]["per_context"])


def test_multi_turn_runs(tmp_path):
    cfg = {"seeds": [0, 1, 2], "multi_turn_choices": [1, 5, 10]}
    res = run_multi_turn(cfg, tmp_path)
    assert res["metrics"]["task"] == "Multi-turn"
    assert len(res["metrics"]["per_turn"]) == 3


def test_orchestrator_order():
    """§71 强制顺序。"""
    assert TASK_ORDER[0] == "t00"
    assert TASK_ORDER[-1] == "t13"
    assert dependencies("t09") == {"t05", "t06", "t07", "t08"}


def test_next_allowed_blocks_failed():
    """§72 Gate FAIL 不能跳过：t01 FAIL → 依赖 t01 的后续任务都不能跑。"""
    # t01 FAIL 后，依赖 t01 的任务都不能跑
    # t04 依赖 {t01, t02, t03} → 阻塞
    # t05/06 依赖 t04 → 阻塞
    # t08 依赖 {t04, t07} → 阻塞
    # t07 依赖 {t00} → 可跑
    # t09 依赖 {t05, t06, t07, t08} → 阻塞
    status = {"t00": "PASS", "t01": "FAIL", "t02": "PASS", "t03": "PASS"}
    nxt = next_allowed("t03", status)
    # 跳过所有依赖 t01 的任务，下一个可跑的就是 t07
    assert nxt == "t07", f"应允许 t07，但 next_allowed 返回 {nxt}"


def test_next_allowed_allows_when_no_failure():
    """无 FAIL 时按顺序推进。"""
    status = {"t00": "PASS", "t01": "PASS", "t02": "PASS"}
    nxt = next_allowed("t02", status)
    assert nxt == "t03"


def test_prereg_writes_expected_fields(tmp_path):
    cfg = {
        "teacher": {"model_id": "T", "revision": "r", "dtype": "bf16", "attention_implementation": "sdpa"},
        "student": {"model_id": "S", "revision": "r", "dtype": "bf16", "attention_implementation": "sdpa", "freeze": True},
        "datasets": {
            "fidelity": ["hellaswag"],
            "teacher_advantage": {"primary": "mmlu", "splits": ["train", "validation", "test"]},
            "long_context": ["ruler"],
            "behavior_sensitive": ["judge"],
        },
        "mapper": {"alpha_max": 0.5, "source_top_k": 2, "de_rope": True, "shared_basis": False, "separate_kv": True},
        "advantage": {"rank": 16, "key": "lowrank", "value": "lowrank", "rms_calibration": True},
        "query_gate": {"enabled": False},
        "seeds": [0, 1, 2],
        "statistics": {"bootstrap_n": 10000, "ci": 0.95, "paired_test": "permutation"},
        "gates": {"retention_min": 0.9, "retention_strong": 0.95, "chg_positive": True, "psr_a_positive": True},
    }
    out = tmp_path / "PREREGISTRATION.md"
    generate_prereg(cfg, out)
    text = out.read_text(encoding="utf-8")
    assert "PREREGISTRATION.md" in text
    assert "α_max" in text
    assert "Negative Result Policy" in text
    assert "Forbidden" in text


def test_write_task_report_12_fields(tmp_path):
    write_task_report(
        run_dir=tmp_path,
        task_id="t05",
        status="PASS",
        objective="验证 Replacement",
        model_pair=("Qwen3-4B", "Qwen3-1.7B"),
        dataset="mmlu",
        config_summary={"context_lengths": [512]},
        implementation="apcs.mapper.runner.run_replacement",
        output_files=["metrics.json", "summary.md"],
        key_metrics={"retention": 0.92},
        statistical_check={"ci": 0.95},
    )
    text = (tmp_path / "task_report.md").read_text(encoding="utf-8")
    for field in [
        "TASK_ID",
        "STATUS",
        "OBJECTIVE",
        "MODEL_PAIR",
        "DATASET",
        "CONFIG",
        "IMPLEMENTATION",
        "OUTPUT_FILES",
        "KEY_METRICS",
        "STATISTICAL_CHECK",
        "BEHAVIOR_CHECK",
        "GEOMETRY_CHECK",
        "SYSTEM_COST",
        "ACCEPTANCE_CRITERIA",
        "FAILURE_ANALYSIS",
        "NEXT_ALLOWED_TASK",
    ]:
        assert field in text, f"task_report.md 缺字段 {field}"