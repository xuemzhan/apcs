"""T08 / T10 / T12 / ablation / multiturn / prereg / orchestrator 测试。

覆盖：
    T08 Advantage State Training（§22/§24/§25/§36）：calibrate_rms 放大上限、
        BoundedAlpha tanh 范围、LowRankResidual K/V 独立参数化 + 非线性 +
        fit 收敛，run_advantage_train 集成上报 loss 指标
    T10 System Cost（§38）：cache_bytes 公式 + system.json 产物字段
    multiturn（§46）：run_multi_turn 多轮稳定性输出结构
    orchestrator（§71/§72）：TASK_ORDER 强制顺序 + Gate FAIL 直接/传递阻塞
    prereg（§70）：PREREGISTRATION.md 必含字段 + B5 修复（Gate 2A bool
        阈值渲染为数值而非 `> True`/`> False` 字面量）
    task_report（§73）：12 字段完整性
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from apcs.advantage.runner import BoundedAlpha, LowRankResidual, SourceLayerMixer, calibrate_rms, run_advantage_train
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


def test_low_rank_residual_tanh_nonlinear():
    """§22 非线性：同一 A/B 下 nonlinear="tanh" 与 nonlinear=None（线性）输出不同。

    论文公式 R = A·σ(B·Z) 的行主序等价实现 h = σ(z@A)，R = h@B。
    两实例同 seed/kind → A、B 完全相同，仅非线性开关不同，输出必须不同。
    """
    import numpy as np

    z = np.random.default_rng(7).standard_normal((20, 16))
    rk_t = LowRankResidual(d_in=16, rank=4, kind="K", nonlinear="tanh")
    rk_l = LowRankResidual(d_in=16, rank=4, kind="K", nonlinear=None)
    # 同 seed/kind → A、B 相同，仅 σ 不同
    assert np.allclose(rk_t.A, rk_l.A) and np.allclose(rk_t.B, rk_l.B)
    out_t = rk_t(z)
    out_l = rk_l(z)
    assert not np.allclose(out_t, out_l, atol=1e-6), "tanh 非线性输出应与线性输出不同（§22）"


def test_low_rank_residual_fit_reduces_loss():
    """§36 真实训练循环：fit 后损失显著下降，history 长度 == n_iter 且单调不增。

    取代"随机初始化后直接 eval"：随机初始化 loss0 显著 > fit 后 loss1。
    """
    import numpy as np

    rng = np.random.default_rng(11)
    rk = LowRankResidual(d_in=16, rank=4, kind="K")
    z = rng.standard_normal((32, 16))
    target = rng.standard_normal((32, 16))
    loss0 = float(np.mean((rk(z) - target) ** 2))
    n_iter = 200
    hist = rk.fit(z, target, n_iter=n_iter, lr=0.1)
    loss1 = float(np.mean((rk(z) - target) ** 2))
    assert len(hist) == n_iter
    # 逐轮 loss 单调不增（梯度下降）
    assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:])), "history 应单调不增"
    # 训练后 loss 显著低于初始
    assert hist[-1] < hist[0] * 0.7, f"loss_final={hist[-1]} 应 < {hist[0] * 0.7}"
    assert loss1 < loss0 * 0.7
    assert loss1 < loss0


def test_run_advantage_train_reports_loss_metrics(tmp_path):
    """§36 集成：run_advantage_train 训练 R_K/R_V（目标 = kv_s - z）并上报 loss 字段。

    metrics 含 loss_init / loss_final / n_loss，且 loss_final < loss_init。
    """
    cfg = {
        "teacher": {"num_layers": 8, "num_kv_heads": 2, "head_dim": 8},
        "student": {"num_layers": 4, "num_kv_heads": 2, "head_dim": 8, "freeze": True},
        "mapper": {"alpha_max": 0.5},
        "advantage": {"rank": 4, "n_iter": 60, "lr": 0.1},
    }
    res = run_advantage_train(cfg, tmp_path)
    m = res["metrics"]
    assert "loss_init" in m and "loss_final" in m and "n_loss" in m
    assert m["n_loss"] == 60
    assert m["loss_final"] < m["loss_init"], "训练后 loss 必须下降"
    # K/V 各自的逐轮 loss 也应上报且下降
    assert "loss_init_K" in m and "loss_final_K" in m
    assert "loss_init_V" in m and "loss_final_V" in m
    assert m["loss_final_K"] < m["loss_init_K"]
    assert m["loss_final_V"] < m["loss_init_V"]


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
    """§38 T10 端到端：run_system_cost 产出 system.json，per_context 每项带 teacher_cache_bytes。"""
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
    """§46 multiturn：多轮稳定性评测，per_turn 长度须等于 multi_turn_choices 选项数。"""
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


def test_next_allowed_blocks_transitive_failure():
    """§72 Gate FAIL 传递阻塞：直接依赖全 PASS 但传递依赖 FAIL 时也必须阻断。

    构造：t04 FAIL（陈旧状态），t05/t06/t07/t08 直接依赖均为 PASS。
    t09 依赖 {t05, t06, t07, t08} → 旧实现 `deps & failed` 只查直接依赖
    会误放行 t09；但 t05/t06/t08 均（间接）依赖 t04 → 传递 FAIL 必须阻断。
    """
    status = {
        "t00": "PASS",
        "t01": "PASS",
        "t02": "PASS",
        "t03": "PASS",
        "t04": "FAIL",  # 传递依赖 FAIL
        "t05": "PASS",
        "t06": "PASS",
        "t07": "PASS",
        "t08": "PASS",
    }
    nxt = next_allowed("t09", status)
    # t09 不能跑；无其他可跑任务（t10/t11/t12/t13 也都传递依赖 t04/t05）
    assert nxt != "t09", f"传递依赖 FAIL 时不得放行 t09，返回 {nxt}"


def test_prereg_writes_expected_fields(tmp_path):
    """§70 生成 PREREGISTRATION.md：必须含 Models/α_max/Negative Result Policy/Forbidden 等冻结字段。"""
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


def test_prereg_gate2a_bool_true_renders_number(tmp_path):
    """§70 B5 修复: Gate 2A 阈值配置为 bool(True) 时应渲染阈值数值（CHG > 0 → 0），
    而不是字面量 `> True`。design.md 中 CHG > 0 / PSR_A > 0 是数值阈值。"""
    cfg = {
        "teacher": {"model_id": "T", "revision": "r"},
        "student": {"model_id": "S", "revision": "r"},
        "datasets": {},
        "mapper": {},
        "advantage": {},
        "statistics": {},
        "gates": {
            "retention_min": 0.8,
            "retention_strong": 0.9,
            "chg_positive": True,
            "psr_a_positive": True,
        },
    }
    out = tmp_path / "PREREGISTRATION.md"
    generate_prereg(cfg, out)
    text = out.read_text(encoding="utf-8")
    assert "> True" not in text, "Gate 2A 不得渲染字面量 `> True`（B5）"
    # CHG > 0 / PSR_A > 0 的阈值为数值 0，两处均应渲染 `> 0`
    assert text.count("`> 0`") == 2, f"Gate 2A 两行都应渲染 `> 0`，实际文本:\n{text}"


def test_prereg_gate2a_bool_false_renders_number(tmp_path):
    """§70 B5 修复: bool(False) 同样不得渲染字面量 `> False`，应渲染阈值 0。"""
    cfg = {
        "teacher": {"model_id": "T", "revision": "r"},
        "student": {"model_id": "S", "revision": "r"},
        "datasets": {},
        "mapper": {},
        "advantage": {},
        "statistics": {},
        "gates": {
            "retention_min": 0.8,
            "retention_strong": 0.9,
            "chg_positive": False,
            "psr_a_positive": False,
        },
    }
    out = tmp_path / "PREREGISTRATION.md"
    generate_prereg(cfg, out)
    text = out.read_text(encoding="utf-8")
    assert "> False" not in text, "Gate 2A 不得渲染字面量 `> False`（B5）"
    assert text.count("`> 0`") == 2, f"Gate 2A 两行都应渲染 `> 0`，实际文本:\n{text}"


def test_write_task_report_12_fields(tmp_path):
    """§73 Task Report 12 字段完整性：逐字段断言都出现在 task_report.md 中。"""
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