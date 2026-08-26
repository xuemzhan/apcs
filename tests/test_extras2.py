"""§11/§12/§52/§13 新模块测试。

覆盖：
    §52 compliance 八条禁止检查器：check_1..check_8 逐条检测 + check_all
        干净路径 + compliance_report 汇总格式
    §12 G2 Mismatched Mapper：HeadProjection（mean/repeat）、
        DimensionProjection（truncate/pad/linear）、MismatchedHeadMapper 端到端
    §11 / T13 Generalization：扫描 base_dir 下所有 run 的 T05/T09，
        汇总 n_pairs_pass_gate2a
    batch ridge 数学等价性（_ridge_closed_form_batch vs 逐 head 循环，atol=1e-9）
    einsum 加速 transform vs 朴素循环等价（Ridge / LowRank）
    compliance runtime 信号 CRUD + §64/§65 metadata 联合集成
    §62 Figure 7 数据流（bug-7/B7 回归：T12 metrics 必须携带 "geometry" 键
        fig7a 才能渲染出 > 1KB 的有效 png）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apcs.compliance import (
    ComplianceViolation,
    check_1_no_student_re_read_x,
    check_2_student_frozen_in_main_exp,
    check_4_no_teacher_win_filtering,
    check_5_no_hidden_teacher_prefill_cost,
    check_6_no_hidden_h2d_cost,
    check_7_no_replacing_chg_with_similarity,
    check_8_no_silent_reprefill,
    check_all,
    compliance_report,
)
from apcs.generalization.runner import run_generalization
from apcs.mapper.mismatched import (
    DimensionProjection,
    HeadProjection,
    MismatchedHeadMapper,
)
from apcs.mapper.math import _ridge_closed_form, _ridge_closed_form_batch


# ---- §52 compliance ----


def test_compliance_check_1_student_re_read_x():
    """§52.1: Student 重新读 X 应被检测到。"""
    bad = check_1_no_student_re_read_x(
        Path("."), {"student_input_has_context": True}
    )
    assert len(bad) == 1
    assert bad[0].rule_id == "§52.1"

    good = check_1_no_student_re_read_x(
        Path("."), {"student_input_has_context": False}
    )
    assert good == []


def test_compliance_check_2_student_frozen():
    """§52.2：主实验微调 Student 却仍称 Runtime Transfer → 必须检出违规。"""
    cfg = {"student": {"freeze": True}}
    bad = check_2_student_frozen_in_main_exp(
        cfg, {"student_params_updated": True}
    )
    assert len(bad) == 1
    assert bad[0].rule_id == "§52.2"


def test_compliance_check_4_teacher_win_filter():
    """§52.4：只挑 Teacher-win 的 test 样本 → 必须检出违规。"""
    bad = check_4_no_teacher_win_filtering(
        {"test_filtered_to_teacher_win": True}
    )
    assert bad[0].rule_id == "§52.4"


def test_compliance_check_5_hidden_teacher_prefill():
    """§52.5：不报告 Teacher Prefill 成本 → 必须检出违规。"""
    bad = check_5_no_hidden_teacher_prefill_cost(
        {"reports_teacher_prefill": False}
    )
    assert bad[0].rule_id == "§52.5"


def test_compliance_check_6_hidden_h2d():
    """§52.6：隐藏 H2D / Cache Load 成本 → 必须检出违规。"""
    bad = check_6_no_hidden_h2d_cost({"hides_h2d_load": True})
    assert bad[0].rule_id == "§52.6"


def test_compliance_check_7_similarity_for_path_a():
    """§52.7：用 R²/Cosine/CKA 相似度主张 Path A → 必须检出违规。"""
    bad = check_7_no_replacing_chg_with_similarity(
        {"claim_path_a_on_similarity_only": True}
    )
    assert bad[0].rule_id == "§52.7"


def test_compliance_check_8_silent_re_prefill():
    """§52.8：Cache 注入失败后静默 re-prefill → 必须检出违规。"""
    bad = check_8_no_silent_reprefill({"silent_re_prefill_on_failure": True})
    assert bad[0].rule_id == "§52.8"


def test_check_all_clean():
    """无任何违反时返回空列表。"""
    out = check_all({}, Path("."), {})
    assert out == []


def test_compliance_report_format():
    """compliance_report 汇总格式：n_violations 计数 + passed 布尔必须正确。"""
    v = [ComplianceViolation("§52.1", "test")]
    r = compliance_report(v)
    assert r["n_violations"] == 1
    assert r["passed"] is False


# ---- §12 G2 Mismatched Mapper ----


def test_head_projection_mean():
    """H_T=4 → H_S=2，mean 策略。"""
    hp = HeadProjection(n_t_heads=4, n_s_heads=2, strategy="mean")
    x = np.random.RandomState(0).standard_normal((3, 8, 4, 5))  # (..., H_T, D)
    y = hp.project(x)
    assert y.shape == (3, 8, 2, 5)
    # y[..., 0, :] 应是 x[..., 0:2, :].mean(axis=-2)
    np.testing.assert_allclose(
        y[..., 0, :], x[..., 0:2, :].mean(axis=-2), atol=1e-6
    )


def test_head_projection_repeat():
    """H_T=2 → H_S=4，repeat 策略：前 2 个 Student head 原样复制 Teacher head 0/1。"""
    hp = HeadProjection(n_t_heads=2, n_s_heads=4, strategy="repeat")
    x = np.random.RandomState(0).standard_normal((3, 8, 2, 5))
    y = hp.project(x)
    assert y.shape == (3, 8, 4, 5)
    # 前两个 head 应等于原 Teacher head 0,1
    np.testing.assert_allclose(y[..., 0, :], x[..., 0, :], atol=1e-6)


def test_dimension_projection_truncate():
    """D_T=8 → D_S=4，truncate 策略：直接截断保留前 4 维。"""
    dp = DimensionProjection(d_t=8, d_s=4, strategy="truncate")
    x = np.random.RandomState(0).standard_normal((10, 8))
    y = dp.project(x)
    assert y.shape == (10, 4)
    np.testing.assert_allclose(y, x[:, :4], atol=1e-6)


def test_dimension_projection_pad():
    """D_T=4 → D_S=8，pad 策略：前 4 维原样保留、后 4 维补零。"""
    dp = DimensionProjection(d_t=4, d_s=8, strategy="pad")
    x = np.random.RandomState(0).standard_normal((10, 4))
    y = dp.project(x)
    assert y.shape == (10, 8)
    np.testing.assert_allclose(y[:, :4], x, atol=1e-6)
    np.testing.assert_allclose(y[:, 4:], 0, atol=1e-6)


def test_dimension_projection_linear_fit_and_project():
    """linear P_d 应当能学回一个简单的对角缩放矩阵。"""
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((100, 8)).astype(np.float32)
    scale = 2.5
    kv_s = (kv_t * scale).astype(np.float32)  # 对角缩放

    dp = DimensionProjection(d_t=8, d_s=8, strategy="linear")
    dp.fit(kv_t, kv_s)
    assert dp.W is not None
    # 应用投影
    proj = dp.project(kv_t)
    # 与 kv_s 应一致
    np.testing.assert_allclose(proj, kv_s, atol=1e-2)


def test_mismatched_mapper_end_to_end():
    """H_T=8, H_S=4；D_T=8, D_S=8 的端到端映射。

    构造 ground truth：
        Student_KV[s] = mean(Teacher_KV[s, head 0:H_T/H_S])  (P_H mean)
                          @ W                              (Ridge 部分)
        即 P_H 用 mean 把 8→4，再用 W 做线性变换。
    """
    n_t, n_s, H_T, H_S, D, S = 4, 2, 8, 4, 8, 64
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((n_t, S, H_T, D)).astype(np.float32)
    W = rng.standard_normal((D, D)).astype(np.float32) / np.sqrt(D)
    # Teacher mean-pool 8 个 head → 4 个 head（每 2 个 head 一组取均值）
    kv_t_grouped = kv_t.reshape(n_t, S, H_S, H_T // H_S, D).mean(axis=3)  # (n_t, S, H_S, D)
    # 内层 W 对每个 token、head 应用
    kv_s = (kv_t_grouped[:n_s] @ W).astype(np.float32)  # (n_s, S, H_S, D)
    layer_map = [[s] for s in range(n_s)]

    mapper = MismatchedHeadMapper(
        n_t_layers=n_t,
        n_s_layers=n_s,
        n_t_heads=H_T,
        n_s_heads=H_S,
        d_t=D,
        d_s=D,
        p_h_strategy="mean",
        p_d_strategy="identity",
    )
    mapper.fit(kv_t, kv_s, layer_map)
    pred = mapper.transform(kv_t, layer_map)
    assert pred.shape == kv_s.shape
    a = pred.reshape(-1, D)
    b = kv_s.reshape(-1, D)
    cos = float(np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    ))
    # mean P_H + Ridge 内层 一次性学出完整映射，cosine 应很高
    assert cos > 0.3, f"G2 mapper 应该至少能学到部分映射，但 cos={cos:.4f}"


# ---- §11 第二 Pair / T13 Generalization ----


def test_t13_generalization_scans_runs(tmp_path):
    """T13 扫描 base_dir 下所有 run 的 T05/T09，汇总 Gate 2A 通过情况。"""
    base = tmp_path / "runs"
    # 造两个 run 数据
    for i, (ret, chg, tgrr) in enumerate([(0.95, 0.05, 0.3), (0.92, -0.01, -0.05)]):
        rid = f"pair-{i}"
        (base / rid / "t05").mkdir(parents=True)
        (base / rid / "t05" / "metrics.json").write_text(
            json.dumps(
                {
                    "task_retention": ret,
                    "mean_kv_cosine": ret,
                    "offline_demo": False,
                    "evidence_grade": "measured_task",
                }
            ), encoding="utf-8"
        )
        (base / rid / "t05" / "config.json").write_text(
            json.dumps(
                {
                    "teacher": {"model_id": f"teacher-{i}"},
                    "student": {"model_id": f"student-{i}"},
                }
            ), encoding="utf-8"
        )
        (base / rid / "t09").mkdir(parents=True)
        (base / rid / "t09" / "metrics.json").write_text(
            json.dumps(
                {
                    "per_method": [
                        {"method": "base_plus_adv", "chg": chg, "tgrr": tgrr}
                    ],
                    "seeds": [0, 1, 2],
                    "offline_demo": False,
                    "evidence_grade": "measured_task",
                }
            ),
            encoding="utf-8",
        )

    cfg = {"output": {"base_dir": str(base)}}
    res = run_generalization(cfg, tmp_path / "out")
    assert res["metrics"]["n_pairs_evaluated"] == 2
    assert res["metrics"]["n_pairs_pass_gate2a"] == 1


def test_t13_aggregates_duplicate_pair_runs_without_hiding_failure(tmp_path):
    """同一模型对的失败复跑必须阻止 pair 通过，不能只选第一条。"""
    base = tmp_path / "runs"
    for i, chg in enumerate((0.08, -0.02)):
        run = base / f"repeat-{i}"
        (run / "t05").mkdir(parents=True)
        (run / "t05" / "config.json").write_text(
            json.dumps(
                {
                    "teacher": {"model_id": "teacher", "revision": "abc"},
                    "student": {"model_id": "student", "revision": "def"},
                }
            ),
            encoding="utf-8",
        )
        (run / "t05" / "metrics.json").write_text(
            json.dumps(
                {
                    "task_retention": 0.95,
                    "mean_kv_cosine": 0.97,
                    "offline_demo": False,
                    "evidence_grade": "measured_task",
                }
            ),
            encoding="utf-8",
        )
        (run / "t09").mkdir()
        (run / "t09" / "metrics.json").write_text(
            json.dumps(
                {
                    "per_method": [
                        {"method": "base_plus_adv", "chg": chg, "tgrr": chg}
                    ],
                    "seeds": [0, 1, 2],
                    "offline_demo": False,
                    "evidence_grade": "measured_task",
                }
            ),
            encoding="utf-8",
        )

    result = run_generalization({"output": {"base_dir": str(base)}}, tmp_path / "out")
    assert result["metrics"]["n_pairs_evaluated"] == 1
    row = result["metrics"]["rows"][0]
    assert row["n_runs"] == 2
    assert row["gate2a_pass"] is False
    assert row["chg_mean"] == pytest.approx(0.03)


# ---- batch ridge 数学等价性 ----


def test_ridge_batch_equivalent_to_loop():
    """_ridge_closed_form_batch 与逐 head _ridge_closed_form 应当数学等价。"""
    rng = np.random.default_rng(0)
    H, n, d_in, d_out = 4, 32, 8, 8
    x_batch = rng.standard_normal((H, n, d_out))
    y_batch = rng.standard_normal((H, n, d_in))
    W_batch = _ridge_closed_form_batch(x_batch, y_batch, lam=1e-3)
    assert W_batch.shape == (H, d_in, d_out)
    for h in range(H):
        W_loop = _ridge_closed_form(x_batch[h], y_batch[h], lam=1e-3)
        np.testing.assert_allclose(W_batch[h], W_loop, atol=1e-9)


def test_ridge_transform_correctness():
    """einsum 加速版 transform 应与朴素循环版数学等价。"""
    from apcs.mapper.math import RidgePerHeadMapper, _apply_or_skip

    n_t, n_s, H, D, S = 4, 2, 4, 16, 32
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    kv_s = (kv_t[:n_s] @ rng.standard_normal((D, D)) / np.sqrt(D) + 0.001).astype(np.float32)
    layer_map = [[s] for s in range(n_s)]

    m = RidgePerHeadMapper(lam=1e-5)
    m.fit(kv_t, kv_s, layer_map, kv_kind="K")
    pred = m.transform(kv_t, layer_map, kv_kind="K")
    assert pred.shape == kv_s.shape

    # 与朴素实现对比（per-head per-token matrix multiply）
    naive = np.zeros_like(pred)
    for s in range(n_s):
        for t in layer_map[s]:
            for h in range(H):
                W = m.W[("K", s, h)]
                naive[s, :, h, :] = kv_t[t, :, h, :] @ W
    naive /= np.mean([len(layer_map[s]) for s in range(n_s)])
    np.testing.assert_allclose(pred, naive, atol=1e-5)


def test_lowrank_transform_correctness():
    """LowRankMapper einsum 加速版应与朴素版数学等价。"""
    from apcs.mapper.math import LowRankMapper

    n_t, n_s, H, D, S, R = 4, 2, 4, 16, 32, 4
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    kv_s = (kv_t[:n_s] @ rng.standard_normal((D, D)) / np.sqrt(D) + 0.001).astype(np.float32)
    layer_map = [[s] for s in range(n_s)]

    m = LowRankMapper(rank=R)
    m.fit(kv_t, kv_s, layer_map, kv_kind="K")
    pred = m.transform(kv_t, layer_map, kv_kind="K")
    assert pred.shape == kv_s.shape

    # 朴素版
    naive = np.zeros_like(pred)
    for s in range(n_s):
        for t in layer_map[s]:
            for h in range(H):
                naive[s, :, h, :] = ((kv_t[t, :, h, :] @ m.A[("K", s, h)]) @ m.B[("K", s, h)])
    naive /= np.mean([len(layer_map[s]) for s in range(n_s)])
    np.testing.assert_allclose(pred, naive, atol=1e-5)


def test_runtime_signals_basic():
    """runtime signals 基本 CRUD。"""
    from apcs.compliance.runtime import (
        collect,
        get,
        reset,
        track,
        track_runtime,
    )

    reset()
    assert collect() == {}
    track("foo", 1)
    assert get("foo") == 1

    with track_runtime("bar", 2):
        assert get("bar") == 2
    # 退出后清除
    assert get("bar") is None


def test_compliance_metadata_integration(tmp_path):
    """§64/§65 metadata 与 §52 compliance 联合集成。"""
    from apcs.compliance import check_all, compliance_report
    from apcs.compliance.runtime import collect, infer_signals_from_cfg, track
    from apcs.io.metadata import (
        REQUIRED_64,
        REQUIRED_65,
        enrich_config,
        enrich_metrics,
        validate_config,
        write_run_metadata,
    )

    cfg = {
        "teacher": {"model_id": "T", "revision": "r", "dtype": "bf16", "device_map": "cuda:0"},
        "student": {"model_id": "S", "revision": "r", "dtype": "bf16", "freeze": True},
        "datasets": {"teacher_advantage": {"primary": "mmlu"}},
        "mapper": {"type": "ridge", "rank": 16, "alpha_max": 0.5},
        "loss": {"lambda_task": 1.0, "lambda_self": 0.1},
        "seeds": [0, 1, 2],
        "context_lengths": [512, 1024],
        "cache_residency": "cpu",
        "output": {"base_dir": str(tmp_path)},
    }
    enriched = enrich_config(cfg)
    missing = validate_config(enriched)
    # 必填字段应全
    for field in REQUIRED_64:
        assert field in enriched, f"missing §64 field {field}"
    assert missing == [], f"validate_config 应返回空列表，得到 {missing}"

    # §65 metrics 补全
    per_method = [
        {"method": "student", "score": 0.5},
        {"method": "teacher", "score": 0.8},
        {"method": "base_plus_adv", "score": 0.66, "chg": 0.16, "tgrr": 0.53, "retention": 1.32, "jcr_vs_student": 0.9},
    ]
    m = enrich_metrics({}, per_method)
    assert m["student_score"] == 0.5
    assert m["teacher_score"] == 0.8
    assert m["handoff_score"] == 0.66
    assert m["chg"] == 0.16
    assert m["tgrr"] == 0.53

    # 写 metadata.json
    (tmp_path / "t09").mkdir()
    write_run_metadata(cfg, {"chg": 0.16}, tmp_path / "t09", per_method=per_method)
    assert (tmp_path / "t09" / "metadata.json").exists()

    # §52 compliance 与 runtime signals 联合
    track("silent_re_prefill_on_failure", False)  # 我们故意设置 False
    sig = {**collect(), **infer_signals_from_cfg(cfg)}
    v = check_all(cfg, tmp_path, sig)
    rep = compliance_report(v)
    assert rep["passed"] is True or all(
        vv.rule_id != "§52.2" for vv in v
    ), "cfg.freeze=True 时不应有 §52.2 违反"


# ---- §62 Figure 7 数据流（bug-7 修复 B7）----


def test_t12_metrics_carries_geometry_and_fig7_renders(tmp_path):
    """§62 Figure 7 数据流：T12 的 metrics 必须携带 "geometry" 键，fig7 才能渲染。

    Given: 最小 T12 配置跑 run_geometry_diagnostics（真实数据流，含 geometry.json/metrics.json 写出）
    When:  fig7a_cka_heatmap 用 T12 返回的 metrics 渲染
    Then:  ① metrics.json（run_dir）含 "geometry" 键（与 geometry.json 一致）
           ② fig7a.png 生成且 > 1KB（修复前 geometry 缺 → 空 rows → 提前 return → 无图 → RED）
    """
    from apcs.figures import fig7a_cka_heatmap
    from apcs.geometry.runner import run_geometry_diagnostics

    cfg = {
        "teacher": {"num_layers": 4},
        "student": {"num_layers": 3},
        "hidden_dim": 16,
        "geometry_rank": 4,
    }
    res = run_geometry_diagnostics(cfg, tmp_path)
    assert res["status"] == "OK"

    # ① 落盘的 metrics.json 必须带 geometry（§62 Figure 7 数据流契约）
    disk_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "geometry" in disk_metrics, "metrics.json 缺 geometry 键 → fig7 拿到空 dict"
    assert disk_metrics["geometry"]["per_layer"] == res["geometry"]["per_layer"]

    # ② 渲染 fig7a 必须产出有效 png（> 1KB），而不是空图/无文件
    out = tmp_path / "fig7a.png"
    fig7a_cka_heatmap(res["metrics"], out)
    assert out.exists(), "fig7a.png 未生成 → geometry 键缺失导致提前 return"
    assert out.stat().st_size > 1024, f"fig7a.png 过小（{out.stat().st_size} B）→ 渲染异常"


def test_t12_real_hidden_states_use_token_geometry(tmp_path):
    """真实 hidden artifact 的 CKA/有效秩必须基于 token×hidden，而非 PCA 基本身。"""
    from apcs.geometry.runner import run_geometry_diagnostics

    rng = np.random.default_rng(11)
    hidden = rng.standard_normal((2, 24, 8))
    artifact = tmp_path / "hidden.npz"
    np.savez(artifact, teacher=hidden, student=hidden.copy())
    run_dir = tmp_path / "geometry"
    run_dir.mkdir()
    result = run_geometry_diagnostics(
        {
            "teacher": {"num_layers": 2},
            "student": {"num_layers": 2},
            "geometry_rank": 3,
            "hidden_states_path": str(artifact),
        },
        run_dir,
    )
    rows = result["geometry"]["per_layer"]
    assert result["geometry"]["placeholder"] is False
    assert all(row["cka"] == pytest.approx(1.0, abs=1e-9) for row in rows)
    assert all(row["principal_angle"] == pytest.approx(0.0, abs=1e-7) for row in rows)
    # 随机 token 表示的有效秩通常高于 geometry_rank=3；若等于 3，说明仍在算 PCA 基。
    assert all(row["effective_rank_t"] > 3.0 for row in rows)
