"""§56-§62 Figure 渲染端到端测试（W6）。

目的：把所有 figN_* 函数喂给一份『真实可用的 metrics dict』，断言每张
该出的图确实落盘到 `out_dir/<name>.png`，且数据缺失的图静默跳过（不抛）。

不是 doc-quote 验证 —— 是真把数据流过 matplotlib 渲染。

数据构造：
- t06_metrics.rows[]          → 3 个 rank 档（PCR < 1，retention 0.7/0.85/0.95）
- t09_metrics.per_method[]    → 5 方法, 含 chg_bootstrap.point / gap_strata
- t10_metrics.per_context[]   → 3 context（128, 256, 512），双 Y 轴
- t12_metrics.geometry.per_layer[] → (n_s × n_t) 个 CKA / PA / ER / attn-cos

依赖：matplotlib（`pip install -e ".[dev]"`）。
"""
from __future__ import annotations

import importlib

import pytest

# matplotlib 是 [dev] / [figures] 的可选依赖；缺则整个文件 skip
_matplotlib = pytest.importorskip("matplotlib")

from apcs.figures import render_all


def _t05_metrics() -> dict:
    return {
        "task": "T05",
        "mean_retention": 0.756,
        "retention_K": 0.756,
        "retention_V": 0.756,
        "gate1": "FAIL",
    }


def _t06_metrics() -> dict:
    return {
        "task": "T06",
        "rows": [
            {"variant": "lowrank-8", "rank": 8, "pcr": 0.15, "retention": 0.70},
            {"variant": "lowrank-16", "rank": 16, "pcr": 0.30, "retention": 0.85},
            {"variant": "shared-basis-32", "rank": 32, "pcr": 0.55, "retention": 0.95},
        ],
    }


def _t07_metrics() -> dict:
    return {"task": "T07", "gap_distribution": {"low": 0.10, "med": 0.20, "high": 0.30}}


def _t09_metrics() -> dict:
    return {
        "task": "T09",
        "per_method": [
            {"method": "base_plus_adv", "chg": 0.17, "tgrr": 0.56, "score": 0.67},
        ],
        "chg_bootstrap": {"point": 0.17, "ci_low": 0.14, "ci_high": 0.21, "ci": 0.95},
        "gap_strata": {
            "low": {"base_plus_adv_chg": 0.15, "base_plus_adv_tgrr": 0.54},
            "medium": {"base_plus_adv_chg": 0.16, "base_plus_adv_tgrr": 0.56},
            "high": {"base_plus_adv_chg": 0.18, "base_plus_adv_tgrr": 0.57},
        },
    }


def _t10_metrics() -> dict:
    return {
        "task": "T10",
        "per_context": [
            {"context": 128, "psr_a_p50": 0.60, "cost_b_p50": 11.0},
            {"context": 256, "psr_a_p50": 0.55, "cost_b_p50": 22.0},
            {"context": 512, "psr_a_p50": 0.50, "cost_b_p50": 44.0},
        ],
    }


def _t12_metrics() -> dict:
    # 8 Student 层 × 4 Teacher 层
    rows = []
    for s in range(8):
        for t in range(4):
            rows.append(
                {
                    "student_layer": s,
                    "teacher_layer": t,
                    "cka": 0.5 + 0.1 * s - 0.1 * t,
                    "principal_angle": 0.3 + 0.05 * (s - t),
                    "attn_output_cosine": 0.6 + 0.02 * s,
                    "effective_rank_s": 4.0 + s * 0.5,
                }
            )
    return {"task": "T12", "geometry": {"per_layer": rows}}


def _multiturn_metrics() -> dict:
    return {
        "per_turn": [
            {"turn": 1, "chg_mean": 0.17, "kl_mean": 0.05, "jcr_mean": 0.66},
            {"turn": 5, "chg_mean": 0.15, "kl_mean": 0.06, "jcr_mean": 0.65},
            {"turn": 10, "chg_mean": 0.14, "kl_mean": 0.07, "jcr_mean": 0.64},
        ],
    }


def _ablation_metrics() -> dict:
    return {
        "rows": [
            {"ablation": "A1", "setting": "rank-8", "mean_retention": 0.80},
            {"ablation": "A1", "setting": "rank-16", "mean_retention": 0.85},
            {"ablation": "A1", "setting": "rank-32", "mean_retention": 0.93},
            {"ablation": "A6", "setting": "de_rope-off", "mean_retention": 0.75},
            {"ablation": "A6", "setting": "de_rope-on", "mean_retention": 0.88},
            {"ablation": "A8", "setting": "kv-adapter-off", "mean_retention": 0.82},
            {"ablation": "A8", "setting": "kv-adapter-on", "mean_retention": 0.92},
        ],
    }


def test_render_all_eight_figures(tmp_path):
    """§56-§62：喂齐 8 张图所需的 metrics dict，断言全部落盘。

    锁定的预期文件清单（与 apcs/figures/__init__.py docstring 一致）：
        fig1.png / fig2.png / fig3.png / fig4.png /
        fig5.png / fig6.png / fig7a.png / fig7bcd.png
    任何一张在该喂齐的 inputs 下**没**落盘 → 数据流断 → 报告 bug。
    """
    artifacts = {
        "t05": _t05_metrics(),
        "t06": _t06_metrics(),
        "t07": _t07_metrics(),
        "t09": _t09_metrics(),
        "t10": _t10_metrics(),
        "t12": _t12_metrics(),
        "multiturn": _multiturn_metrics(),
        "ablation": _ablation_metrics(),
    }
    out_dir = tmp_path / "figures"

    render_all(artifacts, out_dir)

    expected = [
        "fig1.png",
        "fig2.png",
        "fig3.png",
        "fig4.png",
        "fig5.png",
        "fig6.png",
        "fig7a.png",
        "fig7bcd.png",
    ]
    missing = [n for n in expected if not (out_dir / n).exists()]
    assert not missing, f"以下 figure 数据流断裂，未生成：{missing}"


def test_render_all_silent_skip_when_data_missing(tmp_path):
    """数据缺失时 figN_* 函数必须静默跳过（§75 诚实性：不当异常）。

    若任意 fig 函数在输入空 dict / 缺关键键时抛异常，会把单点失败
    扩散到整套渲染，违反 render_all 的"幂等"契约（figures/__init__.py L24）。
    """
    out_dir = tmp_path / "figures"
    # 只给空 metrics；应该全部静默返回，不抛
    render_all(
        {"t05": {}, "t06": {}, "t07": {}, "t09": {}, "t10": {},
         "t12": {}, "multiturn": {}, "ablation": {}},
        out_dir,
    )
    # 无 inputs → 全部 8 张都该跳过，out_dir 应当不存在或为空
    if out_dir.exists():
        # 严格：目录里不应有任何 png（跳过的图不落空图）
        pngs = list(out_dir.glob("*.png"))
        assert pngs == [], f"数据缺失时应静默跳过，落了 png：{pngs}"


def test_render_all_partial_data_runs_subset(tmp_path):
    """部分数据可用时只渲染子集：把 t06 + t09 单喂，验证 fig1 + fig2 + fig3 落盘。

    这是 §13 / §63 DAG 的典型场景：早期 task PASS 后的增量渲染。
    """
    out_dir = tmp_path / "figures"
    artifacts = {
        "t06": _t06_metrics(),
        "t09": _t09_metrics(),
    }
    render_all(artifacts, out_dir)

    # 至少 fig1(fig6 用 t06) 与 fig2(t09.gap_strata) 该出
    assert (out_dir / "fig1.png").exists(), "fig1 应在 t06 有 rows 时落盘"
    assert (out_dir / "fig2.png").exists(), "fig2 应在 t09 有 gap_strata 时落盘"
    # fig7 缺 t12 → 不出
    assert not (out_dir / "fig7a.png").exists(), "缺 t12 时 fig7a 不该出"