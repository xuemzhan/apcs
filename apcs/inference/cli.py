"""inject-eval CLI 入口（§52 zero-prefill / §53 单卡执行顺序）。

═══════════════════════════════════════════════════════════════════════════════
run_inject_eval(cfg, run_dir) → dict

Runner 契约：runner(cfg, run_dir) → dict，CLI 通过 run_with_compliance 包装调用。

本入口函数负责：
    1. 加载数据集（hf_dataset.load 或 synthetic_fidelity_set 回退）
    2. 构造 InjectionEvaluator
    3. 执行四方法评估（student_self / teacher_full / text_handoff / ridge_handoff）
    4. 写入两份 JSON artifact + 可选 hidden_states.npz
    5. 返回 status / metrics / summary 供 CLI 落盘

§53 单卡执行顺序在 InjectionEvaluator.evaluate() 内部保证：
    Teacher Load → Teacher Unload → CUDA Cleanup → Student Load → Student Unload

产物清单（§63 Run 产物规范）：
    <run_dir>/inject_eval/
        replacement_score_artifact.json   — 供 T05 消费
        capability_score_artifact.json    — 供 T09 消费
        hidden_states.npz (可选)          — 供 T12 geometry diagnostics
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_sample_rows(cfg: dict[str, Any]) -> list[Any]:
    """按 cfg.datasets 加载评估样本。

    优先用 hf_dataset.load（真实数据），失败回退到 synthetic_fidelity_set。
    样本数由 cfg.inject_eval.max_samples 控制（默认 32）。
    """
    ie_cfg = cfg.get("inject_eval", {})
    max_n = int(ie_cfg.get("max_samples", 32))
    seed = int(ie_cfg.get("seed", 0))

    # 尝试真实数据集加载
    ds_cfg = cfg.get("datasets", {})
    fidelity = ds_cfg.get("fidelity", [])
    teacher_adv = ds_cfg.get("teacher_advantage", {})
    names = list(fidelity) if isinstance(fidelity, list) else ([fidelity] if fidelity else [])
    if not names and teacher_adv.get("primary"):
        names = [str(teacher_adv["primary"])]

    for name in names:
        try:
            from ..data.hf_dataset import load as load_ds

            rows = load_ds(name, n=max_n, split="test", seed=seed)
            if rows:
                logger.info("[inject-eval] Loaded %d samples from dataset %r", len(rows), name)
                return rows
        except Exception as e:  # noqa: BLE001
            logger.warning("[inject-eval] Dataset %r load failed: %s", name, e)

    # 回退到合成数据
    from ..data import synthetic_fidelity_set

    rows = synthetic_fidelity_set(n=min(max_n, 8))
    logger.info("[inject-eval] Using synthetic fidelity set (%d samples)", len(rows))
    return rows


def run_inject_eval(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """inject-eval 主入口（§52 zero-prefill / §53 单卡执行顺序）。

    Runner 契约：runner(cfg, run_dir) → dict，CLI 通过 run_with_compliance 包装调用。

    返回：
        {
            "status": "PASS" | "OK" | "FAIL",
            "metrics": {...},
            "summary": "Markdown 表格摘要",
        }

    §75 诚实性：
        - 需要 GPU + torch/transformers → 缺失时显式 raise（不伪造分数）
        - 四方法评估全程在 InjectionEvaluator 内执行
        - 产物校验：非有限 score → RuntimeError
    """
    from .evaluator import InjectionEvaluator

    evaluator = InjectionEvaluator(cfg, run_dir)

    # 加载样本
    sample_rows = _load_sample_rows(cfg)
    if not sample_rows:
        raise RuntimeError("inject-eval: 无可用样本（数据集为空）")

    # 执行四方法评估
    logger.info("[inject-eval] Starting evaluation with %d samples", len(sample_rows))
    eval_result = evaluator.evaluate(sample_rows)

    # 写入 artifacts
    evaluator.write_artifacts(eval_result)

    # 构造 CLI 返回
    replacement = eval_result["replacement_artifact"]
    capability = eval_result["capability_artifact"]
    n_replacement = len(replacement.get("records", []))
    n_capability = len(capability.get("records", []))
    zero_prefill_ok = eval_result["zero_prefill_verified"]

    # 统计各方法均分
    method_scores: dict[str, list[float]] = {}
    for rec in capability.get("records", []):
        method = rec.get("method", "unknown")
        method_scores.setdefault(method, []).append(float(rec.get("score", 0.0)))
    method_means = {m: sum(v) / len(v) if v else 0.0 for m, v in method_scores.items()}

    # 计算 CHG = ridge_score - student_score
    ridge_mean = method_means.get("ridge", 0.0)
    student_mean = method_means.get("student", 0.0)
    chg = ridge_mean - student_mean

    metrics = {
        "n_samples": n_replacement,
        "n_capability_records": n_capability,
        "zero_prefill_verified": zero_prefill_ok,
        "method_means": method_means,
        "chg": float(chg),
        "ridge_score_mean": float(ridge_mean),
        "student_score_mean": float(student_mean),
    }

    # Status 判定
    if not zero_prefill_ok:
        status = "FAIL"
        status_reason = "§52 zero-prefill 验证失败"
    elif n_replacement == 0:
        status = "FAIL"
        status_reason = "无有效评估记录"
    else:
        status = "PASS"
        status_reason = "四方法评估完成，zero-prefill 验证通过"

    # Summary（Markdown 表格）
    summary_lines = [
        "# inject-eval 结果",
        "",
        f"- **状态**: {status}（{status_reason}）",
        f"- **样本数**: {n_replacement}",
        f"- **Zero-prefill 验证**: {'✅ 通过' if zero_prefill_ok else '❌ 失败'}",
        "",
        "## 各方法均分",
        "",
        "| 方法 | 均分 |",
        "|------|------|",
    ]
    for method in ["student", "teacher", "text", "ridge"]:
        mean_val = method_means.get(method, 0.0)
        summary_lines.append(f"| {method} | {mean_val:.4f} |")
    summary_lines.append(f"| **CHG** (ridge - student) | **{chg:+.4f} |")
    summary = "\n".join(summary_lines)

    logger.info("[inject-eval] Done. status=%s chg=%+.4f", status, chg)

    return {
        "status": status,
        "metrics": metrics,
        "summary": summary,
    }


__all__ = ["run_inject_eval"]
