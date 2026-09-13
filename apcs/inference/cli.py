"""inject-eval CLI 入口（§52 zero-prefill / §53 单卡执行顺序）。

═══════════════════════════════════════════════════════════════════════════════
run_inject_eval(cfg, run_dir) → dict

Runner 契约：runner(cfg, run_dir) → dict，CLI 通过 run_with_compliance 包装调用。

本入口函数负责：
    1. 加载数据集（hf_dataset.load；A4: 多数据集轮转采样，不再只取第一个）
    2. 构造 InjectionEvaluator
       - 在线模式：cfg.inject_eval.online_kv_dir 指向持久化 KV store 时，
         走 evaluate_online（全程不加载 Teacher —— 核心目标的冷启动半程）
       - 离线模式：evaluate（可 persist_kv 落盘供在线阶段消费）
    3. 执行四方法评估（student_self / teacher_full / text_handoff / ridge_handoff）
    4. A1 指标换轨：以 gold_prob / accuracy 为主口径计算 CHG/TGRR +
       bootstrap CI + permutation p；置信度口径保留为兼容字段
    5. 写入两份 JSON artifact + 可选 hidden_states.npz
    6. 返回 status / metrics / summary 供 CLI 落盘

§53 单卡执行顺序在 InjectionEvaluator.evaluate() 内部保证：
    Teacher Load → Teacher Unload → CUDA Cleanup → Student Load → Student Unload

产物清单（§63 Run 产物规范）：
    <run_dir>/inject_eval/
        replacement_score_artifact.json   — 供 T05 消费
        capability_score_artifact.json    — 供 T09 消费
        kv_store/                         — B1 持久化（persist_kv=True 时）
        hidden_states.npz (可选)          — 供 T12 geometry diagnostics
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _load_sample_rows(cfg: dict[str, Any]) -> list[Any]:
    """按 cfg.datasets 加载评估样本。

    A4 修复：fidelity 列表含多个数据集时**轮转采样**（每个数据集取
    max_n // len 个，交错合并后截断到 max_n），不再只取第一个成功的 ——
    此前声明 [hellaswag, arc_challenge, winogrande] 实际只跑 hellaswag，
    与 config 承诺不符。单个数据集加载失败跳过并告警（诚实降级）。
    """
    ie_cfg = cfg.get("inject_eval", {})
    max_n = int(ie_cfg.get("max_samples", 32))
    seed = int(ie_cfg.get("seed", 0))

    ds_cfg = cfg.get("datasets", {})
    fidelity = ds_cfg.get("fidelity", [])
    teacher_adv = ds_cfg.get("teacher_advantage", {})
    names = list(fidelity) if isinstance(fidelity, list) else ([fidelity] if fidelity else [])
    if not names and teacher_adv.get("primary"):
        names = [str(teacher_adv["primary"])]

    if names:
        per_ds = max(1, max_n // len(names))
        buckets: list[list[Any]] = []
        loaded_names: list[str] = []
        for name in names:
            try:
                from ..data.hf_dataset import load as load_ds

                # Long-context loaders need the configured target length; this
                # was previously dropped here, so needle runs silently fell back
                # to the 1,024-token default.
                extra: dict[str, Any] = {}
                _tt = ds_cfg.get("needle_target_tokens")
                if _tt and str(name) in ("needle_mcqa", "needle_longctx"):
                    extra["target_tokens"] = int(_tt)
                rows = load_ds(name, n=per_ds, split="test", seed=seed, **extra)
                if rows:
                    buckets.append(list(rows))
                    loaded_names.append(str(name))
                    logger.info(
                        "[inject-eval] Loaded %d samples from dataset %r", len(rows), name
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("[inject-eval] Dataset %r load failed: %s", name, e)
        if buckets:
            # 轮转交错：ds1[0], ds2[0], ..., ds1[1], ds2[1], ... 截断到 max_n
            merged: list[Any] = []
            for i in range(max(len(b) for b in buckets)):
                for b in buckets:
                    if i < len(b):
                        merged.append(b[i])
            if len(loaded_names) < len(names):
                logger.warning(
                    "[inject-eval] Only %d/%d datasets loaded: %s",
                    len(loaded_names), len(names), loaded_names,
                )
            return merged[:max_n]

    # 全部失败 → 合成数据兜底（如实标注）
    from ..data import synthetic_fidelity_set

    rows = synthetic_fidelity_set(n=min(max_n, 8))
    logger.info("[inject-eval] Using synthetic fidelity set (%d samples)", len(rows))
    return rows


def _paired_diffs(
    records: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    field: str,
) -> list[float]:
    """按 sample_id 配对取 method_a - method_b 的逐样本差值（跳过缺失/None）。"""
    vals: dict[str, dict[str, float]] = {}
    for rec in records:
        if rec.get("skipped"):
            continue
        sid = rec.get("sample_id")
        v = rec.get(field)
        if sid is None or v is None:
            continue
        vals.setdefault(sid, {})[rec.get("method", "")] = float(v)
    diffs = []
    for sid, m in vals.items():
        if method_a in m and method_b in m:
            diffs.append(m[method_a] - m[method_b])
    return diffs


def _method_field_map(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in records:
        if rec.get("skipped"):
            continue
        v = rec.get(field)
        if v is not None:
            out[rec.get("sample_id", "")] = float(v)
    return out


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
        - capability_gate 只看 gold 口径（accuracy/gold_prob），不看置信度
    """
    from .evaluator import InjectionEvaluator

    evaluator = InjectionEvaluator(cfg, run_dir)

    online_kv_dir = cfg.get("inject_eval", {}).get("online_kv_dir")
    if online_kv_dir:
        # ── B1 在线阶段：从持久化 KV store 冷启动（不加载 Teacher）──
        logger.info("[inject-eval] ONLINE mode from kv_store: %s", online_kv_dir)
        eval_result = evaluator.evaluate_online(Path(str(online_kv_dir)))
    else:
        sample_rows = _load_sample_rows(cfg)
        if not sample_rows:
            raise RuntimeError("inject-eval: 无可用样本（数据集为空）")
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
    provenance = capability.get("provenance", {})

    records = capability.get("records", [])
    methods = sorted({r.get("method", "unknown") for r in records})

    # ── A1: 每方法 accuracy / gold_prob / 置信度 / PPL 汇总 ──
    method_stats: dict[str, dict[str, Any]] = {}
    for m in methods:
        mrecs = [r for r in records if r.get("method") == m and not r.get("skipped")]
        confs = [float(r["score"]) for r in mrecs]
        golds = [float(r["gold_prob"]) for r in mrecs if r.get("gold_prob") is not None]
        ppls = [float(r["ppl"]) for r in mrecs if r.get("ppl") is not None
                and float(r["ppl"]) != float("inf")]
        corr = [1.0 if r.get("correct") else 0.0 for r in mrecs if "correct" in r]
        lcos = [float(r["logit_cos_vs_student"]) for r in mrecs
                if r.get("logit_cos_vs_student") is not None]
        method_stats[m] = {
            "total": len(mrecs),
            "accuracy": float(np.mean(corr)) if corr else None,
            "gold_prob_mean": float(np.mean(golds)) if golds else None,
            "conf_mean": float(np.mean(confs)) if confs else 0.0,
            "ppl_mean": float(np.mean(ppls)) if ppls else None,
            "logit_cos_mean": float(np.mean(lcos)) if lcos else None,
        }

    from ..metrics import bootstrap_ci, permutation_p

    # 逐消融方法的 gold / accuracy 口径 CHG + 显著性（主口径 = gold_prob）
    handoff_methods = [m for m in methods if m.startswith("ridge_")]
    chg_gold: dict[str, Any] = {}
    chg_acc: dict[str, Any] = {}
    for m in handoff_methods:
        diffs_g = _paired_diffs(records, m, "student", "gold_prob")
        diffs_a = _paired_diffs(records, m, "student", "correct")
        entry_g: dict[str, Any] = {}
        entry_a: dict[str, Any] = {}
        if diffs_g:
            pt, lo, hi = bootstrap_ci(diffs_g, n_boot=1000)
            p = permutation_p(diffs_g, n_perm=1000)
            entry_g = {"mean": pt, "ci_low": lo, "ci_high": hi, "p": p, "n": len(diffs_g)}
            # TGRR（gold 口径）：需要 teacher-gold 均值
            t_gold = method_stats.get("teacher", {}).get("gold_prob_mean")
            s_gold = method_stats.get("student", {}).get("gold_prob_mean")
            if t_gold is not None and s_gold is not None and (t_gold - s_gold) > 0:
                entry_g["tgrr"] = pt / (t_gold - s_gold)
        if diffs_a:
            pt, lo, hi = bootstrap_ci(diffs_a, n_boot=1000)
            p = permutation_p(diffs_a, n_perm=1000)
            entry_a = {"mean": pt, "ci_low": lo, "ci_high": hi, "p": p, "n": len(diffs_a)}
        chg_gold[m] = entry_g
        chg_acc[m] = entry_a

    primary = "ridge_kv_both" if "ridge_kv_both" in chg_gold else (
        handoff_methods[0] if handoff_methods else None
    )
    gate_entry = chg_gold.get(primary, {}) if primary else {}
    capability_gate = {
        "metric": "gold_prob",
        "method": primary,
        "chg": gate_entry.get("mean"),
        "ci_low": gate_entry.get("ci_low"),
        "ci_high": gate_entry.get("ci_high"),
        "p": gate_entry.get("p"),
        "gate": (
            "PASS"
            if gate_entry and gate_entry.get("ci_low") is not None
            and gate_entry["ci_low"] > 0 and gate_entry.get("p", 1.0) < 0.05
            else "FAIL"
        ),
    }

    # 兼容字段（T05/T09 消费方依赖）：置信度口径 CHG
    method_means = {m: method_stats[m]["conf_mean"] for m in methods}
    ppl_means = {m: method_stats[m]["ppl_mean"] for m in methods
                 if method_stats[m]["ppl_mean"] is not None}
    ridge_kv_both_mean = method_means.get("ridge_kv_both", 0.0)
    student_mean = method_means.get("student", 0.0)
    chg_conf = ridge_kv_both_mean - student_mean

    psr_summary = eval_result.get("psr_summary", {})
    kv_diag = eval_result.get("kv_diagnostics", {})

    metrics = {
        "n_samples": n_replacement,
        "n_capability_records": n_capability,
        "zero_prefill_verified": zero_prefill_ok,
        "protocol_version": provenance.get("protocol_version", "unknown"),
        "git_hash": provenance.get("git_hash", "unknown"),
        "online": provenance.get("online", False),
        "calib_eval_disjoint": provenance.get("calib_eval_disjoint"),
        "rope_align": provenance.get("rope_align"),
        # A1: 主口径（gold / accuracy）
        "method_stats": method_stats,
        "chg_gold": chg_gold,
        "chg_accuracy": chg_acc,
        "capability_gate": capability_gate,
        # B2: 系统收益
        "psr": psr_summary,
        # P0.6: KV 范数诊断（教师/学生尺度比；ratio_mean 偏离 1 越远 ⇒ 尺度失配越重）
        "kv_norm_diagnostics": kv_diag,
        # 兼容字段（置信度口径，仅供历史对照，不作能力结论依据）
        "method_means": method_means,
        "ppl_means": ppl_means,
        "chg": float(chg_conf),
        "ridge_kv_both_score_mean": float(ridge_kv_both_mean),
        "student_score_mean": float(student_mean),
    }

    # Status 判定（执行语义，与能力 gate 分离 —— §75 不混为一谈）
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
    show_methods = [
        m for m in ["teacher", "student", "text",
                    "ridge_self_kv", "ridge_native", "ridge_k_only",
                    "ridge_v_only", "ridge_kv_both"]
        if m in method_stats
    ] + [m for m in methods if m not in
         ["teacher", "student", "text",
          "ridge_self_kv", "ridge_native", "ridge_k_only",
          "ridge_v_only", "ridge_kv_both"]]
    summary_lines = [
        "# inject-eval 结果",
        "",
        f"- **状态**: {status}（{status_reason}）",
        f"- **样本数**: {n_replacement}"
        f"（calib/eval 互斥: {provenance.get('calib_eval_disjoint')}）",
        f"- **Zero-prefill 验证**: {'✅ 通过' if zero_prefill_ok else '❌ 失败'}"
        f"（真实 cache 审计，协议 v{metrics['protocol_version']}）",
        f"- **能力 Gate（gold 口径）**: {capability_gate['gate']}"
        f"（{primary}: chg={capability_gate.get('chg')}, "
        f"ci_low={capability_gate.get('ci_low')}, p={capability_gate.get('p')}）",
    ]
    if psr_summary.get("mean") is not None:
        summary_lines.append(
            f"- **PSR**: mean={psr_summary['mean']:.3f}, median={psr_summary['median']:.3f}"
            f"（n={psr_summary['n']}，含测量开销偏保守）"
        )
    summary_lines += [
        "",
        "## 各方法指标",
        "",
        "| 方法 | accuracy | gold_prob | 置信度 | PPL |",
        "|------|----------|-----------|--------|-----|",
    ]
    for m in show_methods:
        st = method_stats[m]
        acc = f"{st['accuracy']:.3f}" if st["accuracy"] is not None else "-"
        gold = f"{st['gold_prob_mean']:.4f}" if st["gold_prob_mean"] is not None else "-"
        ppl = f"{st['ppl_mean']:.2f}" if st["ppl_mean"] is not None else "-"
        summary_lines.append(f"| {m} | {acc} | {gold} | {st['conf_mean']:.4f} | {ppl} |")
    if primary and chg_gold.get(primary):
        e = chg_gold[primary]
        summary_lines += [
            "",
            f"## CHG（gold 口径，{primary} − student）",
            "",
            f"- mean={e['mean']:+.4f}, 95% CI=[{e['ci_low']:+.4f}, {e['ci_high']:+.4f}], "
            f"p={e['p']:.4f}" + (f", TGRR={e['tgrr']:.3f}" if "tgrr" in e else ""),
        ]
    summary = "\n".join(summary_lines)

    logger.info("[inject-eval] Done. status=%s gate=%s", status, capability_gate["gate"])

    return {
        "status": status,
        "metrics": metrics,
        "summary": summary,
    }


__all__ = ["run_inject_eval"]
