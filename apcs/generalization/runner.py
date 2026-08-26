"""T13 Generalization runner（design.md §11 / §44 / §69）。

═══════════════════════════════════════════════════════════════════════════════
泛化评测：能力迁移不能只在单对模型上成立。T13 扫描 base_dir 下所有已完成
Run 的 T05/T09 metrics，对每个模型 ID+revision 聚合所有运行，判定是否满足
§69 Path A 的 Gate 2A 条件（retention ≥ 0.90 且 CHG > 0 且 TGRR > 0），再汇总
n_pairs_pass_gate2a / gate2a_rate。

为什么这样算：
    - 一个模型对可以有多个 run_id，任何失败运行都不会被挑选规则隐藏；
    - retention 取自 T05（Gate 1 替代保真度），CHG/TGRR 取自 T09 中
      base_plus_adv（最强候选方法）——这正是论文主结论的口径；
    - 每个 Pair 默认还要求至少 3 个唯一 seed；论文要求（Table 1）：
      ≥ 2 个 Pair 同时通过 Gate 2A 才支持"泛化"结论，
      < 2 个 Pair 时本模块在 summary 中显式告警。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from ..io.runs import write_json, write_text
from ..evidence import is_measured_task


def _load_run_metrics(base_dir: Path, run_id: str, task: str) -> dict | None:
    """读取 reports/runs/<run_id>/<task>/metrics.json；文件不存在返回 None。

    任务名用小写（t05 / t09），与 orchestrator 的产物目录约定一致（§63）。
    """
    p = base_dir / run_id / task / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _pair_identity(run: Path) -> tuple[str, str, str]:
    """从 T05 冻结配置提取模型对；不把两次 seed/run 误算为两个 pair。"""
    config_path = run / "t05" / "config.json"
    if not config_path.exists():
        return (f"unknown::{run.name}", "unknown", "unknown")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    teacher_cfg = config.get("teacher", {})
    student_cfg = config.get("student", {})
    teacher = str(teacher_cfg.get("model_id", "unknown"))
    student = str(student_cfg.get("model_id", "unknown"))
    teacher_rev = str(teacher_cfg.get("revision", "default"))
    student_rev = str(student_cfg.get("revision", "default"))
    return (
        f"{teacher}@{teacher_rev}=>{student}@{student_rev}",
        f"{teacher}@{teacher_rev}",
        f"{student}@{student_rev}",
    )


def run_generalization(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T13 入口：扫描所有 run_id，列出 (pair, run_id) 的 T05/T09 结果。

    流程：
        1. 遍历 base_dir 下的每个 run 目录（一个 run = 一个 model pair）；
        2. 同时有 t05 和 t09 metrics 的 run 才进入评估（缺任一个跳过）；
        3. 从 t09.per_method 取 base_plus_adv 的 chg / tgrr，从 t05 取 mean_retention；
        4. 按 §69 Path A 判定 gate2a_pass，汇总通过数 / 通过率；
        5. <2 个 Pair 时在 summary 中明确警告（§11 要求 ≥2）。
    """
    base = Path(cfg["output"]["base_dir"])
    if not base.exists():
        # 空跑：还没有任何 Run 产物，返回空 metrics 而不是报错
        return {"status": "OK", "metrics": {}, "summary": "无 Run 数据"}

    pair_runs: dict[str, list[dict[str, Any]]] = {}
    n_runs_scanned = 0
    for run in sorted(base.iterdir()):
        if not run.is_dir():
            continue
        t05 = _load_run_metrics(base, run.name, "t05")
        t09 = _load_run_metrics(base, run.name, "t09")
        if not t05 or not t09:
            continue  # 只统计 T05/T09 都完成的 Run（Gate 1 + Gate 2A 的完整产物）
        n_runs_scanned += 1
        pair_key, teacher_id, student_id = _pair_identity(run)
        chg = 0.0
        tgrr = 0.0
        for row in t09.get("per_method", []):
            if row["method"] == "base_plus_adv":  # 最强候选方法，即论文主结论口径
                chg = float(row["chg"])
                tgrr = float(row["tgrr"])
        task_ret = t05.get("task_retention")
        ret = float(task_ret) if isinstance(task_ret, (int, float)) else 0.0
        evidence_ready = is_measured_task(t05) and is_measured_task(t09)
        # §69 Path A 判定：Retention≥0.90（Gate 1）+ CHG>0 + TGRR>0（Gate 2A）
        passed_a = evidence_ready and (ret >= 0.90 and chg > 0 and tgrr > 0)
        candidate = {
                "run_id": run.name,
                "pair": pair_key,
                "teacher_model_id": teacher_id,
                "student_model_id": student_id,
                "retention": ret,
                "kv_cosine_diagnostic": float(
                    t05.get("mean_kv_cosine", t05.get("mean_retention", 0.0))
                ),
                "chg": chg,
                "tgrr": tgrr,
                "evidence_ready": evidence_ready,
                "run_gate2a_pass": passed_a,
                "seeds": sorted({int(seed) for seed in t09.get("seeds", [])}),
            }
        # 同一 pair 的所有运行都保留，防止“挑第一条/最好一条”隐藏失败。
        pair_runs.setdefault(pair_key, []).append(candidate)

    min_seeds = int(cfg.get("generalization", {}).get("min_unique_seeds", 3))
    rows: list[dict[str, Any]] = []
    for pair_key, runs in pair_runs.items():
        measured = [row for row in runs if row["evidence_ready"]]
        unique_seeds = sorted({seed for row in runs for seed in row["seeds"]})
        all_runs_measured = len(measured) == len(runs)
        enough_seeds = len(unique_seeds) >= min_seeds
        gate2a_pass = (
            bool(runs)
            and all_runs_measured
            and enough_seeds
            and all(row["run_gate2a_pass"] for row in runs)
        )

        def _mean(key: str) -> float:
            return float(fmean(row[key] for row in measured)) if measured else 0.0

        def _std(key: str) -> float:
            return float(pstdev(row[key] for row in measured)) if len(measured) > 1 else 0.0

        rows.append(
            {
                "pair": pair_key,
                "teacher_model_id": runs[0]["teacher_model_id"],
                "student_model_id": runs[0]["student_model_id"],
                "n_runs": len(runs),
                "n_measured_runs": len(measured),
                "unique_seeds": unique_seeds,
                "n_unique_seeds": len(unique_seeds),
                "min_unique_seeds_required": min_seeds,
                "retention_mean": _mean("retention"),
                "retention_std": _std("retention"),
                "retention_min": min((row["retention"] for row in measured), default=0.0),
                "chg_mean": _mean("chg"),
                "chg_std": _std("chg"),
                "tgrr_mean": _mean("tgrr"),
                "tgrr_std": _std("tgrr"),
                "all_runs_measured": all_runs_measured,
                "gate2a_pass": gate2a_pass,
                "runs": runs,
            }
        )
    rows.sort(key=lambda row: row["pair"])

    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["gate2a_pass"])
    metrics = {
        "task": "T13",
        "n_runs_scanned": n_runs_scanned,
        "n_pairs_evaluated": n_total,
        "n_pairs_measured": sum(1 for r in rows if r["all_runs_measured"]),
        "n_pairs_pass_gate2a": n_pass,
        "min_unique_seeds_required": min_seeds,
        "rows": rows,
        # §11 / §44 泛化通过率（论文 Table 1 口径）
        "gate2a_rate": n_pass / max(1, n_total),  # max(1,·) 防 0 个 Pair 时除零
    }
    write_json(run_dir / "metrics.json", metrics)

    md = (
        "# T13 Generalization\n\n"
        f"- Pairs evaluated: {n_total}\n"
        f"- Pairs passing Gate 2A: {n_pass}\n\n"
        f"- Minimum unique seeds per pair: {min_seeds}\n\n"
        "| pair | runs | seeds | evidence | retention mean±std | CHG mean±std | TGRR mean±std | gate2a |\n"
        "| ---- | ---: | ----: | -------- | -----------------: | -----------: | ------------: | ------ |\n"
        + "\n".join(
            f"| {r['pair']} | {r['n_runs']} | {r['n_unique_seeds']} | "
            f"{'measured' if r['all_runs_measured'] else 'blocked'} | "
            f"{r['retention_mean']:.4f}±{r['retention_std']:.4f} | "
            f"{r['chg_mean']:+.4f}±{r['chg_std']:.4f} | "
            f"{r['tgrr_mean']:+.4f}±{r['tgrr_std']:.4f} | "
            f"{'PASS' if r['gate2a_pass'] else 'FAIL'} |"
            for r in rows
        )
        + "\n"
    )
    if n_total < 2:
        # §11：<2 个 Pair 不足以支撑泛化结论，显式提示运行第二 Pair
        md += (
            "\n⚠️ 仅 < 2 个 Pair 评估。论文要求 §69 Path A 需在 ≥ 2 个 Large→Small "
            "Pair 上同时成立（Table 1）。请运行第二 Pair (`--config configs/pair_smollm2.yaml`)。\n"
        )
    write_text(run_dir / "summary.md", md)
    return {"status": "OK", "metrics": metrics, "summary": md}
