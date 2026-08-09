"""T13 Generalization runner（design.md §11 / §44 / §69）。

═══════════════════════════════════════════════════════════════════════════════
泛化评测：能力迁移不能只在单对模型上成立。T13 扫描 base_dir 下所有已完成
Run 的 T05/T09 metrics，对每个 (pair, seed) 判定是否满足 §69 Path A 的
Gate 2A 条件（retention ≥ 0.90 且 CHG > 0 且 TGRR > 0），再汇总
n_pairs_pass_gate2a / gate2a_rate。

为什么这样算：
    - 一个 Run = 一个 Large→Small 模型 Pair（一个 run_id）；
    - retention 取自 T05（Gate 1 替代保真度），CHG/TGRR 取自 T09 中
      base_plus_adv（最强候选方法）——这正是论文主结论的口径；
    - 论文要求（Table 1）：≥ 2 个 Pair 同时通过 Gate 2A 才支持"泛化"结论，
      < 2 个 Pair 时本模块在 summary 中显式告警。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
from pathlib import Path

from ..io.runs import write_json, write_text


def _load_run_metrics(base_dir: Path, run_id: str, task: str) -> dict | None:
    """读取 reports/runs/<run_id>/<task>/metrics.json；文件不存在返回 None。

    任务名用小写（t05 / t09），与 orchestrator 的产物目录约定一致（§63）。
    """
    p = base_dir / run_id / task / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


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

    rows = []
    for run in sorted(base.iterdir()):
        if not run.is_dir():
            continue
        t05 = _load_run_metrics(base, run.name, "t05")
        t09 = _load_run_metrics(base, run.name, "t09")
        if not t05 or not t09:
            continue  # 只统计 T05/T09 都完成的 Run（Gate 1 + Gate 2A 的完整产物）
        chg = 0.0
        tgrr = 0.0
        for row in t09.get("per_method", []):
            if row["method"] == "base_plus_adv":  # 最强候选方法，即论文主结论口径
                chg = float(row["chg"])
                tgrr = float(row["tgrr"])
        ret = float(t05.get("mean_retention", 0.0))
        # §69 Path A 判定：Retention≥0.90（Gate 1）+ CHG>0 + TGRR>0（Gate 2A）
        passed_a = (ret >= 0.90 and chg > 0 and tgrr > 0)
        rows.append(
            {
                "run_id": run.name,
                "retention": ret,
                "chg": chg,
                "tgrr": tgrr,
                "gate2a_pass": passed_a,
            }
        )

    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["gate2a_pass"])
    summary = {
        "n_pairs_evaluated": n_total,
        "n_pairs_pass_gate2a": n_pass,
        "rows": rows,
    }
    metrics = {
        "task": "T13",
        "n_pairs_evaluated": n_total,
        "n_pairs_pass_gate2a": n_pass,
        # §11 / §44 泛化通过率（论文 Table 1 口径）
        "gate2a_rate": n_pass / max(1, n_total),  # max(1,·) 防 0 个 Pair 时除零
    }
    write_json(run_dir / "metrics.json", metrics)

    md = (
        "# T13 Generalization\n\n"
        f"- Pairs evaluated: {n_total}\n"
        f"- Pairs passing Gate 2A: {n_pass}\n\n"
        "| run_id | retention | CHG | TGRR | gate2a |\n"
        "| ------ | --------: | --: | ---: | ------ |\n"
        + "\n".join(
            f"| {r['run_id']} | {r['retention']:.4f} | {r['chg']:+.4f} | "
            f"{r['tgrr']:+.4f} | {'PASS' if r['gate2a_pass'] else 'FAIL'} |"
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