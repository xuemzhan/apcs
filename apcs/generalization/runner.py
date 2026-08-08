"""T13 Generalization runner。

读所有已完成 Run 的 T05/T09 metrics，列出每个 (pair, seed) 的 CHG 与 Retention，
对照 §69 Path A 判定各 Pair 是否通过 Gate 2A。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..io.runs import write_json, write_text


def _load_run_metrics(base_dir: Path, run_id: str, task: str) -> dict | None:
    p = base_dir / run_id / task / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def run_generalization(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T13 入口：扫描所有 run_id，列出 (pair, run_id) 的 T05/T09 结果。"""
    base = Path(cfg["output"]["base_dir"])
    if not base.exists():
        return {"status": "OK", "metrics": {}, "summary": "无 Run 数据"}

    rows = []
    for run in sorted(base.iterdir()):
        if not run.is_dir():
            continue
        t05 = _load_run_metrics(base, run.name, "t05")
        t09 = _load_run_metrics(base, run.name, "t09")
        if not t05 or not t09:
            continue
        chg = 0.0
        tgrr = 0.0
        for row in t09.get("per_method", []):
            if row["method"] == "base_plus_adv":
                chg = float(row["chg"])
                tgrr = float(row["tgrr"])
        ret = float(t05.get("mean_retention", 0.0))
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
        "gate2a_rate": n_pass / max(1, n_total),
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
        md += (
            "\n⚠️ 仅 < 2 个 Pair 评估。论文要求 §69 Path A 需在 ≥ 2 个 Large→Small "
            "Pair 上同时成立（Table 1）。请运行第二 Pair (`--config configs/pair_smollm2.yaml`)。\n"
        )
    write_text(run_dir / "summary.md", md)
    return {"status": "OK", "metrics": metrics, "summary": md}