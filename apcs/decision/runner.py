"""T11 MVP Decision（design.md §39 / §69）。

═══════════════════════════════════════════════════════════════════════════════
读入前序任务产物（T05 / T09 / T10）并输出三类结论之一：
    A. GO — Runtime Capability Transfer
    B. GO — Efficient State Handoff
    C. STOP / REDESIGN

§69 判定规则（严格顺序）：
    D:  retention < 0.80                                   → Stop / Redesign
    A:  retention ≥ 0.90 AND chg > 0 AND tgrr > 0 AND psr_a > 0
    B:  retention ≥ 0.90 AND chg <= 0 AND psr_a > 0
    C:  retention ≥ 0.90 AND chg <= 0                        → Mechanism Boundary
    else: Inconclusive

命名约定：run_id 形如 `<name>-<timestamp>` 或 `<name>-<seed>-<timestamp>`。
一个完整 experiment 内所有 task 共享同一 run_id；T11 读
`<base>/<shared_run_id>/<task>/metrics.json`。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..io.runs import write_json, write_text


def _read_metrics(base_dir: Path, task: str, shared_run_id: str | None) -> dict | None:
    """从 base_dir/<shared_run_id>/<task>/metrics.json 读取。

    若 shared_run_id 为 None：遍历 base_dir 下所有 run_id，挑出含目标 task
    metrics 的最近修改时间的那个。
    """
    if shared_run_id is not None:
        p = base_dir / shared_run_id / task / "metrics.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None
    if not base_dir.exists():
        return None
    candidates: list[Path] = []
    for run in base_dir.iterdir():
        if not run.is_dir():
            continue
        if (run / task / "metrics.json").exists():
            candidates.append(run)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads((candidates[0] / task / "metrics.json").read_text(encoding="utf-8"))


def _read_system(base_dir: Path, shared_run_id: str | None) -> dict | None:
    """T10 的 system 写在 system.json 而非 metrics.json，单独处理。"""
    if shared_run_id is not None:
        p = base_dir / shared_run_id / "t10" / "system.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None
    if not base_dir.exists():
        return None
    candidates: list[Path] = []
    for run in base_dir.iterdir():
        if not run.is_dir():
            continue
        if (run / "t10" / "system.json").exists():
            candidates.append(run)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads((candidates[0] / "t10" / "system.json").read_text(encoding="utf-8"))


def decide(retention: float, chg: float, tgrr: float, psr_a: float) -> str:
    """§69 严格顺序的判定。"""
    if retention < 0.80:
        return "D_STOP_REPLACEABILITY_UNSTABLE"
    if retention >= 0.90 and chg > 0 and tgrr > 0 and psr_a > 0:
        return "A_RUNTIME_CAPABILITY_TRANSFER"
    if retention >= 0.90 and chg <= 0 and psr_a > 0:
        return "B_EFFICIENT_STATE_HANDOFF"
    if retention >= 0.90 and chg <= 0:
        return "C_MECHANISM_BOUNDARY"
    return "C_INCONCLUSIVE"


def run_mvp_decision(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    base = Path(cfg["output"]["base_dir"])
    # 命名约定（本文件 docstring L16-18 / design.md §39）：run_id 形如
    # `<name>-<timestamp>`；一个 experiment 内所有 task 共享同一 run_id，
    # 报告根为 `<base>/<run_id>/`。CLI（apcs/cli.py L133-135）把每个 task 的
    # run_dir 建为 `<base>/<run_id>/<task>/`，因此 run_id 就是
    # run_dir.parent.name。
    # bug-6 修复：原来误用 run_dir.name（即 "t11"）做正则匹配，恒不匹配 →
    # shared_run_id 恒 None → 落入 _read_metrics 的 mtime fallback，
    # 多实验并存时可能读到另一个 run 的数据。正则在此仅作 run_id 形状校验，
    # shared_run_id 取完整目录名（_read_metrics 用其拼路径，
    # 取 group(1) 的 `<name>` 前缀将无法解析到 `<base>/<run_id>/`）。
    parent_name = run_dir.parent.name
    if re.match(r"^(.+)-\d{8}-\d{6}$", parent_name):
        shared_run_id = parent_name
    elif re.match(r"^(.+)-\d{8}-\d{6}$", run_dir.name):
        # 兼容 run_dir 直接是 run_id 根目录（无 task 子目录）的情形
        shared_run_id = run_dir.name
    else:
        shared_run_id = None

    t05 = _read_metrics(base, "t05", shared_run_id)
    t09 = _read_metrics(base, "t09", shared_run_id)
    t10_sys = _read_system(base, shared_run_id)

    retention = (
        float(t05.get("mean_retention", 0.0)) if isinstance(t05, dict) else 0.0
    )
    chg_val = 0.0
    tgrr_val = 0.0
    if isinstance(t09, dict):
        for row in t09.get("per_method", []):
            if row["method"] == "base_plus_adv":
                chg_val = float(row["chg"])
                tgrr_val = float(row["tgrr"])
    psr_a_val = 0.0
    if isinstance(t10_sys, dict):
        pc = t10_sys.get("per_context", [])
        if pc:
            psr_a_val = float(pc[0]["psr_a_p50"])

    verdict = decide(retention, chg_val, tgrr_val, psr_a_val)

    md = (
        "# T11 MVP Decision\n\n"
        f"- Shared run_id pattern: `{shared_run_id or '(auto-detect)'}`\n"
        f"- Retention (T05): {retention:.4f}\n"
        f"- CHG (T09 Base+Adv): {chg_val:+.4f}\n"
        f"- TGRR (T09 Base+Adv): {tgrr_val:+.4f}\n"
        f"- PSR_A (T10 p50, smallest ctx): {psr_a_val:.3f}\n\n"
        f"## Verdict: **{verdict}**\n\n"
    )
    if verdict.startswith("A"):
        md += "Runtime Capability Transfer via KV State Handoff.\n"
    elif verdict.startswith("B"):
        md += "Efficient Cross-Model State Handoff (paper path B).\n"
    elif verdict.startswith("C_MECHANISM"):
        md += "State Compatibility does not imply Capability Compatibility.\n"
    elif verdict.startswith("C_INCONCLUSIVE"):
        md += "Inconclusive；继续完善 T07/T08/T09 后再判。\n"
    else:
        md += "Stop / Redesign — Replacement 不稳，优先 Alignment / Geometry。\n"

    metrics = {
        "task": "T11",
        "retention": retention,
        "chg": chg_val,
        "tgrr": tgrr_val,
        "psr_a": psr_a_val,
        "verdict": verdict,
        "sources": {
            "t05_found": t05 is not None,
            "t09_found": t09 is not None,
            "t10_found": t10_sys is not None,
        },
    }
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "summary.md", md)
    return {"status": "OK", "metrics": metrics, "summary": md}