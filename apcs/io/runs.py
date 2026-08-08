"""Run 产物写出工具（design.md §63）。

═══════════════════════════════════════════════════════════════════════════════
每个 Run 输出到 reports/runs/<run_id>/<task>/ 下：
    config.json          — 完整配置（含解析后的占位符）
    metrics.json         — 该 task 的指标
    system.json          — 系统成本（仅 T10 写）
    geometry.json        — 几何诊断（仅 T12 写）
    summary.md           — 人类可读的 Markdown 摘要
    stdout.log           — 捕获的标准输出/错误（§63）
    task_report.md       — §73 Task Report 12 字段

§64 config.yaml 字段：
    teacher_model, teacher_commit, student_model, student_commit,
    tokenizer, dtype, attention_implementation, rope_config, cache_layout,
    context_length, dataset, split, seed, mapper, rank, source_top_k,
    alpha_max, loss_weights, hardware, cache_residency

§65 metrics.json 字段：
    student_score, teacher_score, handoff_score, retention, chg, tgrr,
    pcr, psr_a, next_token_kl, jcr, attention_output_cosine,
    linear_cka, principal_angle, effective_rank
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 写出 JSON（ensure_ascii=False 保留中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    """以 UTF-8 写出纯文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def ensure_run_dir(base: Path, run_id: str) -> Path:
    """创建 `<base>/<run_id>/` 目录并返回其路径。"""
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir