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

run_id 与目录结构：
    run_id 形如 "qwen3-4b-to-1.7b-20260808-195506"（实验名 + 时间戳），
    时间戳格式 %Y%m%d-%H%M%S（与 apcs.io.load_config 的 ${run.timestamp} 一致）。
    同一 run_id 下的多个 <task>/ 子目录共享同一份配置与实验上下文
    （T11 / T13 必须按 run_id 前缀共享 T05/T09/T10 数据，§71）。
    UUID 不在此模块生成——可复现性优先用"实验名+时间戳"式 run_id，
    需要全局唯一标识时由调用方在 run_id 上追加短 UUID。

res 产物结构约定：
    所有产物均为"UTF-8 文本文件"，其中 JSON 用 ensure_ascii=False（保留中文），
    indent=2 缩进便于 diff 与人工审查；
    写前自动 mkdir parents（目录不存在则创建），幂等可重复写入。

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
    """以 UTF-8 写出 JSON（ensure_ascii=False 保留中文）。

    输入：path=目标文件路径，payload=任意可 JSON 序列化的对象。
    输出：无（副作用：创建父目录并写文件）。
    编码约定：indent=2 + ensure_ascii=False → 产物可读、可 diff、保留中文键值。
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # 父目录不存在则递归创建
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    """以 UTF-8 写出纯文本。

    用于 stdout.log / summary.md / task_report.md 等非 JSON 产物（§63/§73）。
    输入：path=目标文件路径，payload=待写字符串。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def ensure_run_dir(base: Path, run_id: str) -> Path:
    """创建 `<base>/<run_id>/` 目录并返回其路径。

    输入：base=reports/runs 根目录，run_id=实验名-时间戳（如 qwen3-4b-to-1.7b-20260808-195506）。
    输出：base/run_id 目录 Path（幂等：已存在则直接复用，不重建不清空）。
    后续 <task>/ 子目录由各 task 在 run_dir 下再建。
    """
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)  # 幂等创建：重复调用不会清空已有产物
    return run_dir