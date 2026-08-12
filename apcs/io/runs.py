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


# ---------------------------------------------------------------------------
# run_id 粘性解析（§71 修复：跨 task 共享同一个 run_id）
#
# ◆ 修复背景（架构审查 P0-1）：
#   configs/*.yaml 里 `run_id: ${experiment.name}-${run.timestamp}`，而
#   `${run.timestamp}` 在 **load_config 时**展开为“当前时刻”。CLI 每个
#   task 是一个独立进程 → 每次调用都得到**不同的 run_id** → README §7.2
#   那 14 行命令会产出最多 14 个互不相干的 run 目录，
#   T11/T13 依赖的“同一 run_id 下共享 T05/T09/T10”前提被破坏。
#
# ◆ 修复策略（粘性 run_id）：
#   把“本实验当前使用的 run_id”持久化到 `<base>/<experiment_name>.current`
#   指针文件。首个 task 落盘，后续 task 读取复用；`--run-id` 显式覆盖，
#   `--new-run` 强制开新实验。这样时间戳只在实验**开始时**取一次。
# ---------------------------------------------------------------------------

_POINTER_SUFFIX = ".current"


def _pointer_path(base: Path, experiment_name: str) -> Path:
    """返回记录“当前 run_id”的指针文件路径 `<base>/<experiment_name>.current`。

    以 experiment.name（而非 run_id）为键：同一个实验（如 qwen3-4b-to-1.7b）
    在 base_dir 下只有一个“当前 run”，不同实验（第二 Pair）互不干扰。
    """
    # experiment_name 可能含路径分隔符等非法字符，做一次保守清洗
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in experiment_name)
    return base / f"{safe}{_POINTER_SUFFIX}"


def resolve_run_id(
    base: Path,
    experiment_name: str,
    cfg_run_id: str,
    explicit: str | None = None,
    new_run: bool = False,
) -> tuple[str, bool]:
    """决定本次 task 使用的 run_id，并维护 `<base>/<name>.current` 指针。

    优先级（从高到低）：
        1. explicit（CLI `--run-id`）  — 完全由调用方指定，同时刷新指针；
        2. new_run（CLI `--new-run`）  — 强制用 cfg_run_id 开一个新实验；
        3. 指针文件中已记录的 run_id   — 复用当前实验（**默认路径**）；
        4. cfg_run_id                  — 指针不存在时（实验的第一个 task）。

    返回：(run_id, created_new)
        created_new=True 表示本次新建/切换了实验（指针被写入新值），
        CLI 借此打印提示，让使用者清楚“新实验开始了”。

    注意：本函数只负责“选 id + 写指针”，不创建 run 目录（由 ensure_run_dir 做）。
    """
    base.mkdir(parents=True, exist_ok=True)
    ptr = _pointer_path(base, experiment_name)
    # 先读旧值：后续判断“是否切换了实验”必须基于**写入前**的状态
    previous = ptr.read_text(encoding="utf-8").strip() if ptr.exists() else ""

    if explicit:
        # 显式指定：以调用方为准，并把指针对齐到该 run（便于后续 task 免参数复用）
        _write_pointer(ptr, explicit)
        return explicit, explicit != previous

    if new_run:
        # 强制开新实验：用 cfg 里刚展开的时间戳 run_id
        _write_pointer(ptr, cfg_run_id)
        return cfg_run_id, True

    if previous:
        # ★ 默认路径：复用当前实验的 run_id（时间戳不再每次刷新）
        return previous, False

    # 指针不存在 → 这是该实验的第一个 task，用 cfg_run_id 建立指针
    _write_pointer(ptr, cfg_run_id)
    return cfg_run_id, True


def _write_pointer(ptr: Path, run_id: str) -> None:
    """把 run_id 写入指针文件（UTF-8，无换行噪声）。"""
    ptr.parent.mkdir(parents=True, exist_ok=True)
    ptr.write_text(run_id, encoding="utf-8")