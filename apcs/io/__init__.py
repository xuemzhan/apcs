"""配置加载工具（design.md §64 config.yaml 字段）。

═══════════════════════════════════════════════════════════════════════════════
支持 ${...} 占位符与 ${run.timestamp} 时间戳注入。

§64 config.yaml 必须记录的字段：
    teacher_model / teacher_commit / student_model / student_commit /
    tokenizer / dtype / attention_implementation / rope_config /
    cache_layout / context_length / dataset / split / seed / mapper /
    rank / source_top_k / alpha_max / loss_weights / hardware /
    cache_residency

本 loader 不做严格 schema 校验（避免过度耦合），只保证 YAML 能解析、
占位符能展开。schema 校验由 config schema（可选）单独完成。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any

import yaml

# 匹配 ${...} 占位符；支持嵌套（如 ${experiment.name}）
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_string(value: str, ctx: dict[str, Any]) -> str:
    """把字符串里的 ${...} 替换为 ctx 中的值；未匹配则原样保留。"""

    def replace(m: re.Match) -> str:
        key = m.group(1).strip()
        # §64 隐式：${run.timestamp} 是实验运行时间戳
        if key == "run.timestamp":
            return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if key in ctx:
            return str(ctx[key])
        return m.group(0)

    return _PLACEHOLDER_RE.sub(replace, value)


def _deep_resolve(obj: Any, ctx: dict[str, Any]) -> Any:
    """递归解析所有占位符。"""
    if isinstance(obj, str):
        return _resolve_string(obj, ctx)
    if isinstance(obj, dict):
        return {k: _deep_resolve(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_resolve(v, ctx) for v in obj]
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置并解析所有 ${...} 占位符。

    占位符上下文：
        - experiment.name    （来自 config 自身）
        - run.timestamp      （运行时注入）

    返回的 dict 与 YAML 完全对应；占位符被替换为具体字符串。
    """
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    ctx: dict[str, Any] = {}
    if isinstance(raw, dict):
        exp = raw.get("experiment")
        if isinstance(exp, dict) and "name" in exp:
            ctx["experiment"] = exp
            ctx["experiment.name"] = exp["name"]
    return _deep_resolve(raw, ctx)