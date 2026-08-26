"""跨任务的证据分级与结论门控。

表示相似度、CUDA 代理耗时和合成得分都可以用于诊断，但不能单独
支撑“任务保真”、“能力迁移”或“系统加速”结论。本模块是 T11/T13
的单一事实源，防止下游将不同证据等级混在同一个 verdict 中。
"""
from __future__ import annotations

from typing import Any


MEASURED_TASK_GRADES = {"measured", "measured_task", "measured_functional"}


def is_measured_task(metrics: dict[str, Any] | None) -> bool:
    """判断产物是否包含可用于论文结论的真实任务评分。"""
    if not isinstance(metrics, dict):
        return False
    return (
        metrics.get("offline_demo") is False
        and metrics.get("evidence_grade") in MEASURED_TASK_GRADES
    )


def conclusion_blockers(
    t05: dict[str, Any] | None,
    t09: dict[str, Any] | None,
    t10: dict[str, Any] | None,
) -> list[str]:
    """返回 Path A/B 结论尚缺失的真实证据。"""
    blockers: list[str] = []
    if not is_measured_task(t05) or not isinstance(t05.get("task_retention"), (int, float)):
        blockers.append("T05 缺少真实注入后的 task_retention")
    if not is_measured_task(t09):
        blockers.append("T09 CHG/TGRR 不是真实 held-out 任务得分")
    if not isinstance(t10, dict) or t10.get("timing_evidence") != "end_to_end_handoff":
        blockers.append("T10 缺少 HandoffPipeline 端到端计时")
    return blockers
