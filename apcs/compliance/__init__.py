"""§52 八条禁止的自动检查器。

═══════════════════════════════════════════════════════════════════════════════
design.md §52 列出 8 条禁止项，必须有自动化检测来确保实验不被静默违反。
每条规则都以一个 checker 函数实现，返回 list[str]（违反项描述）。

调用：
    from apcs.compliance import check_all
    violations = check_all(cfg, run_dir, runtime_state)
    if violations:
        raise ComplianceError(violations)

设计为独立模块，可嵌入 CLI / orchestrator / 单测。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComplianceViolation:
    rule_id: str  # §52.1, §52.2 ...
    message: str


class ComplianceError(Exception):
    def __init__(self, violations: list[ComplianceViolation]):
        self.violations = violations
        super().__init__(
            "Compliance violation: " +
            "; ".join(f"[{v.rule_id}] {v.message}" for v in violations)
        )


# ---- 各条规则实现 ----


def check_1_no_student_re_read_x(run_dir: Path, runtime: dict) -> list[ComplianceViolation]:
    """§52.1 Student 在主实验中重新读取 X。

    检测：Student forward 期间如果接收了除 q 之外的额外 input_ids，
    即视为重新读取 X。
    """
    out: list[ComplianceViolation] = []
    student_re_read_x = runtime.get("student_input_has_context", False)
    if student_re_read_x:
        out.append(
            ComplianceViolation(
                "§52.1",
                "Student 在主实验 forward 时接收了 X context（应只接收 q）",
            )
        )
    return out


def check_2_student_frozen_in_main_exp(cfg: dict, runtime: dict) -> list[ComplianceViolation]:
    """§52.2 微调 Student 主体后仍称 Runtime State Transfer。

    检测：主实验阶段 Student 主体不应有可训练参数被更新。
    """
    out: list[ComplianceViolation] = []
    student_updated = runtime.get("student_params_updated", False)
    cfg_freeze = cfg.get("student", {}).get("freeze", True)
    if student_updated and cfg_freeze:
        out.append(
            ComplianceViolation(
                "§52.2",
                "cfg.student.freeze=true 但 Student 参数被更新（违反 Runtime State Transfer 前提）",
            )
        )
    return out


def check_3_no_test_tuning(cfg: dict, runtime: dict) -> list[ComplianceViolation]:
    """§52.3 Test 调参。

    检测：test 阶段不应做超参搜索。
    """
    out: list[ComplianceViolation] = []
    test_hp_search = runtime.get("test_hp_search", False)
    if test_hp_search:
        out.append(
            ComplianceViolation(
                "§52.3",
                "Test 阶段做了超参搜索（仅 validation 上允许）",
            )
        )
    return out


def check_4_no_teacher_win_filtering(runtime: dict) -> list[ComplianceViolation]:
    """§52.4 只挑 Teacher-win Test Sample。

    检测：test 数据集应完整使用，不应事后筛选 'Teacher 对 / Student 错'。
    """
    out: list[ComplianceViolation] = []
    filtered = runtime.get("test_filtered_to_teacher_win", False)
    if filtered:
        out.append(
            ComplianceViolation(
                "§52.4",
                "Test 数据集被筛选为仅 Teacher-win 样本（违反 §16 / §52.4）",
            )
        )
    return out


def check_5_no_hidden_teacher_prefill_cost(runtime: dict) -> list[ComplianceViolation]:
    """§52.5 隐藏 Teacher Prefill 成本。

    检测：PSR_A 报告中必须包含 teacher_prefill 耗时，
    即使 Scenario A 把它视为沉没成本。
    """
    out: list[ComplianceViolation] = []
    has_teacher_prefill = runtime.get("reports_teacher_prefill", True)
    if not has_teacher_prefill:
        out.append(
            ComplianceViolation(
                "§52.5",
                "PSR_A 报告未包含 teacher_prefill 耗时（即使 Scenario A 是沉没成本也应报告）",
            )
        )
    return out


def check_6_no_hidden_h2d_cost(runtime: dict) -> list[ComplianceViolation]:
    """§52.6 隐藏 H2D / Cache Load。

    检测：所有 cache load 时间必须出现在 PSR_A 公式中。
    """
    out: list[ComplianceViolation] = []
    hidden_load = runtime.get("hides_h2d_load", False)
    if hidden_load:
        out.append(
            ComplianceViolation(
                "§52.6",
                "H2D / Cache Load 成本未计入 PSR_A",
            )
        )
    return out


def check_7_no_replacing_chg_with_similarity(runtime: dict) -> list[ComplianceViolation]:
    """§52.7 用 R² / Cosine / CKA 代替 CHG。

    检测：主结论必须基于 CHG，不能仅用相似度指标支撑。
    """
    out: list[ComplianceViolation] = []
    claim_path_a_on_similarity_only = runtime.get(
        "claim_path_a_on_similarity_only", False
    )
    if claim_path_a_on_similarity_only:
        out.append(
            ComplianceViolation(
                "§52.7",
                "在没有 CHG>0 的情况下用 R² / Cosine / CKA 主张 Path A",
            )
        )
    return out


def check_8_no_silent_reprefill(runtime: dict) -> list[ComplianceViolation]:
    """§52.8 Cache 注入失败后 Silent Re-prefill。

    检测：cache 注入失败时必须报错或显式降级，不能静默重新 prefill。
    """
    out: list[ComplianceViolation] = []
    silent_re_prefill = runtime.get("silent_re_prefill_on_failure", False)
    if silent_re_prefill:
        out.append(
            ComplianceViolation(
                "§52.8",
                "Cache 注入失败后静默重新 Prefill X（违反 §52.8）",
            )
        )
    return out


# ---- 统一入口 ----


def check_all(cfg: dict, run_dir: Path, runtime: dict | None = None) -> list[ComplianceViolation]:
    """跑全部 §52 检查，返回违规列表。"""
    runtime = runtime or {}
    violations: list[ComplianceViolation] = []
    violations += check_1_no_student_re_read_x(run_dir, runtime)
    violations += check_2_student_frozen_in_main_exp(cfg, runtime)
    violations += check_3_no_test_tuning(cfg, runtime)
    violations += check_4_no_teacher_win_filtering(runtime)
    violations += check_5_no_hidden_teacher_prefill_cost(runtime)
    violations += check_6_no_hidden_h2d_cost(runtime)
    violations += check_7_no_replacing_chg_with_similarity(runtime)
    violations += check_8_no_silent_reprefill(runtime)
    return violations


def compliance_report(violations: list[ComplianceViolation]) -> dict[str, Any]:
    """生成 compliance 报告（写到 metrics.json 旁）。"""
    return {
        "n_violations": len(violations),
        "violations": [
            {"rule_id": v.rule_id, "message": v.message} for v in violations
        ],
        "passed": len(violations) == 0,
    }


def run_compliance_check(cfg: dict, run_dir, runtime: dict | None = None) -> dict[str, Any]:
    """CLI 入口：跑 §52 全套检查。

    runtime 是可选的运行时信号 dict。如果不传，默认"无任何违反"。

    Returns:
        dict with status / metrics / summary
    """
    from ..io.runs import write_json, write_text

    violations = check_all(cfg, run_dir, runtime or {})
    rep = compliance_report(violations)
    metrics = {"task": "compliance", **rep}
    write_json(run_dir / "metrics.json", metrics)
    md = (
        "# §52 Compliance Report\n\n"
        f"- Passed: {rep['passed']}\n"
        f"- Violations: {rep['n_violations']}\n\n"
    )
    if violations:
        md += "| Rule | Message |\n| ---- | ------- |\n"
        for v in violations:
            md += f"| {v.rule_id} | {v.message} |\n"
    else:
        md += "All §52 forbidden actions checked clean.\n"
    write_text(run_dir / "summary.md", md)
    return {
        "status": "PASS" if rep["passed"] else "FAIL",
        "metrics": metrics,
        "summary": md,
    }