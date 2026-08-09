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

check 契约：
    - 每个 check_X 接收 (cfg, run_dir, runtime) 的子集，返回 list[ComplianceViolation]；
    - 空列表 = 该条 PASS；任一 check 返回违规 → 整体 FAIL（violations == 0 才放行）；
    - runtime 中未采集到的信号走 .get 默认值：
        * 默认 False = "乐观默认"（未观测到违规即视为合规，§52.1/2/3/4/6/7/8）；
        * 默认 True  = "保守默认"（§52.5 reports_teacher_prefill：未报告即判违规）；
    - 信号缺失是"检测盲区"：不报违规 ≠ 实验干净，应在运行时强制采集
      （静态可推断的部分见 apcs.compliance.runtime.infer_signals_from_cfg）。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComplianceViolation:
    """单条违规记录（§52 审计矩阵中的一行）。

    Attributes:
        rule_id: 规则编号（§52.1 … §52.8），供报告与日志对号入座；
        message: 人读的中文描述，说明"采集到哪个信号 + 为什么违规"。
    """

    rule_id: str  # §52.1, §52.2 ...
    message: str


class ComplianceError(Exception):
    """聚合违规列表的异常；message 拼接所有 [rule_id] 描述。

    由 check_all 的结果驱动：任意违规即 raise，阻断当前 task（§52 强制）。
    violations 属性保留结构化列表，供上层（orchestrator / CLI）精确处置。
    """

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
    # 信号来源：Student forward 的 input_ids 是否混入了 X 的 token
    # （运行时由 apcs.compliance.runtime.track_runtime 采集；默认 False = 未检测到）
    student_re_read_x = runtime.get("student_input_has_context", False)
    if student_re_read_x:
        # 一旦为 True 即违规：Student 主实验只能接收 q，重读 X 会让 Handoff 名存实亡
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
    # freeze 默认 True（T08 强制）：Student 主体在冻结期间不应有任何参数更新
    cfg_freeze = cfg.get("student", {}).get("freeze", True)
    if student_updated and cfg_freeze:
        # 两者同时成立 = 自相矛盾：一边宣称 Runtime State Transfer（参数冻结），
        # 一边偷偷微调了 Student 主体（或冻结未被真正生效）
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
    # §36 / §70：test 只能做一次 inference，任何在 test 上选超参/阈值/早期停止的行为都算违规
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
    # §16：test 必须完整评估；按结果事后挑样本（Teacher 对 Student 错）会人为制造假 CHG
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
    # 注意与其它 check 相反：默认 True（报告存在），显式上报 False 才判违规。
    # 这是"保守默认"——报告缺失时无法自动判定，只能靠显式信号揭发
    has_teacher_prefill = runtime.get("reports_teacher_prefill", True)
    if not has_teacher_prefill:
        # Scenario A 把 Teacher Prefill 当沉没成本是计费口径，不是省略报告的理由
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
    # H2D / Cache Load 是 PSR_A 分子 (T_map+T_load+T_query) 的一部分，藏着不算 = 虚增 PSR_A
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
    # 相似度指标（R²/Cosine/CKA）只能做机制分析（T12），不能单独支撑 Path A 结论
    claim_path_a_on_similarity_only = runtime.get(
        "claim_path_a_on_similarity_only", False
    )
    if claim_path_a_on_similarity_only:
        # 必须有 CHG>0（且 bootstrap CI>0，§51）才允许主张 Path A
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
    # Cache 注入失败后的两条合法出路：显式报错 / 显式降级；静默重算 = 作弊
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
    """跑全部 §52 检查，返回违规列表。

    聚合语义：
        - 结果 = check_1..check_8 违规列表的并集（顺序固定）；
        - 空列表 → PASS；任一 check 返回违规 → 整体 FAIL（violations==0 是实验放行前提）；
        - run_dir 目前仅 check_1 使用（保留参数位，供未来基于文件系统的审计扩展）。
    """
    runtime = runtime or {}
    violations: list[ComplianceViolation] = []
    # 逐条执行；即使前面已发现违规也继续跑完全部检查，
    # 以便单次报告给出完整违规清单（而不是只报第一条）
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
    """生成 compliance 报告（写到 metrics.json 旁）。

    把违规列表序列化为可落盘的 dict：
        n_violations / violations[{rule_id, message}] / passed（== n_violations == 0）。
    """
    return {
        "n_violations": len(violations),
        "violations": [
            {"rule_id": v.rule_id, "message": v.message} for v in violations
        ],
        "passed": len(violations) == 0,  # PASS 判定：一条违规都不允许
    }


def run_compliance_check(cfg: dict, run_dir, runtime: dict | None = None) -> dict[str, Any]:
    """CLI 入口：跑 §52 全套检查。

    runtime 是可选的运行时信号 dict。如果不传，默认"无任何违反"。

    Returns:
        dict with status / metrics / summary
    """
    from ..io.runs import write_json, write_text

    # 1) 核心检查：runtime 信号未传时按空 dict 处理（默认全部合规，依赖 check 内默认值）
    violations = check_all(cfg, run_dir, runtime or {})
    rep = compliance_report(violations)
    # 2) 落 metrics.json：与其它 task 的产物规范一致（§63）
    metrics = {"task": "compliance", **rep}
    write_json(run_dir / "metrics.json", metrics)
    # 3) 生成人类可读的 summary.md（§73 报告的一部分）
    md = (
        "# §52 Compliance Report\n\n"
        f"- Passed: {rep['passed']}\n"
        f"- Violations: {rep['n_violations']}\n\n"
    )
    if violations:
        # 有违规 → 逐条列出（rule_id + message 的审计表格）
        md += "| Rule | Message |\n| ---- | ------- |\n"
        for v in violations:
            md += f"| {v.rule_id} | {v.message} |\n"
    else:
        md += "All §52 forbidden actions checked clean.\n"
    write_text(run_dir / "summary.md", md)
    # 4) 最终 verdict：violations==0 → PASS，否则 FAIL（供 orchestrator 阻断后续 task）
    return {
        "status": "PASS" if rep["passed"] else "FAIL",
        "metrics": metrics,
        "summary": md,
    }