"""架构审查修复的回归测试（P0-1 / P0-2 / P0-3 / P1-2 / P1-4）。

═══════════════════════════════════════════════════════════════════════════════
本文件锁死 5 个已修复的架构缺陷，防止回归：

    P0-1  run_id 每次 CLI 调用都变 → 同一实验的 task 散落到多个 run 目录，
          T11/T13 依赖的"共享 run_id"前提被破坏。
    P0-2  orchestrator 的 Gate 阻断 / compliance 包装只有 tests 调用，
          CLI 完全不校验 → §72"Gate FAIL 不能跳过"没有执行点。
    P0-3  compliance 每条 check 都 `runtime.get(sig, False)`，
          信号缺失即判合规 → 输出虚假的 "violations==0 / passed=True"。
    P1-2  cli.py 用了 typing.Any 但从未 import（被 PEP 563 掩盖）。
    P1-4  T01/T10 为纯仿真却无任何 offline 标注，
          T01 更是 Gate 0 恒 PASS —— 最易被误读为真实证据。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apcs.cli import main as cli_main  # noqa: E402
from apcs.compliance import compliance_report, signal_coverage  # noqa: E402
from apcs.io.runs import resolve_run_id  # noqa: E402


def _setup_cfg(tmp_path: Path) -> Path:
    """写一份最小可用 cfg：run_id 含 ${run.timestamp}（复现 P0-1 的触发条件）。"""
    cfg = {
        "experiment": {
            "name": "regr-pair",
            "run_id": "regr-pair-${run.timestamp}",
        },
        "teacher": {"model_id": "T/teacher", "num_layers": 36,
                    "num_kv_heads": 8, "head_dim": 128, "num_attention_heads": 32},
        "student": {"model_id": "S/student", "num_layers": 28, "freeze": True,
                    "num_kv_heads": 8, "head_dim": 128, "num_attention_heads": 16},
        "output": {"base_dir": str(tmp_path / "runs")},
        "seeds": [0],
        "context_lengths": [128],
        "context_lengths_extended": [128],
        "mapper": {"type": "ridge", "rank": 8},
        "statistics": {"ci": 0.95},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


# ───────────────────────── P0-1: run_id 粘性 ─────────────────────────


def test_p0_1_run_id_is_sticky_across_invocations(tmp_path):
    """同一实验连续调用 CLI，必须复用同一个 run_id（不因时间戳漂移而分裂）。"""
    cfg_path = _setup_cfg(tmp_path)
    base = tmp_path / "runs"

    assert cli_main(["t00", "--config", str(cfg_path)]) == 0
    assert cli_main(["t01", "--config", str(cfg_path)]) == 0

    # 关键断言：base 下只能有 **一个** run 目录，且同时含 t00 与 t01
    run_dirs = [d for d in base.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, f"期望共享 1 个 run 目录，实际 {[d.name for d in run_dirs]}"
    assert (run_dirs[0] / "t00").is_dir()
    assert (run_dirs[0] / "t01").is_dir()


def test_p0_1_resolve_run_id_semantics(tmp_path):
    """resolve_run_id 的四条优先级：explicit > new_run > 指针复用 > cfg 初始化。"""
    base = tmp_path / "runs"

    # 首次：指针不存在 → 用 cfg_run_id 并建立指针
    rid1, created1 = resolve_run_id(base, "exp", "exp-111")
    assert (rid1, created1) == ("exp-111", True)

    # 再次（cfg_run_id 已漂移到 exp-222，模拟时间戳变化）→ 必须复用 exp-111
    rid2, created2 = resolve_run_id(base, "exp", "exp-222")
    assert (rid2, created2) == ("exp-111", False), "时间戳漂移不应改变当前实验的 run_id"

    # --new-run：强制开新实验
    rid3, created3 = resolve_run_id(base, "exp", "exp-333", new_run=True)
    assert (rid3, created3) == ("exp-333", True)

    # --run-id：显式覆盖，并把指针对齐
    rid4, _ = resolve_run_id(base, "exp", "exp-444", explicit="exp-custom")
    assert rid4 == "exp-custom"
    rid5, created5 = resolve_run_id(base, "exp", "exp-555")
    assert (rid5, created5) == ("exp-custom", False)

    # 不同实验名互不干扰
    rid6, created6 = resolve_run_id(base, "other-exp", "other-999")
    assert (rid6, created6) == ("other-999", True)


# ─────────────────── P0-2: §72 Gate 准入在 CLI 生效 ───────────────────


def test_p0_2_cli_blocks_task_with_missing_dependencies(tmp_path):
    """t05 依赖 t04；未跑 t04 时 CLI 必须阻断并返回退出码 2。"""
    cfg_path = _setup_cfg(tmp_path)
    cli_main(["t00", "--config", str(cfg_path)])

    rc = cli_main(["t05", "--config", str(cfg_path)])
    assert rc == 2, "缺少前置 t04 时应以退出码 2 阻断（2=准入阻断，区别于 1=Gate 未过）"
    # 被阻断的 task 不应产生任何产物目录
    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    assert not (run_dir / "t05").exists(), "被阻断的 task 不应写出产物"


def test_p0_2_force_bypasses_admission(tmp_path):
    """--force 显式放行（调试用途），此时 task 应正常执行。"""
    cfg_path = _setup_cfg(tmp_path)
    cli_main(["t00", "--config", str(cfg_path)])

    rc = cli_main(["t02", "--config", str(cfg_path), "--force"])
    assert rc == 0
    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    assert (run_dir / "t02" / "metrics.json").exists()


def test_p0_2_cli_emits_compliance_json(tmp_path):
    """每个 task 都应经 run_with_compliance 包装 → 落 compliance.json。

    修复前：全仓 compliance.json 数量为 0，而 README 声称每个 task 都会落。
    """
    cfg_path = _setup_cfg(tmp_path)
    cli_main(["t00", "--config", str(cfg_path)])

    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    cj = run_dir / "t00" / "compliance.json"
    assert cj.exists(), "CLI 未产出 compliance.json（run_with_compliance 未接线）"
    payload = json.loads(cj.read_text(encoding="utf-8"))
    assert "signals" in payload and "report" in payload


# ─────────────────── P0-3: compliance 不再空转通过 ───────────────────


def test_p0_3_missing_signals_are_unknown_not_pass():
    """信号全缺失时：passed 可为 True，但 fully_verified 必须为 False。"""
    rep = compliance_report([], runtime={})
    assert rep["n_violations"] == 0
    assert rep["passed"] is True, "无违规时 passed=True（语义：已采集信号中没有违规）"
    # ★ 核心：不能据此声称合规 —— 8 条全是检测盲区
    assert rep["n_unknown"] == 8
    assert rep["fully_verified"] is False, "零埋点却报 fully_verified=True 即虚假保证"
    assert set(rep["coverage"].values()) == {"UNKNOWN"}


def test_p0_3_tracked_signal_becomes_checked():
    """显式采集到的信号应标记为 CHECKED（哪怕值是 False=未违规）。"""
    cov = signal_coverage({"student_input_has_context": False})
    assert cov["§52.1"] == "CHECKED"
    assert cov["§52.2"] == "UNKNOWN"

    rep = compliance_report([], runtime={"student_input_has_context": False})
    assert rep["n_unknown"] == 7
    assert rep["fully_verified"] is False


def test_p0_3_no_spurious_525_violation(tmp_path):
    """§52.5 不得在信号缺失时误报。

    修复前：infer_signals_from_cfg 用一个**不存在的 cfg 键**推断
    reports_teacher_prefill → 恒 False，叠加 check_5 的 default=True 语义，
    导致每个 task 都被误报 §52.5 违规。
    """
    cfg_path = _setup_cfg(tmp_path)
    cli_main(["t00", "--config", str(cfg_path)])

    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    payload = json.loads((run_dir / "t00" / "compliance.json").read_text(encoding="utf-8"))
    rules = [v["rule_id"] for v in payload["report"]["violations"]]
    assert "§52.5" not in rules, f"§52.5 误报（信号缺失不应判违规）：{rules}"


# ───────────────────── P1-2: cli.py 的 Any import ─────────────────────


def test_p1_2_cli_type_hints_resolve():
    """cli.py 曾使用 typing.Any 却未 import（被 PEP 563 字符串注解掩盖）。

    get_type_hints 会真正求值注解 —— 缺 import 时抛 NameError。
    """
    import typing

    from apcs import cli as cli_mod

    hints = typing.get_type_hints(cli_mod._write_task_report)
    assert hints, "注解应可正常求值（缺少 `from typing import Any` 会 NameError）"


# ───────────────── P1-4: 仿真任务的诚实性标注 ─────────────────


def test_p1_4_t01_is_labeled_offline(tmp_path):
    """T01 Gate 0 恒 PASS，必须带 offline_demo 标注 + summary 警告。"""
    cfg_path = _setup_cfg(tmp_path)
    assert cli_main(["t01", "--config", str(cfg_path), "--force"]) == 0

    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    m = json.loads((run_dir / "t01" / "metrics.json").read_text(encoding="utf-8"))
    assert m["offline_demo"] is True
    assert "note" in m
    summary = (run_dir / "t01" / "summary.md").read_text(encoding="utf-8")
    assert "offline demo" in summary.lower()


def test_p1_4_t10_is_labeled_offline_and_emits_metrics(tmp_path):
    """T10 计时为公式模拟，必须标注；且 metrics.json 不得为空（fig3/fig4 依赖）。"""
    cfg_path = _setup_cfg(tmp_path)
    assert cli_main(["t10", "--config", str(cfg_path), "--force"]) == 0

    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    m = json.loads((run_dir / "t10" / "metrics.json").read_text(encoding="utf-8"))
    assert m["offline_demo"] is True
    # 修复前 t10 返回 "metrics": {} → metrics.json 为空 → fig3/fig4 取不到 per_context
    assert m.get("per_context"), "T10 metrics.json 缺少 per_context（fig3/fig4 会断流）"


def test_p1_4_task_report_marks_simulated(tmp_path):
    """§75 全局守卫：仿真结果必须在 task_report.md 的 STATUS 行标 [SIMULATED]。"""
    cfg_path = _setup_cfg(tmp_path)
    cli_main(["t01", "--config", str(cfg_path), "--force"])

    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    report = (run_dir / "t01" / "task_report.md").read_text(encoding="utf-8")
    status_line = [l for l in report.splitlines() if l.startswith("- STATUS:")][0]
    assert "[SIMULATED]" in status_line, f"仿真任务未标记：{status_line}"
    assert "不可作为真实实验证据" in report


def test_p1_4_real_math_task_not_marked_simulated(tmp_path):
    """反向断言：T02（真实 RoPE 数学）不应被误标为 SIMULATED。"""
    cfg_path = _setup_cfg(tmp_path)
    cli_main(["t02", "--config", str(cfg_path), "--force"])

    base = tmp_path / "runs"
    run_dir = [d for d in base.iterdir() if d.is_dir()][0]
    report = (run_dir / "t02" / "task_report.md").read_text(encoding="utf-8")
    status_line = [l for l in report.splitlines() if l.startswith("- STATUS:")][0]
    assert "[SIMULATED]" not in status_line, "真实数学任务不应被标记为仿真"
