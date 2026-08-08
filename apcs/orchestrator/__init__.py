"""§72 Agent Orchestrator + §73 Task Report。

═══════════════════════════════════════════════════════════════════════════════
§72 Agent 强制规则：
    1. 一次只允许执行一个 Task
    2. 每个 Task 有 Acceptance Test
    3. 每个 Task 生成结果报告
    4. Gate FAIL 后不能跳过
    5. Test 不得用于 Auto-Hyperparameter Search
    6. Agent 不得为了产生正 CHG 自动增加无上限实验
    7. 所有 Negative Result 必须保留

§73 Task Report 12 字段：
    TASK_ID, STATUS, OBJECTIVE, MODEL_PAIR, DATASET, CONFIG,
    IMPLEMENTATION, OUTPUT_FILES, KEY_METRICS, STATISTICAL_CHECK,
    BEHAVIOR_CHECK, GEOMETRY_CHECK, SYSTEM_COST, ACCEPTANCE_CRITERIA,
    RESULT, FAILURE_ANALYSIS, NEXT_ALLOWED_TASK
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

# §71 强制顺序
TASK_ORDER = [
    "t00",
    "t01",
    "t02",
    "t03",
    "t04",
    "t05",
    "t06",
    "t07",
    "t08",
    "t09",
    "t10",  # 与 t09 并行
    "t12",  # 与 t09 并行
    "t11",
    "t13",
]


def dependencies(task: str) -> set[str]:
    """§71 任务的依赖关系。"""
    deps = {
        "t00": set(),
        "t01": {"t00"},
        "t02": {"t00"},
        "t03": {"t00"},
        "t04": {"t01", "t02", "t03"},
        "t05": {"t04"},
        "t06": {"t04"},
        "t07": {"t00"},
        "t08": {"t04", "t07"},
        "t09": {"t05", "t06", "t07", "t08"},
        "t10": {"t05", "t08"},
        "t11": {"t05", "t09", "t10"},
        "t12": {"t04"},
        "t13": {"t09"},
        "ablation": {"t04", "t08"},
        "multiturn": {"t09"},
        "compliance": set(),  # compliance 不依赖其他 task
    }
    return deps.get(task, set())


def run_with_compliance(cfg: dict, runner, run_dir, runtime: dict | None = None):
    """§72 orchestrator 包装：先 reset + 自动信号，再跑 task，最后检查 §52。

    用法：
        from apcs.orchestrator import run_with_compliance
        from apcs.mapper.runner import run_replacement

        result = run_with_compliance(cfg, run_replacement, run_dir)

    runner 签名：runner(cfg, run_dir) → dict
    """
    from ..compliance import check_all, compliance_report
    from ..compliance.runtime import collect, reset, infer_signals_from_cfg
    from ..io.runs import write_json

    reset()
    # 从 cfg 静态推断
    for k, v in infer_signals_from_cfg(cfg).items():
        from ..compliance.runtime import track

        track(k, v)

    # 跑任务
    result = runner(cfg, run_dir)

    # 收集运行时信号 + 静态推断，跑 §52 检查
    signals = collect()
    violations = check_all(cfg, run_dir, signals)
    rep = compliance_report(violations)
    write_json(
        run_dir / "compliance.json",
        {"signals": signals, "report": rep},
    )
    result.setdefault("compliance", rep)
    return result


def _transitive_deps(task: str) -> set[str]:
    """task 的全部传递依赖（依赖的依赖…，含间接依赖）。

    §72 Gate FAIL 不得跳过：`dependencies()` 只给直接依赖，
    next_allowed 必须检查闭包 —— 若 t04 失败，则所有（间接）依赖 t04 的
    任务（如 t09 经 t05/t06/t08）都不得放行。
    """
    seen: set[str] = set()
    stack = list(dependencies(task))
    while stack:
        d = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        stack.extend(dependencies(d))
    return seen


def next_allowed(task: str, last_status: dict[str, str]) -> str | None:
    """根据前序状态决定下一步可执行任务。

    §72 规则：
    - 任务 t 可运行的条件：所有依赖都已 PASS/OK，且依赖里没有 FAIL
    - 若有 FAIL 阻塞，按 TASK_ORDER 跳过它继续找下一个
    """
    failed = {t for t, s in last_status.items() if s not in {"PASS", "OK"}}
    for t in TASK_ORDER:
        if t in last_status:
            continue
        deps = dependencies(t)
        # 所有依赖必须都已 PASS/OK
        if any(d not in last_status or last_status[d] not in {"PASS", "OK"} for d in deps):
            continue
        # 也不能与失败任务有传递依赖（闭包内任一 FAIL 即阻断，
        # ◆ bug：旧实现只查 `deps & failed` 直接依赖，t04 FAIL 而
        #   t05/t06/t07/t08 全 PASS 时会误放行 t09）
        if _transitive_deps(t) & failed:
            continue
        return t
    return None


def write_task_report(
    run_dir: Path,
    task_id: str,
    status: str,
    objective: str,
    model_pair: tuple[str, str],
    dataset: str,
    config_summary: dict,
    implementation: str,
    output_files: list[str],
    key_metrics: dict,
    statistical_check: dict | None = None,
    behavior_check: dict | None = None,
    geometry_check: dict | None = None,
    system_cost: dict | None = None,
    acceptance_criteria: dict | None = None,
    failure_analysis: str | None = None,
) -> None:
    """§73 Task Report 写入。"""
    next_t = None
    if status in {"PASS", "OK"}:
        idx = TASK_ORDER.index(task_id) if task_id in TASK_ORDER else -1
        if 0 <= idx < len(TASK_ORDER) - 1:
            next_t = TASK_ORDER[idx + 1]

    lines = [
        "# Task Report",
        "",
        f"- TASK_ID: `{task_id}`",
        f"- STATUS: **{status}**",
        f"- Generated at: {_dt.datetime.now().isoformat()}",
        "",
        f"## OBJECTIVE",
        objective,
        "",
        f"## MODEL_PAIR",
        f"- Teacher: `{model_pair[0]}`",
        f"- Student: `{model_pair[1]}`",
        "",
        f"## DATASET",
        dataset,
        "",
        f"## CONFIG",
        "```json",
        json.dumps(config_summary, indent=2, ensure_ascii=False),
        "```",
        "",
        f"## IMPLEMENTATION",
        implementation,
        "",
        f"## OUTPUT_FILES",
    ]
    for f in output_files:
        lines.append(f"- `{f}`")
    lines += [
        "",
        f"## KEY_METRICS",
        "```json",
        json.dumps(key_metrics, indent=2, ensure_ascii=False),
        "```",
        "",
        f"## STATISTICAL_CHECK",
        json.dumps(statistical_check or {}, indent=2, ensure_ascii=False),
        "",
        f"## BEHAVIOR_CHECK",
        json.dumps(behavior_check or {}, indent=2, ensure_ascii=False),
        "",
        f"## GEOMETRY_CHECK",
        json.dumps(geometry_check or {}, indent=2, ensure_ascii=False),
        "",
        f"## SYSTEM_COST",
        json.dumps(system_cost or {}, indent=2, ensure_ascii=False),
        "",
        f"## ACCEPTANCE_CRITERIA",
        json.dumps(acceptance_criteria or {}, indent=2, ensure_ascii=False),
        "",
        f"## FAILURE_ANALYSIS",
        failure_analysis or "(无)",
        "",
        f"## NEXT_ALLOWED_TASK",
        next_t or "(全部完成)",
    ]
    (run_dir / "task_report.md").write_text("\n".join(lines), encoding="utf-8")