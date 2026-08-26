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

# §71 强制顺序：这是 Orchestrator 的"主线时钟"。
# 编排语义：
#   - t00 → t01 → … → t08 严格串行（每步都是下一步的直接依赖）；
#   - t09 之后 t10 / t12 与 t09 无相互依赖，可并行跑，故排在 t09 之后；
#   - t11 依赖 t09/t10，t13 依赖 t09，排在最后。
# next_allowed 与 write_task_report 都按此顺序推进。
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
    "t10",  # 与 t09 并行（依赖 t05/t08，不依赖 t09）
    "t12",  # 与 t09 并行（只依赖 t04）
    "t11",
    "t13",
]


def dependencies(task: str) -> set[str]:
    """§71 任务的直接依赖关系（DAG 边）。

    返回的是**直接**依赖：比如 t09 直接依赖 t05/t06/t07/t08。
    判断"能否执行"时必须用 _transitive_deps 检查闭包（间接依赖），
    因为 §72 要求任一（直接或间接）前置 Gate FAIL 都不得放行。
    compliance 是独立 subcommand，不依赖任何 task（返回空集）。
    """
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
        "inject-eval": {"t00"},  # inject-eval 依赖 t00（需 model_compatibility）
    }
    return deps.get(task, set())


def run_with_compliance(cfg: dict, runner, run_dir, runtime: dict | None = None):
    """§72 orchestrator 包装：先 reset + 自动信号，再跑 task，最后检查 §52。

    编排流程（顺序固定）：
        1. reset()               清空上一轮运行时信号，避免跨 task 污染；
        2. 静态信号              从 cfg 推断（infer_signals_from_cfg），
                                  逐条 track 写入信号库；
        3. 执行 runner           原样调用 runner(cfg, run_dir)，不侵入业务逻辑；
        4. collect()             收集 runner 运行期间登记的运行时信号；
        5. check_all + 落盘      §52 八条禁止检查，写 compliance.json，
                                  报告挂到 result["compliance"] 返回。

    组合性：本包装可叠加 —— runner 可以是任意 task 的入口
    （replacement / advantage / system …），只要满足签名
    runner(cfg, run_dir) → dict 即可，无需改造任务本身。

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
    # ---- 第 1 步：静态信号（只读 cfg，无需真实运行）----
    # 覆盖 run_with_compliance 开始前就可能存在的违规信号，
    # 例如 cfg.student.freeze=false 或 cfg 暴露的 test_hp_search 开关。
    for k, v in infer_signals_from_cfg(cfg).items():
        from ..compliance.runtime import track

        track(k, v)

    # ---- 第 2 步：执行任务本体 ----
    result = runner(cfg, run_dir)

    # ---- 第 3 步：收尾检查 + 落盘 ----
    # collect() 合并静态信号与运行时信号；check_all 按 §52 逐条判定。
    signals = collect()
    violations = check_all(cfg, run_dir, signals)
    rep = compliance_report(violations, signals)
    write_json(
        run_dir / "compliance.json",
        {"signals": signals, "report": rep},
    )
    # 报告挂回 result，上层（CLI / 其它 runner）无需感知 compliance 的存在
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
    # 迭代 DFS：出栈一个依赖 → 收入 seen → 再扩展它的直接依赖，
    # 直到栈空。seen 即闭包（含全部间接依赖），重复依赖天然去重。
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
    # last_status: {task_id: status}；非 PASS/OK（FAIL/CONDITIONAL/UNKNOWN）
    # 一律视为"失败"，进入阻断集合。
    failed = {t for t, s in last_status.items() if s not in {"PASS", "OK"}}
    # 沿 §71 主线顺序找第一个"未执行且可执行"的任务；
    # 已执行过的任务（出现在 last_status 中）直接跳过，不重复放行。
    for t in TASK_ORDER:
        if t in last_status:
            continue
        deps = dependencies(t)
        # 第一重过滤：所有直接依赖必须已存在且 PASS/OK（Gate 阻断）
        if any(d not in last_status or last_status[d] not in {"PASS", "OK"} for d in deps):
            continue
        # 第二重过滤：传递闭包内任一失败任务即阻断。
        # ◆ bug：旧实现只查 `deps & failed` 直接依赖，t04 FAIL 而
        #   t05/t06/t07/t08 全 PASS 时会误放行 t09（传递依赖漏检）
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
    """§73 Task Report 写入：按 §73 12 字段渲染 Markdown 到 task_report.md。

    NEXT_ALLOWED_TASK 字段 = TASK_ORDER 中紧邻的下一个任务（仅当本任务
    PASS/OK 时给出；FAIL 则留空 —— 见 §72 "Gate FAIL 不能跳过"）。
    """
    # 只有 PASS/OK 才推进主线：取 TASK_ORDER 中 task_id 的下一个
    next_t = None
    if status in {"PASS", "OK"}:
        idx = TASK_ORDER.index(task_id) if task_id in TASK_ORDER else -1
        # idx < len-1 说明后面还有任务；最后一个（t13）→ None → "(全部完成)"
        if 0 <= idx < len(TASK_ORDER) - 1:
            next_t = TASK_ORDER[idx + 1]

    # ---- §75 全局诚实性守卫（架构审查 P1-4）----
    # 任何 task 的 metrics 里带 offline_demo / placeholder 标记时，
    # 在 STATUS 行强制加 [SIMULATED] 前缀 —— 让「仿真」在报告最显眼处出现，
    # 而不是埋在 metrics.json 深处等人去翻。
    simulated = bool(key_metrics.get("offline_demo") or key_metrics.get("placeholder"))
    status_line = f"[SIMULATED] {status}" if simulated else status

    # 组装 Markdown 行：json 字段用 ensure_ascii=False 保持中文可读
    lines = [
        "# Task Report",
        "",
        f"- TASK_ID: `{task_id}`",
        f"- STATUS: **{status_line}**",
        f"- Generated at: {_dt.datetime.now().isoformat()}",
        "",
    ]
    if simulated:
        # 醒目警告块：紧跟在头部字段之后，任何人打开报告第一屏就能看到
        lines += [
            "> ⚠️ **本任务结果为离线模拟/占位数据（offline demo / placeholder），"
            "不可作为真实实验证据**（design.md §75）。",
            "",
        ]
    lines += [
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
    # 各字段间以空行分隔（"\n".join(lines)）；utf-8 保证中文正常写入
    (run_dir / "task_report.md").write_text("\n".join(lines), encoding="utf-8")