"""§52 运行时信号采集器（Runtime Signal Collector）。

═══════════════════════════════════════════════════════════════════════════════
§52 八条禁止依赖运行时信号（如 "student_input_has_context"、
"student_params_updated" 等）。这些信号无法从静态 cfg 推断，必须从
训练/推理循环中自动捕获。

本模块提供：
    - RuntimeSignals: 线程安全的 dict，存所有运行时信号
    - track(name, value): 写入信号（用作 decorator 或 context manager）
    - collect(): 返回当前所有信号（dict 快照）
    - reset(): 清空

典型用法：
    # 在 mapper.transform 前后包起来
    with track_runtime("student_input_has_context", False):
        pred = mapper.transform(kv_t, layer_map)

    # 在 orchestrator 调度前
    from apcs.compliance import check_all
    violations = check_all(cfg, run_dir, collect_runtime())
    if violations:
        raise ComplianceError(violations)

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import contextlib
import threading
from typing import Any, Iterator

# 全局线程安全：所有读写统一走 _LOCK。
# _SIGNALS 是模块级状态（跨 Run 共享），reset() 在每个 Run 开始时清空，
# 避免上一 Run 的信号串扰到下一 Run 的检查结果。
_LOCK = threading.Lock()
_SIGNALS: dict[str, Any] = {}


def reset() -> None:
    """清空所有信号（一般在每次 Run 开始时调用）。

    若不清空，上一 Run 残留信号（如 student_input_has_context=True）会
    被下一 Run 的 check_all 读到，造成误报违规。
    """
    with _LOCK:
        _SIGNALS.clear()


def track(name: str, value: Any) -> None:
    """直接写入信号。

    原子写：与 collect() 并发时不会读到半写状态。
    value 为任意类型（bool / int / float / str / dict），由 check_X 自行解释。
    """
    with _LOCK:
        _SIGNALS[name] = value


def get(name: str, default: Any = None) -> Any:
    """读取单个信号；不存在时返回 default。

    default 通常对应 check_X 的"未违规"默认值（如 §52.5 的 True）。
    """
    with _LOCK:
        return _SIGNALS.get(name, default)


def collect() -> dict[str, Any]:
    """返回当前所有信号的快照（线程安全 copy）。

    返回 dict 副本而非引用，调用方后续改动不会污染模块内部状态；
    orchestrator 在每个 task 收尾时调用一次，交给 check_all 判定。
    """
    with _LOCK:
        return dict(_SIGNALS)


@contextlib.contextmanager
def track_runtime(name: str, value: Any = True) -> Iterator[None]:
    """Context manager：在 with 块内设置信号，退出时恢复旧值。

    例如：
        with track_runtime("student_input_has_context", True):
            # ... Student forward 接收了 context
            ...
        # 块结束 → 信号清除（恢复 default=None）

    与 track() 的差异：track 是"常驻"写入（覆盖旧值、不自动撤销）；
    track_runtime 是"临时"埋点，精确限定在可疑代码路径的作用域内，
    避免因异常/提前 return 导致信号残留。
    """
    with _LOCK:
        prev = _SIGNALS.get(name)   # 保存旧值（可能为 None = 从未设置）
        _SIGNALS[name] = value      # 块内生效
    try:
        yield
    finally:
        with _LOCK:
            # 退出时恢复旧值：原本不存在则删除，原本存在则还原
            if prev is None:
                _SIGNALS.pop(name, None)
            else:
                _SIGNALS[name] = prev


# ---- 自动信号推断工具 ----


def infer_signals_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """从 cfg 静态推断部分信号。

    能推断的：
        - reports_teacher_prefill: cfg["mapper"] 里有 calibration/eval 上下文即为 True
        - student_params_updated: cfg["student"]["freeze"]==True 时不可能为 True
        - claim_path_a_on_similarity_only: 总是 False（无法从 cfg 推断）

    不能推断的（运行时才知道）：
        - student_input_has_context（取决于 forward 输入）
        - test_hp_search（取决于循环逻辑）
        - test_filtered_to_teacher_win（取决于 test 评估）
        - hides_h2d_load（取决于报告）
        - silent_re_prefill_on_failure（取决于 cache 注入错误处理）

    静态推断的结果是可复现的基线；真实实验中以 track/track_runtime
    采集的运行时信号为准（后者覆盖前者，见 collect 的合并顺序）。
    """
    return {
        # §52.5 信号：mapper 配置里只要含 calibration 上下文，即认为会报告
        # teacher_prefill（静态推断；运行时仍可 track 覆盖）
        "reports_teacher_prefill": "calibration_context" in cfg.get("mapper", {}),
        # §52.2 信号：freeze=True 时 Student 参数本就不可能被更新 → 静态判定未违规；
        # freeze=False 时静态无法确认是否真被更新，置 True（保守），由运行时信号覆盖
        "student_params_updated": not cfg.get("student", {}).get("freeze", True),
        # §52.7 信号：cfg 层面看不出"只靠相似度主张 Path A"，只能运行时揭发 → 恒 False
        "claim_path_a_on_similarity_only": False,
    }