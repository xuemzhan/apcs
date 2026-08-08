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

_LOCK = threading.Lock()
_SIGNALS: dict[str, Any] = {}


def reset() -> None:
    """清空所有信号（一般在每次 Run 开始时调用）。"""
    with _LOCK:
        _SIGNALS.clear()


def track(name: str, value: Any) -> None:
    """直接写入信号。"""
    with _LOCK:
        _SIGNALS[name] = value


def get(name: str, default: Any = None) -> Any:
    """读取单个信号。"""
    with _LOCK:
        return _SIGNALS.get(name, default)


def collect() -> dict[str, Any]:
    """返回当前所有信号的快照（线程安全 copy）。"""
    with _LOCK:
        return dict(_SIGNALS)


@contextlib.contextmanager
def track_runtime(name: str, value: Any = True) -> Iterator[None]:
    """Context manager：在 with 块内设置信号，退出时清除。

    例如：
        with track_runtime("student_input_has_context", True):
            # ... Student forward 接收了 context
            ...
        # 块结束 → 信号清除（恢复 default=None）
    """
    with _LOCK:
        prev = _SIGNALS.get(name)
        _SIGNALS[name] = value
    try:
        yield
    finally:
        with _LOCK:
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
    """
    return {
        "reports_teacher_prefill": "calibration_context" in cfg.get("mapper", {}),
        "student_params_updated": not cfg.get("student", {}).get("freeze", True),
        "claim_path_a_on_similarity_only": False,
    }