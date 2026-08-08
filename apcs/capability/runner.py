"""T07 Teacher Gap Freeze（design.md §16 / §36 / §48）。

═══════════════════════════════════════════════════════════════════════════════
流程：
1. 在 Validation split 上分别跑 Teacher 与 Student，得到 score_t、score_s；
2. 划分 Low / Medium / High Gap（§48）；
3. 写 teacher_gap.json 作为后续 T08/T09 的冻结输入；
4. test 不可用（§36 / §70）。

本模块当前为离线数值演示 (CPU)；真实 GPU 实验时把 evaluate() 替换为
HF evaluate 即可。train/val/test 严格分层（§16）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..data import synthetic_teacher_advantage_set
from ..io.runs import write_json


def _synthetic_teacher_score(seed: int) -> float:
    """模拟 Teacher 模型得分（高于 Student）。"""
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.65, 0.85))


def _synthetic_student_score(seed: int) -> float:
    """模拟 Student 模型得分（低于 Teacher）。"""
    rng = np.random.default_rng(seed + 7919)
    return float(rng.uniform(0.40, 0.70))


def _bucket(gap: float) -> str:
    """§48 按 gap 大小分桶。"""
    if gap < 0.10:
        return "low"
    if gap < 0.25:
        return "medium"
    return "high"


def run_main_capability(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T09 CLI 入口（thin wrapper，转调 main.py）。"""
    from .main import run_main_capability as _impl

    return _impl(cfg, run_dir)


def run_teacher_gap_freeze(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T07 入口：计算 gap 分布 + 输出 teacher_gap.json + 触发 PREREGISTRATION.md。"""
    samples = synthetic_teacher_advantage_set(n=64)
    rows = []
    for s in samples:
        seed = int(s.sample_id.split("-")[-1])
        st = _synthetic_teacher_score(seed)
        ss = _synthetic_student_score(seed)
        gap = st - ss
        rows.append(
            {
                "sample_id": s.sample_id,
                "split": s.split,
                "score_teacher": st,
                "score_student": ss,
                "gap": gap,
                "bucket": _bucket(gap),
            }
        )

    # 仅 validation 用于冻结（§36）
    val_rows = [r for r in rows if r["split"] == "validation"]
    summary = {
        "n_total": len(rows),
        "n_validation": len(val_rows),
        "n_train": len([r for r in rows if r["split"] == "train"]),
        "gap_distribution": {
            "low": sum(1 for r in val_rows if r["bucket"] == "low"),
            "medium": sum(1 for r in val_rows if r["bucket"] == "medium"),
            "high": sum(1 for r in val_rows if r["bucket"] == "high"),
        },
        "mean_gap_validation": float(np.mean([r["gap"] for r in val_rows])),
    }
    write_json(run_dir / "teacher_gap.json", {"rows": rows, "summary": summary})
    write_json(
        run_dir / "metrics.json",
        {"task": "T07", "frozen_after": "validation", "summary": summary},
    )
    md = (
        "# T07 Teacher Gap Freeze\n\n"
        f"- Total: {summary['n_total']} (train={summary['n_train']}, "
        f"validation={summary['n_validation']})\n"
        f"- Mean gap (validation): {summary['mean_gap_validation']:.4f}\n"
        f"- Distribution: {summary['gap_distribution']}\n\n"
        "Test set 不可用于调整 teacher guidance / α_max / rank (design.md §36)。\n"
        "t07 后会自动生成 PREREGISTRATION.md（§70）。\n"
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    return {"status": "OK", "metrics": summary, "summary": md}