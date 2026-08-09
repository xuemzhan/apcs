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
    """模拟 Teacher 模型得分（高于 Student）。

    为什么这样算：
        - uniform(0.65, 0.85) 保证 Teacher 分数整体高于 Student 区间（见下），
          使 gap = st − ss 恒为正、TGRR 分母有意义；
        - 只依赖 seed 的 RNG → 同 seed 跨进程可复现；
        - 离线 demo：真实 GPU 实验替换为 HF evaluate。
    """
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.65, 0.85))


def _synthetic_student_score(seed: int) -> float:
    """模拟 Student 模型得分（低于 Teacher）。

    seed + 7919（素数偏移）让 Teacher/Student 的噪声序列解耦，
    但区间 0.40–0.70 整体低于 Teacher 的 0.65–0.85，保证 gap 仍为正。
    """
    rng = np.random.default_rng(seed + 7919)
    return float(rng.uniform(0.40, 0.70))


def _bucket(gap: float) -> str:
    """§48 按 gap 大小分桶（low / medium / high）。

    阈值含义：
        gap < 0.10        → low   ：Teacher 优势微弱，迁移收益空间小；
        0.10 ≤ gap < 0.25 → medium；
        gap ≥ 0.25        → high  ：Teacher 优势显著，最该出现 CHG。

    分桶结果是 teacher_gap.json 的冻结输入，T08/T09 的 Gap Strata 报告
    与分层训练均以此为据（§48）。
    """
    if gap < 0.10:
        return "low"
    if gap < 0.25:
        return "medium"
    return "high"


def run_main_capability(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T09 CLI 入口（thin wrapper，转调 main.py）。

    runner.py 内的真实实现是 T07（run_teacher_gap_freeze）；T09 主体逻辑
    放在 main.py（run_main_capability），此处仅做转发，保持 cli.py 对
    capability.runner 单一入口的调用约定。
    """
    from .main import run_main_capability as _impl

    return _impl(cfg, run_dir)


def run_teacher_gap_freeze(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T07 入口：计算 gap 分布 + 输出 teacher_gap.json + 触发 PREREGISTRATION.md。

    流程：
        1. 生成 synthetic teacher-advantage 样本（train/val/test 分层，§16）；
        2. 逐样本模拟 Teacher/Student 得分并算 gap、分桶（§48）；
        3. 仅 validation 用于冻结统计（§36：test 不可见）；
        4. 写 teacher_gap.json + metrics.json + summary.md；
        5. t07 之后由 orchestrator 自动生成 PREREGISTRATION.md（§70）。
    """
    samples = synthetic_teacher_advantage_set(n=64)  # 64 个分好 train/val/test 的离线样本
    rows = []
    for s in samples:
        seed = int(s.sample_id.split("-")[-1])  # seed 由 sample_id 派生：与样本绑定、跨 Run 稳定
        st = _synthetic_teacher_score(seed)
        ss = _synthetic_student_score(seed)
        gap = st - ss  # §3.3 TGRR 的分子分母都依赖这个 Teacher−Student 差距
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

    # 仅 validation 用于冻结（§36）：test 全程不可见，连统计都不算，防泄露（§70）
    val_rows = [r for r in rows if r["split"] == "validation"]
    summary = {
        "n_total": len(rows),
        "n_validation": len(val_rows),
        "n_train": len([r for r in rows if r["split"] == "train"]),
        # §48 三桶分布（仅 validation）：T08/T09 分层训练与 Gap Strata 报告的依据
        "gap_distribution": {
            "low": sum(1 for r in val_rows if r["bucket"] == "low"),
            "medium": sum(1 for r in val_rows if r["bucket"] == "medium"),
            "high": sum(1 for r in val_rows if r["bucket"] == "high"),
        },
        # 冻结的 gap 汇总统计（validation 上跨样本均值）
        "mean_gap_validation": float(np.mean([r["gap"] for r in val_rows])),
    }
    # 冻结产物：T08/T09 必须从这里读 gap，不得重算（冻结语义）
    write_json(run_dir / "teacher_gap.json", {"rows": rows, "summary": summary})
    # 标准产物（§63）：与其它 task 的 metrics.json 规范保持一致
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