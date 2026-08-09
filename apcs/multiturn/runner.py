"""§46 Multi-turn stability 实验（design.md §46 / §60 Figure 5）。

═══════════════════════════════════════════════════════════════════════════════
每轮（1 / 5 / 10 / 20）记录：
    - CHG
    - KL
    - JCR (§45)
    - Task Score
    - Latency

最终 Figure 5 同时画 CHG / KL / JCR 三条曲线。

离线实现：用合成数据模拟"轮数对得分的影响"。
真实实验：每轮注入新的 KV cache，模型继续 decode，记录上述指标。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..io.runs import write_json
from ..metrics import jcr, kl_divergence


def _multi_turn_score_decay(turn: int, base: float = 0.66, decay: float = 0.005) -> float:
    """轮数 → 得分的简单模拟：每轮略微衰减（offline demo）。

    边界（数据流）：turn 从 1 起，score = base - decay*(turn-1)，
    即第 1 轮取 base，之后每轮线性衰减；再加 N(0, 0.01) 的噪声模拟
    实验抖动，最后 clip 到 [0, 1] —— 防止衰减+噪声越界为负分或超 1
    （真实实验里该模拟会被逐轮注入 KV cache 的真实 decode 替换）。
    """
    return float(np.clip(base - decay * (turn - 1) + np.random.normal(0, 0.01), 0, 1))


def run_multi_turn(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """§46 多轮稳定性入口。

    turn_choices 默认 1 / 5 / 10 / 20。

    跨轮 KV 缓存语义（真实路径）：每轮对话结束后把 KV cache 保留下来，
    下一轮只对新输入做 decode（注入新 KV、继续复用旧 KV）—— 这正是
    多轮场景里"缓存复用"的核心；指标随轮数变化刻画长期稳定性。
    本离线 demo 用合成得分模拟"轮数 → 得分衰减"，验证 Figure 5
    （CHG / KL / JCR 三曲线）的渲染数据通路。
    """
    turn_choices = cfg.get("multi_turn_choices", [1, 5, 10, 20])
    n_seeds = len(cfg.get("seeds", [0, 1, 2]))
    # 基线：student 自产得分起点 0.50；handoff 合成得分起点 0.66
    student_base = 0.50
    handoff_base = 0.66

    rows = []
    for t in turn_choices:
        # 每轮固定随机种子（0），保证不同轮之间只有"轮数"在变化，
        # 衰减趋势可归因于轮数而非随机性
        np.random.seed(0)
        chg_list = []
        kl_list = []
        jcr_list = []
        score_list = []
        latencies = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            # student 自产得分随轮数轻微下降：模拟长上下文下 student 基线退化
            s_self = student_base - 0.005 * (t - 1)
            # handoff 得分用模拟衰减函数（含随机抖动）
            s_hand = _multi_turn_score_decay(t, handoff_base)
            # CHG = handoff 得分 − student 自产得分（design.md §3 首要科学端点）
            chg_list.append(s_hand - s_self)
            # KL：每轮近似（真实实现用真实输出分布计算）
            p = np.random.dirichlet(np.ones(20))
            q = np.random.dirichlet(np.ones(20))
            kl_list.append(kl_divergence(p, q))
            # JCR：decisions 一致率（§45；handoff 与 student 决策的逐项一致性）
            decs_self = np.random.randint(0, 2, size=50).tolist()
            # 让 handoff 与 student 大致 90% 一致（10% 概率翻转）
            decs_hand = [
                d if np.random.rand() > 0.1 else 1 - d for d in decs_self
            ]
            jcr_list.append(jcr(decs_hand, decs_self))
            score_list.append(s_hand)
            latencies.append(8.0 + t * 0.5)  # 模拟每轮 ~0.5ms 增量
        rows.append(
            {
                "turn": t,
                "chg_mean": float(np.mean(chg_list)),
                "chg_std": float(np.std(chg_list)),
                "kl_mean": float(np.mean(kl_list)),
                "jcr_mean": float(np.mean(jcr_list)),
                "task_score_mean": float(np.mean(score_list)),
                "latency_p50_ms": float(np.median(latencies)),
            }
        )

    # per_turn 行供 Figure 5 绘制 CHG / KL / JCR 随轮数曲线（§60）
    metrics = {"task": "Multi-turn", "per_turn": rows, "n_seeds": n_seeds}
    write_json(run_dir / "metrics.json", metrics)
    md = (
        "# §46 Multi-turn Stability (Figure 5)\n\n"
        "| turns | CHG mean | KL | JCR | task_score | latency p50 |\n"
        "| ----: | -------: | -: | --: | ---------: | ----------: |\n"
        + "\n".join(
            f"| {r['turn']} | {r['chg_mean']:+.4f} | {r['kl_mean']:.4f} | "
            f"{r['jcr_mean']:.4f} | {r['task_score_mean']:.4f} | "
            f"{r['latency_p50_ms']:.2f} |"
            for r in rows
        )
        + "\n"
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    return {"status": "OK", "metrics": metrics, "summary": md}