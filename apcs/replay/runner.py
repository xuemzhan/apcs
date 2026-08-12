"""T01 Self-KV Replay：验证 Student 自产 KV 保存-注入与原生推理等价（design.md §29 / H0 / Gate 0）。

═══════════════════════════════════════════════════════════════════════════════
流程（§29）：
    Student Prefill(X)
        ↓
    save C_S(X)
        ↓
    Native Decode(q)             ← reference
        ↓
    clear
        ↓
    inject C_S(X)
        ↓
    q only
        ↓
    Decode                        ← 待对比

比较：Logit Cosine / Max Error / Token Agreement

§5 Gate 0：若 Self-KV Replay 失败 → 禁止进行 Cross-Model Mapper 实验。
§52 禁止 8：Cache 注入失败后禁止 Silent Re-prefill。

双轨实现：
1. 离线模拟（_simulated_replay）：用于 CI/单元测试，返回 1.0 一致。
2. 真实 GPU 实验：把 _simulated_replay 替换为 HuggingFace model.generate() +
   past_key_values 注入 + logit 对比。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..metrics import cosine, jcr
from ..io.runs import write_json


@dataclass
class ReplayResult:
    """单条 Self-KV Replay 结果。

    字段含义（§29 对比指标）：
        sample_id:       样本标识（来自 synthetic_fidelity_set）
        logit_cosine:    原生推理 vs 注入推理的 logit 余弦相似度（=1.0 等价）
        max_error:       logit 最大绝对误差（=0.0 等价）
        token_agreement: 输出 token 一致率（=1.0 等价）
        passed:          是否通过 Gate 0（三个指标同时达标）

    边界（数据流）：单条 passed 决定汇总 gate0 —— 只要有一条样本
    未通过，Gate 0 即为 FAIL 并阻断后续 Cross-Model Mapper 实验（§5）。
    """
    sample_id: str
    logit_cosine: float
    max_error: float
    token_agreement: float
    passed: bool


def _simulated_replay(sample: Any) -> ReplayResult:
    """离线模拟：原生推理与注入推理完全等价 → 1.0 一致性。

    用于 CPU/CI 环境验证 Self-KV 通路本身的正确性。
    真实 GPU 实验请使用 HF model + past_key_values 替换。

    缓存语义：同一份 C_S(X)（Student 自产 KV）先走原生 Decode 作
    reference，再 clear 后 inject 重放走 Decode —— 模拟实现把
    "保存-注入"视为无损，故三个指标全为理想值；这正是 Gate 0
    （Engineering Correctness）在该环境下的上界参考。
    """
    return ReplayResult(
        sample_id=sample.sample_id,
        logit_cosine=1.0,
        max_error=0.0,
        token_agreement=1.0,
        passed=True,
    )


def run_self_kv_replay(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """CLI 入口：跑一遍 Self-KV Replay。

    流程（§29）：synthetic 样本 → 逐条 _simulated_replay →
    汇总均值 + gate0 判定 → 写 metrics.json / summary.md。
    cfg 当前仅用于预留（样本数 n=4 为 demo 固定值），真实实验替换
    _simulated_replay 后，cfg 可控制样本集 / 上下文长度。
    """
    from ..data import synthetic_fidelity_set

    # §29 合成保真样本（n=4）：每条含 sample_id 与 token 数据
    samples = synthetic_fidelity_set(n=4)
    results = [_simulated_replay(s) for s in samples]

    metrics = {
        "task": "T01",
        "n_samples": len(results),
        "mean_logit_cosine": float(
            sum(r.logit_cosine for r in results) / max(1, len(results))
        ),
        "mean_max_error": float(
            sum(r.max_error for r in results) / max(1, len(results))
        ),
        "mean_token_agreement": float(
            sum(r.token_agreement for r in results) / max(1, len(results))
        ),
        # Gate 0（§5）：全部样本 passed 才 PASS，任一失败即 FAIL
        # 边界（防除零）：均值分母用 max(1, len(results))，空结果时
        # 三项均值落在理想值 1.0 / 0.0 上而 gate0 由 all([]) → True
        "gate0": "PASS" if all(r.passed for r in results) else "FAIL",
        # ---- §75 诚实性标注（架构审查 P1-4 修复）----
        # ★ 关键：_simulated_replay **无条件**返回理想值（cosine=1.0/err=0/agree=1.0），
        #   因此 Gate 0 在当前实现下**恒 PASS**，这是构造出来的结果、不是测量结果。
        #   一份写着「Gate 0 PASS」却不带任何仿真标记的报告最容易被误读为真实证据，
        #   故与 T09 一致地落 offline_demo/note，并在 summary 顶部给出醒目警告。
        # TODO(real-gpu): 接入真实 GPU 路径时替换 _simulated_replay 为
        #   HF model + past_key_values 注入重放，并把 offline_demo 置 False。
        "offline_demo": True,
        "note": (
            "T01 为离线模拟：_simulated_replay 无条件返回理想值，Gate 0 恒 PASS，"
            "不可作为 Engineering Correctness 的真实证据；"
            "design.md §29/§75 要求真实 Student 自产 KV 注入重放验证。"
        ),
    }
    write_json(run_dir / "metrics.json", metrics)
    summary = (
        "# T01 Self-KV Replay\n\n"
        "> ⚠️ **offline demo**：本任务为离线模拟，`_simulated_replay` 无条件返回理想值，"
        "Gate 0 **恒 PASS**，不可作为真实 Engineering Correctness 的证据"
        "（design.md §29 / §75）。\n\n"
        f"- Samples: {metrics['n_samples']}\n"
        f"- Mean logit cosine: {metrics['mean_logit_cosine']:.4f}\n"
        f"- Mean max error: {metrics['mean_max_error']:.4e}\n"
        f"- Mean token agreement: {metrics['mean_token_agreement']:.4f}\n"
        f"- **Gate 0: {metrics['gate0']}**（模拟值）\n\n"
        "若 FAIL → 禁止进行 Cross-Model Mapper 实验 (design.md §5 / Gate 0)。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    return {"status": metrics["gate0"], "metrics": metrics, "summary": summary}


# 对外导出：ReplayResult（结果容器）+ 入口函数 + 复用 metrics 工具，
# 便于测试与真实 GPU 分支直接 import（§52 禁止 8：不提供静默 re-prefill 捷径）
__all__ = ["ReplayResult", "run_self_kv_replay", "cosine", "jcr"]