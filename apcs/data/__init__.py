"""数据集加载（design.md §14-§18 数据集分层）。

═══════════════════════════════════════════════════════════════════════════════
数据集分类（§14：不能用同一 benchmark 同时回答所有问题）：
    §15 Fidelity Set              目的：Handoff 是否破坏 Student 基础能力
                                  候选：HellaSwag, ARC-Challenge, MMLU subset, WinoGrande
                                  输出：Score, Retention, KL, Token Agreement
    §16 Teacher-Advantage Set     目的：Teacher 相对 Student 的优势能否迁移
                                  允许：dataset-level 上选 Teacher 稳定优于 Student 的任务
                                  禁止：test 后挑 "Teacher 对 / Student 错" 的题
                                  必须严格分 Train / Validation / Test
                                  Train: 计算 Teacher Margin / Guidance Weight
                                  Validation: 决定 Rank / α_max / Top-k / Loss Weight
                                  Test: 完全冻结（§36 / §70）
    §17 Long-Context Set          建议：RULER, LongBench, Multi-document QA, Multi-hop QA,
                                  Retrieval / Needle；长度 4K/8K/16K/32K
    §18 Behavior-Sensitive Set    LLM Judge, Ranker, Multi-candidate comparison
                                  检查 Accuracy 不变时 Decision Behavior 是否变化

§52 禁止 4：只挑 Teacher-win Test Sample → 设计上用 dataset-level 选取。

本模块提供最小可复现的合成数据（用于 CI/单元测试）；
真实实验由各 task 替换为 HF datasets 加载。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    """统一 sample 结构。"""
    sample_id: str
    context: str
    query: str
    answer: str | None = None
    split: str = "test"            # train | validation | test（§16）


def synthetic_fidelity_set(n: int = 8) -> list[Sample]:
    """§15 Fidelity Set 最小可复现样本（用于 T01 / T05 fidelity 通路验证）。"""
    out: list[Sample] = []
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
    )
    for i in range(n):
        ctx = base * (i + 1)
        out.append(
            Sample(
                sample_id=f"fidelity-{i}",
                context=ctx,
                query="What animal is mentioned first?",
                answer="fox",
                split="test",
            )
        )
    return out


def synthetic_teacher_advantage_set(n: int = 16) -> list[Sample]:
    """§16 Teacher-Advantage Set 最小样本。

    注意：
        - 一半 train（计算 Teacher margin） + 一半 validation（决定超参）
        - 真实实现禁止使用 test 做超参搜索（§36）
    """
    out: list[Sample] = []
    for i in range(n):
        ctx = (
            f"Consider this fact table #{i}: "
            "Alpha is greater than Beta. Beta is greater than Gamma. "
            "Gamma is greater than Delta. "
        )
        out.append(
            Sample(
                sample_id=f"ta-{i}",
                context=ctx,
                query="Which is greater, Alpha or Delta?",
                answer="Alpha",
                split="train" if i < n // 2 else "validation",
            )
        )
    return out


def synthetic_long_context_set(n: int = 4, length_tokens_approx: int = 512) -> list[Sample]:
    """§17 Long-Context Set 最小样本。"""
    block = "word " * 32
    out: list[Sample] = []
    for i in range(n):
        ctx = block * (max(1, length_tokens_approx // 32))
        out.append(
            Sample(
                sample_id=f"lc-{i}",
                context=ctx,
                query="Repeat the last word of the context.",
                answer="word",
                split="test",
            )
        )
    return out


def synthetic_behavior_set(n: int = 6) -> list[Sample]:
    """§18 Behavior-Sensitive Set 最小样本（Judge / Ranker 用）。"""
    out: list[Sample] = []
    for i in range(n):
        ctx = f"Decision scenario #{i}. Option A is safe. Option B is risky but profitable."
        out.append(
            Sample(
                sample_id=f"beh-{i}",
                context=ctx,
                query="Choose the safer option.",
                answer="A",
                split="test",
            )
        )
    return out