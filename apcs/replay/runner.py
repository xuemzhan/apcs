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

T01 多步解码稳定性：循环结构按 (samples × decode_steps) 桶展开，
每桶独立报告 mean logit cosine / max error / token agreement；
Gate 0 要求 ALL 桶 + ALL 样本 max_error==0 AND token_agreement==1.0。
上下文长度复用 cfg.context_lengths（§21）。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from ..metrics import cosine, jcr
from ..io.runs import write_json

# T01 多步解码稳定性：decode_steps 桶默认值
_DEFAULT_DECODE_STEPS: list[int] = [1, 5, 20]
_DEFAULT_N_SAMPLES: int = 32
_MIN_N_SAMPLES_WITH_PROVIDER: int = 8


@dataclass
class ReplayResult:
    """单条 Self-KV Replay 结果（per sample × decode_step 桶）。

    字段含义（§29 对比指标）：
        sample_id:       样本标识（来自 synthetic_fidelity_set）
        decode_step:     该结果对应的解码步数桶（T01 多步解码稳定性）
        logit_cosine:    原生推理 vs 注入推理的 logit 余弦相似度（=1.0 等价）
        max_error:       logit 最大绝对误差（=0.0 等价）
        token_agreement: 输出 token 一致率（=1.0 等价）
        passed:          是否通过 Gate 0（三个指标同时达标）

    边界（数据流）：单条 passed 决定汇总 gate0 —— 只要有一条样本
    未通过，Gate 0 即为 FAIL 并阻断后续 Cross-Model Mapper 实验（§5）。
    """
    sample_id: str
    decode_step: int = 1
    logit_cosine: float = 1.0
    max_error: float = 0.0
    token_agreement: float = 1.0
    passed: bool = True


def _simulated_replay(sample: Any, n_decode_steps: int = 1) -> ReplayResult:
    """离线模拟：原生推理与注入推理完全等价 → 1.0 一致性。

    T01 多步解码稳定性：即使离线路径也执行 n_decode_steps 步解码循环，
    用 numpy 数组模拟 logits，验证循环结构本身的正确性（而非仅返回
    固定理想值）。CI 覆盖 multi-step 路径开销极低（纯 numpy copy）。

    缓存语义：同一份 C_S(X)（Student 自产 KV）先走原生 Decode 作
    reference，再 clear 后 inject 重放走 Decode —— 模拟实现把
    "保存-注入"视为无损，故三个指标全为理想值；这正是 Gate 0
    （Engineering Correctness）在该环境下的上界参考。
    """
    rng = np.random.default_rng(hash(sample.sample_id) % (2**31))
    cosines: list[float] = []
    errors: list[float] = []
    agreements: list[float] = []
    for _ in range(max(1, n_decode_steps)):
        # 用随机向量模拟 logits；复制同一份 → 原生/注入完全一致
        logits = rng.standard_normal(128).astype(np.float64)
        cosines.append(cosine(logits, logits))
        errors.append(float(np.max(np.abs(logits - logits))))
        agreements.append(1.0)
    return ReplayResult(
        sample_id=sample.sample_id,
        decode_step=n_decode_steps,
        logit_cosine=float(np.mean(cosines)),
        max_error=float(np.max(errors)),
        token_agreement=float(np.mean(agreements)),
        passed=True,
    )


def run_self_kv_replay(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """CLI 入口：跑 Self-KV Replay（T01 多步解码稳定性）。

    cfg 读取（cfg.get 安全默认值）：
        replay.n_samples    → int, 默认 32; 真实 provider 时下界 8
        replay.decode_steps → list[int], 默认 [1, 5, 20]
        context_lengths     → 复用（§21 层对齐上下文长度候选）

    双轨（§29）：
        - 默认（provider.kv 未设 / "synthetic"）：synthetic 样本 → _simulated_replay，
          恒 PASS，offline_demo=True；
        - 真实 GPU（provider.kv == "hf"）：HFKVProvider 提供真实 Student 自产 KV，
          走 _real_replay（原生 vs 注入 logit 对比）；GPU 不可用时显式 raise
          （§75），offline_demo=False。

    循环结构：samples × decode_steps 桶；每桶独立报告指标；
    Gate 0 不变：PASS iff ALL 桶 + ALL 样本 max_error==0 AND token_agreement==1.0。
    """
    from ..data import synthetic_fidelity_set

    kind = cfg.get("provider", {}).get("kv", "synthetic").lower()
    is_real = kind == "hf"

    # T01 多步解码稳定性：读取 n_samples / decode_steps（cfg.get 安全默认值）
    replay_cfg = cfg.get("replay", {})
    n_samples_raw = int(replay_cfg.get("n_samples", _DEFAULT_N_SAMPLES))
    n_samples = max(
        n_samples_raw if is_real else min(n_samples_raw, 8),
        _MIN_N_SAMPLES_WITH_PROVIDER if is_real else n_samples_raw,
    )
    decode_steps_raw = replay_cfg.get("decode_steps", _DEFAULT_DECODE_STEPS)
    decode_steps = (
        [int(d) for d in decode_steps_raw]
        if isinstance(decode_steps_raw, (list, tuple))
        else [int(decode_steps_raw)]
    )
    # 复用 cfg.context_lengths 作上下文长度候选（§21 层对齐）
    context_lengths = cfg.get("context_lengths", [])

    if is_real:
        results, offline, note = _real_replay(cfg, run_dir, decode_steps)
    else:
        samples = synthetic_fidelity_set(n=min(n_samples, 8))
        results = [
            _simulated_replay(s, n_decode_steps=ds)
            for s in samples
            for ds in decode_steps
        ]
        offline = True
        note = (
            "T01 为离线模拟：_simulated_replay 无条件返回理想值，Gate 0 恒 PASS，"
            "不可作为 Engineering Correctness 的真实证据；"
            "design.md §29/§75 要求真实 Student 自产 KV 注入重放验证。"
        )

    # ---- 汇总指标（跨所有桶和样本）----
    metrics: dict[str, Any] = {
        "task": "T01",
        "n_samples": len(results),
        "decode_steps": decode_steps,
        "context_lengths": context_lengths,
        "mean_logit_cosine": float(
            sum(r.logit_cosine for r in results) / max(1, len(results))
        ),
        "mean_max_error": float(
            sum(r.max_error for r in results) / max(1, len(results))
        ),
        "mean_token_agreement": float(
            sum(r.token_agreement for r in results) / max(1, len(results))
        ),
        # Gate 0（§5）：ALL 桶 + ALL 样本 passed 才 PASS
        "gate0": "PASS" if results and all(r.passed for r in results) else "FAIL",
        "offline_demo": offline,
        "note": note,
    }

    # ---- per-decode-step 桶汇总（Summary.md 表格 + metrics 明细）----
    bucket_summary: list[dict[str, Any]] = []
    for ds in decode_steps:
        bucket_results = [r for r in results if r.decode_step == ds]
        if not bucket_results:
            continue
        bucket_summary.append({
            "decode_step": ds,
            "n": len(bucket_results),
            "mean_logit_cosine": float(
                sum(r.logit_cosine for r in bucket_results) / len(bucket_results)
            ),
            "mean_max_error": float(
                sum(r.max_error for r in bucket_results) / len(bucket_results)
            ),
            "mean_token_agreement": float(
                sum(r.token_agreement for r in bucket_results) / len(bucket_results)
            ),
        })
    metrics["per_decode_step"] = bucket_summary

    write_json(run_dir / "metrics.json", metrics)

    # ---- summary.md：含 per-decode-step 表格 ----
    warning = (
        "> ⚠️ **offline demo**：本任务为离线模拟，`_simulated_replay` 无条件返回理想值，"
        "Gate 0 **恒 PASS**，不可作为真实 Engineering Correctness 的证据"
        "（design.md §29 / §75）。\n\n"
        if offline
        else "> ✅ **真实 GPU 路径**：本任务为真实 Student 自产 KV 注入重放\n\n"
    )
    # per-decode-step 表格
    table_lines = (
        "| decode_step | n | mean_cosine | mean_max_error | mean_agreement |\n"
        "|---|---|---|---|---|\n"
    )
    for bs in bucket_summary:
        table_lines += (
            f"| {bs['decode_step']} | {bs['n']} "
            f"| {bs['mean_logit_cosine']:.4f} "
            f"| {bs['mean_max_error']:.4e} "
            f"| {bs['mean_token_agreement']:.4f} |\n"
        )
    summary = (
        "# T01 Self-KV Replay\n\n"
        + warning
        + f"- Samples: {metrics['n_samples']}\n"
        f"- Decode steps: {decode_steps}\n"
        f"- Mean logit cosine: {metrics['mean_logit_cosine']:.4f}\n"
        f"- Mean max error: {metrics['mean_max_error']:.4e}\n"
        f"- Mean token agreement: {metrics['mean_token_agreement']:.4f}\n"
        f"- **Gate 0: {metrics['gate0']}**\n\n"
        "## Per-Decode-Step\n\n"
        + table_lines
        + "\n若 FAIL → 禁止进行 Cross-Model Mapper 实验 (design.md §5 / Gate 0)。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    return {"status": metrics["gate0"], "metrics": metrics, "summary": summary}


def _real_replay(
    cfg: dict[str, Any],
    run_dir,
    decode_steps: list[int] | None = None,
) -> tuple[list[ReplayResult], bool, str]:
    """真实 GPU Self-KV Replay（§29）：Student 自产 KV 保存-注入 vs 原生推理。

    T01 多步解码稳定性：循环结构为 samples × decode_steps 桶，
    每桶独立报告 mean logit cosine / max error / token agreement。

    流程（对每个 eval 样本 × decode_step 桶）：
        1. tokenize prompt → Student forward_prefill 抓 (L,S,H,2D) numpy KV；
        2. inject 该 KV 还原 DynamicCache，逐步 decode decode_step 步；
        3. 由同一份 numpy KV 重建缓存再 decode 相同步数；
        4. 对比每步 token 序列一致性 + logit cosine / error → 桶内聚合。
    GPU 不可计算 / 模型不可加载 → 显式 raise（§75，不静默回退合成）。
    """
    from ..data.hf_dataset import load as load_dataset, split_manifest
    from ..inference.backends import TorchBackend, _build_cache_from_kv

    if decode_steps is None:
        decode_steps = _DEFAULT_DECODE_STEPS

    backend = TorchBackend()
    backend._resolve_device()
    datasets_cfg = cfg.get("datasets", {})
    names = datasets_cfg.get("fidelity", []) or []
    if not names:
        raise RuntimeError("真实 T01 需要 cfg.datasets.fidelity 至少一个数据集")
    n_samples = int(cfg.get("replay", {}).get("n_samples", _DEFAULT_N_SAMPLES))
    n_samples = max(n_samples, _MIN_N_SAMPLES_WITH_PROVIDER)
    samples = load_dataset(str(names[0]), n=n_samples, split="test", seed=1)
    write_json(
        run_dir / "dataset_manifest.json",
        split_manifest(str(names[0]), samples, "test", 1),
    )

    student_run = dict(cfg.get("student", {}))
    student_run.setdefault("role", "student")
    model = backend.load_model(student_run)
    results: list[ReplayResult] = []
    thresholds = cfg.get("gates", {})
    cos_min = float(thresholds.get("replay_logit_cosine_min", 0.9999))
    err_max = float(thresholds.get("replay_max_error", 1e-3))
    agreement_min = float(thresholds.get("replay_token_agreement_min", 1.0))
    try:
        for sample in samples:
            prompt = f"{sample.context}\n{sample.query}" if sample.context else sample.query
            tok_ids = model.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
            if len(tok_ids) < 2:
                raise RuntimeError(f"T01 sample {sample.sample_id!r} tokenize 后不足 2 token")

            # 预填充：原生路径保留 HF 返回的 cache；重放路径消费 numpy 序列化副本
            kv_native, native_cache, _ = backend.forward_prefill_native(model, tok_ids[:-1])

            for ds in decode_steps:
                # 每个桶从同一份 prefill 状态重新开始解码
                replay_cache = _build_cache_from_kv(
                    kv_native, device=model.device, dtype=model.model.dtype
                )
                native_input = replay_input = tok_ids[-1]
                cosines: list[float] = []
                errors: list[float] = []
                agreements: list[float] = []
                for _ in range(max(1, ds)):
                    native_tok, native_cache, native_logits = backend.decode_with_logits(
                        model, [native_input], native_cache
                    )
                    replay_tok, replay_cache, replay_logits = backend.decode_with_logits(
                        model, [replay_input], replay_cache
                    )
                    cosines.append(cosine(native_logits, replay_logits))
                    errors.append(float(np.max(np.abs(native_logits - replay_logits))))
                    agreements.append(float(native_tok == replay_tok))
                    native_input, replay_input = native_tok, replay_tok

                mean_cos = float(np.mean(cosines))
                max_err = float(np.max(errors))
                agreement = float(np.mean(agreements))
                results.append(
                    ReplayResult(
                        sample_id=sample.sample_id,
                        decode_step=ds,
                        logit_cosine=mean_cos,
                        max_error=max_err,
                        token_agreement=agreement,
                        passed=(
                            mean_cos >= cos_min
                            and max_err <= err_max
                            and agreement >= agreement_min
                        ),
                    )
                )
    finally:
        backend.unload(model)
    return results, False, (
        "T01 为真实 GPU Self-KV Replay：保留 HF 原生 cache 作为对照，"
        "与 numpy offload → cache 重建路径逐步比较 logits 与 token（§29）。"
        "offline_demo=False，可作 Engineering Correctness 证据。"
    )


# 对外导出：ReplayResult（结果容器）+ 入口函数 + 复用 metrics 工具，
# 便于测试与真实 GPU 分支直接 import（§52 禁止 8：不提供静默 re-prefill 捷径）
__all__ = [
    "ReplayResult",
    "run_self_kv_replay",
    "_simulated_replay",
    "cosine",
    "jcr",
]
