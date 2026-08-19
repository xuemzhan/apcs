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

    双轨（§29）：
        - 默认（provider.kv 未设 / "synthetic"）：synthetic 样本 → _simulated_replay，
          恒 PASS，offline_demo=True；
        - 真实 GPU（provider.kv == "hf"）：HFKVProvider 提供真实 Student 自产 KV，
          走 _real_replay（原生 vs 注入 logit 对比）；GPU 不可用时显式 raise
          （§75），offline_demo=False。
    """
    from ..data import synthetic_fidelity_set

    kind = cfg.get("provider", {}).get("kv", "synthetic").lower()
    if kind == "hf":
        results, offline, note = _real_replay(cfg, run_dir)
    else:
        # §29 合成保真样本（n=4）：每条含 sample_id 与 token 数据
        samples = synthetic_fidelity_set(n=4)
        results = [_simulated_replay(s) for s in samples]
        offline = True
        note = (
            "T01 为离线模拟：_simulated_replay 无条件返回理想值，Gate 0 恒 PASS，"
            "不可作为 Engineering Correctness 的真实证据；"
            "design.md §29/§75 要求真实 Student 自产 KV 注入重放验证。"
        )

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
        "gate0": "PASS" if results and all(r.passed for r in results) else "FAIL",
        # ---- §75 诚实性标注（架构审查 P1-4 修复 + real-gpu 接入）----
        # 合成路径：_simulated_replay **无条件**返回理想值 → Gate 0 恒 PASS，
        #   是构造结果不是测量结果，故 offline_demo=True + 醒目警告；
        # 真实路径（provider.kv=hf）：真实 logit 对比，offline_demo=False。
        "offline_demo": offline,
        "note": note,
    }
    write_json(run_dir / "metrics.json", metrics)
    warning = (
        "> ⚠️ **offline demo**：本任务为离线模拟，`_simulated_replay` 无条件返回理想值，"
        "Gate 0 **恒 PASS**，不可作为真实 Engineering Correctness 的证据"
        "（design.md §29 / §75）。\n\n"
        if offline
        else "> ✅ **真实 GPU 路径**：本任务为真实 Student 自产 KV 注入重放\n\n"
    )
    summary = (
        "# T01 Self-KV Replay\n\n"
        + warning
        + f"- Samples: {metrics['n_samples']}\n"
        f"- Mean logit cosine: {metrics['mean_logit_cosine']:.4f}\n"
        f"- Mean max error: {metrics['mean_max_error']:.4e}\n"
        f"- Mean token agreement: {metrics['mean_token_agreement']:.4f}\n"
        f"- **Gate 0: {metrics['gate0']}**\n\n"
        "若 FAIL → 禁止进行 Cross-Model Mapper 实验 (design.md §5 / Gate 0)。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    return {"status": metrics["gate0"], "metrics": metrics, "summary": summary}


def _real_replay(cfg: dict[str, Any], run_dir) -> tuple[list[ReplayResult], bool, str]:
    """真实 GPU Self-KV Replay（§29）：Student 自产 KV 保存-注入 vs 原生推理。

    流程（对每个 eval 样本）：
        1. tokenize prompt → Student forward_prefill 抓 (L,S,H,2D) numpy KV；
        2. inject 该 KV 还原 DynamicCache，逐步 decode 得到序列 A；
        3. 由同一份 numpy KV 重建缓存再 decode 得序列 B（模拟保存-重放）；
        4. 对比 token 序列一致性 + 上界 cosine/error 指标。
    GPU 不可计算 / 模型不可加载 → 显式 raise（§75，不静默回退合成）。
    """
    from ..data.hf_dataset import load as load_dataset, split_manifest
    from ..inference.backends import TorchBackend, _build_cache_from_kv

    # 先做 GPU/torch 能力检查；真实路径不可用时给出明确原因，而不是被后续
    # 数据配置错误遮蔽。这里只校验，不加载模型权重。
    backend = TorchBackend()
    backend._resolve_device()
    datasets_cfg = cfg.get("datasets", {})
    names = datasets_cfg.get("fidelity", []) or []
    if not names:
        raise RuntimeError("真实 T01 需要 cfg.datasets.fidelity 至少一个数据集")
    n_samples = int(cfg.get("replay", {}).get("n_samples", 4))
    n_steps = int(cfg.get("replay", {}).get("decode_steps", 8))
    samples = load_dataset(str(names[0]), n=n_samples, split="test", seed=1)
    write_json(
        run_dir / "dataset_manifest.json",
        split_manifest(str(names[0]), samples, "test", 1),
    )

    student_run = dict(cfg.get("student", {}))
    student_run.setdefault("role", "student")
    model = backend.load_model(student_run)
    results = []
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

            # 原生路径保留 HF 返回的 cache；重放路径只消费其 numpy 序列化副本。
            kv_native, native_cache, _ = backend.forward_prefill_native(model, tok_ids[:-1])
            replay_cache = _build_cache_from_kv(
                kv_native, device=model.device, dtype=model.model.dtype
            )
            native_input = replay_input = tok_ids[-1]
            cosines: list[float] = []
            errors: list[float] = []
            agreements: list[float] = []
            for _ in range(max(1, n_steps)):
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
            max_error = float(np.max(errors))
            agreement = float(np.mean(agreements))
            results.append(
                ReplayResult(
                    sample_id=sample.sample_id,
                    logit_cosine=mean_cos,
                    max_error=max_error,
                    token_agreement=agreement,
                    passed=(
                        mean_cos >= cos_min
                        and max_error <= err_max
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
__all__ = ["ReplayResult", "run_self_kv_replay", "cosine", "jcr"]
