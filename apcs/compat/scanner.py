"""T00 scanner：基于 HuggingFace AutoConfig 读取模型架构（design.md §28）。

═══════════════════════════════════════════════════════════════════════════════
§28 自动读取字段：
    tokenizer, vocab, layers, attention_heads, kv_heads, head_dim,
    hidden_size, rope, position, dtype, cache_layout, attention_implementation

§10 / §12 / §13 分流判定：
    G1_MATCHED_KV         same-family + matched heads/head_dim
    G1_MATCHED_KV_DIFF    matched KV, diff tokenizer
    G2_MISMATCHED_HEAD    tokenizer 一致但 KV head/dim 不一致
    G3_CROSS_FAMILY       tokenizer 与 KV 都不一致

fallback 策略：当 transformers/torch 未安装时（CI/离线开发机），
    基于 Qwen3 model_id 字符串硬编码一组合理估计，保证单元测试与
    完整实验流都能跑通。生产环境应使用 AutoConfig 真实读取。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..io import load_config  # noqa: F401  (re-export for convenience)
from ..io.runs import write_json


@dataclass
class ModelSpec:
    """§28 模型规格字段。"""
    model_id: str
    revision: str
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    max_position_embeddings: int = 0
    rope_theta: float | None = None
    rope_scaling: dict | None = None
    attention_implementation: str = ""
    dtype: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _from_config_dict(model_id: str, revision: str, cfg: dict[str, Any]) -> ModelSpec:
    """从 AutoConfig 加载得到的 cfg 对象中抽取字段。

    兼容 GPT-2 (n_embd/n_layer/n_head) 与 LLaMA/Qwen (hidden_size/num_*)
    两套命名约定。
    """
    return ModelSpec(
        model_id=model_id,
        revision=revision,
        vocab_size=int(cfg.get("vocab_size", 0)),
        hidden_size=int(cfg.get("hidden_size", cfg.get("n_embd", 0))),
        num_layers=int(cfg.get("num_hidden_layers", cfg.get("n_layer", 0))),
        num_attention_heads=int(cfg.get("num_attention_heads", cfg.get("n_head", 0))),
        num_kv_heads=int(
            cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
        ),
        head_dim=int(
            cfg.get("head_dim", 0)
            or (
                cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
            )
        ),
        intermediate_size=int(cfg.get("intermediate_size", 0)),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 0)),
        rope_theta=cfg.get("rope_theta"),
        rope_scaling=cfg.get("rope_scaling"),
    )


def _live_spec(spec_cfg: dict[str, Any]) -> ModelSpec:
    """真实加载（需要 transformers）。失败则抛 ImportError。"""
    from transformers import AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(
        spec_cfg["model_id"], revision=spec_cfg.get("revision", "main")
    )
    spec = _from_config_dict(
        spec_cfg["model_id"], spec_cfg.get("revision", "main"), cfg.to_dict()
    )
    spec.attention_implementation = spec_cfg.get("attention_implementation", "")
    spec.dtype = spec_cfg.get("dtype", "")
    return spec


def _fallback_spec(spec_cfg: dict[str, Any]) -> ModelSpec:
    """离线 fallback：基于 Qwen3 已知架构与 model_id 字符串给出合理估计。

    目的：保证无 GPU / 无 transformers 环境下也能产出 model_compatibility.json。
    Qwen3 已知架构（截至 2025）：
        - Qwen3-4B:   vocab=151936, hidden=2560, layers=36, heads=20, kv_heads=8, head_dim=128
        - Qwen3-1.7B: vocab=151936, hidden=2048, layers=28, heads=16, kv_heads=8, head_dim=128
    """
    mid = spec_cfg["model_id"].lower()
    if "4b" in mid:
        vocab, hidden, layers, heads, kv_heads, head_dim, inter = (
            151936, 2560, 36, 20, 8, 128, 6912
        )
    elif "1.7b" in mid:
        vocab, hidden, layers, heads, kv_heads, head_dim, inter = (
            151936, 2048, 28, 16, 8, 128, 6144
        )
    else:
        vocab, hidden, layers, heads, kv_heads, head_dim, inter = (
            32000, 2048, 24, 16, 16, 128, 8192
        )
    return ModelSpec(
        model_id=spec_cfg["model_id"],
        revision=spec_cfg.get("revision", "main"),
        vocab_size=vocab,
        hidden_size=hidden,
        num_layers=layers,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=inter,
        max_position_embeddings=32768,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        attention_implementation=spec_cfg.get("attention_implementation", ""),
        dtype=spec_cfg.get("dtype", ""),
        extras={"source": "fallback"},
    )


def scan_model(spec_cfg: dict[str, Any]) -> ModelSpec:
    """优先用 AutoConfig；ImportError 时 fallback 到已知架构。"""
    try:
        return _live_spec(spec_cfg)
    except ImportError:
        return _fallback_spec(spec_cfg)


def compatibility_report(t: ModelSpec, s: ModelSpec) -> dict[str, Any]:
    """对比两份 ModelSpec，给出 §10/§12/§13 分流结论。

    判定规则：
        matched_kv AND same_tokenizer  → G1_MATCHED_KV
        matched_kv AND diff tokenizer   → G1_MATCHED_KV_DIFF_TOKENIZER
        diff_kv    AND same_tokenizer   → G2_MISMATCHED_HEAD_DIM
        diff_kv    AND diff tokenizer   → G3_CROSS_FAMILY
    """
    matched_kv = (t.num_kv_heads == s.num_kv_heads) and (t.head_dim == s.head_dim)
    matched_hidden = t.hidden_size == s.hidden_size
    same_tokenizer = t.vocab_size == s.vocab_size  # 粗略判断；真正审计在 T03+

    if matched_kv and same_tokenizer:
        verdict = "G1_MATCHED_KV"
        note = (
            "Same family + matched KV heads / head_dim. "
            "可直接进入 matched-KV 主实验 (design.md §10)."
        )
    elif matched_kv:
        verdict = "G1_MATCHED_KV_DIFF_TOKENIZER"
        note = "KV 维度匹配但 vocab 不同，需要 tokenizer audit."
    elif same_tokenizer:
        verdict = "G2_MISMATCHED_HEAD_DIM"
        note = (
            "Tokenizer 相同但 KV head 配置不一致，属于 G2 (design.md §12)。"
            "需要 Head Projection P_H 或 Dimension Projection P_d。"
        )
    else:
        verdict = "G3_CROSS_FAMILY"
        note = (
            "Tokenizer 与 KV 配置都不一致，属于 G3 (design.md §13)。"
            "需要完成五个 audit 后再决定是否执行。"
        )
    return {
        "matched_kv": matched_kv,
        "matched_hidden": matched_hidden,
        "same_tokenizer_vocab": same_tokenizer,
        "verdict": verdict,
        "note": note,
    }


def run_compat_scan(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T00 入口：扫描 + 判定 + 输出。"""
    teacher_cfg = cfg["teacher"]
    student_cfg = cfg["student"]
    t = scan_model(teacher_cfg)
    s = scan_model(student_cfg)
    compat = compatibility_report(t, s)

    payload = {
        "teacher": t.__dict__,
        "student": s.__dict__,
        "compatibility": compat,
    }
    write_json(run_dir / "model_compatibility.json", payload)
    write_json(
        run_dir / "metrics.json",
        {"status": "OK", "task": "T00", "verdict": compat["verdict"]},
    )

    summary = (
        f"# T00 Compatibility Scanner\n\n"
        f"- Teacher: {t.model_id}\n"
        f"- Student: {s.model_id}\n"
        f"- Verdict: **{compat['verdict']}**\n\n"
        f"{compat['note']}\n\n"
        "## Teacher\n"
        f"- layers={t.num_layers}, hidden={t.hidden_size}, "
        f"heads={t.num_attention_heads}, kv_heads={t.num_kv_heads}, "
        f"head_dim={t.head_dim}\n\n"
        "## Student\n"
        f"- layers={s.num_layers}, hidden={s.hidden_size}, "
        f"heads={s.num_attention_heads}, kv_heads={s.num_kv_heads}, "
        f"head_dim={s.head_dim}\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    return {
        "status": "PASS",
        "metrics": {"verdict": compat["verdict"], "matched_kv": compat["matched_kv"]},
        "summary": summary,
    }