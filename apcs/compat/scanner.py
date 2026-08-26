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

T00 溯源防串跑：scanner 输出新增 revision / rope_theta / tokenizer_hash，
    tokenizer_hash 基于 HF 本地缓存 tokenizer_config.json 的 SHA-256，
    离线环境（无缓存）返回 "unavailable:<reason>" 而不崩溃。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..io import load_config  # noqa: F401  (re-export for convenience)
from ..io.runs import write_json


@dataclass
class ModelSpec:
    """§28 模型规格字段：从 AutoConfig（或 fallback）抽取的架构快照。

    关键字段的判定用途：
        - num_kv_heads / head_dim / hidden_size → KV 维度匹配判定（G1/G2）
        - vocab_size → tokenizer 是否一致的粗略判定（同 vocab 视为同词表）
        - num_layers → T03 Layer Alignment 的层映射依据（L_t / L_s）
        - revision → 防串跑：确保扫描时使用的模型版本与预期一致
        - rope_theta → RoPE 频率参数（T02 需要读取）
        - tokenizer_hash → 溯源：SHA-256 哈希 tokenizer_config.json，
          防止同名模型不同 tokenizer 版本导致串跑
    """
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
    tokenizer_hash: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _tokenizer_hash(model_id: str) -> str:
    """T00 溯源：计算 HF 本地缓存中 tokenizer_config.json 的 SHA-256 哈希。

    缓存路径：~/.cache/huggingface/hub/models--{org}--{model}/snapshots/<snap>/tokenizer_config.json
    离线环境（无缓存）返回 "unavailable:<reason>" 而不崩溃。

    为什么要 hash 而不是直接比较 tokenizer_name：同名模型不同版本
    可能更换 tokenizer（vocab 扩充 / special tokens 变化），
    SHA-256 能精确区分 tokenizer_config.json 的任何字节差异。
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    # HF 缓存目录命名规则："org/model" → "models--org--model"
    cache_model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = cache_model_dir / "snapshots"

    if not snapshots_dir.exists():
        return f"unavailable:HF cache not found for {model_id}"

    # 遍历所有 snapshot 目录，取第一个包含 tokenizer_config.json 的
    for snap in sorted(snapshots_dir.iterdir()):
        tok_config = snap / "tokenizer_config.json"
        if tok_config.exists():
            data = tok_config.read_bytes()
            return hashlib.sha256(data).hexdigest()

    return f"unavailable:no tokenizer_config.json in any snapshot for {model_id}"


def _from_config_dict(model_id: str, revision: str, cfg: dict[str, Any]) -> ModelSpec:
    """从 AutoConfig 加载得到的 cfg 对象中抽取字段。

    兼容 GPT-2 (n_embd/n_layer/n_head) 与 LLaMA/Qwen (hidden_size/num_*)
    两套命名约定。
    """
    return ModelSpec(
        model_id=model_id,
        revision=revision,
        # LLaMA/Qwen 命名：hidden_size；GPT-2 命名：n_embd（field 级 fallback 链）
        vocab_size=int(cfg.get("vocab_size", 0)),
        hidden_size=int(cfg.get("hidden_size", cfg.get("n_embd", 0))),
        num_layers=int(cfg.get("num_hidden_layers", cfg.get("n_layer", 0))),
        num_attention_heads=int(cfg.get("num_attention_heads", cfg.get("n_head", 0))),
        # num_key_value_heads 缺失（非 GQA 模型）时退化为 num_attention_heads
        num_kv_heads=int(
            cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
        ),
        # head_dim 未显式给出时用 hidden_size // num_attention_heads 推算
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


def _hub_reachable(timeout: float = 1.0) -> bool:
    """模型 hub 是否可达的快速探测（offline 开发机/CI 无需等待连接超时）。

    优先探测 modelscope.cn（本项目模型源）；失败再退回 huggingface.co。
    仅做一次轻量 TCP 握手（不下载任何内容），失败返回 False。
    探测时间受 `timeout` 约束（默认 1s），确保离线环境下
    scan_model 的 fallback 路径不被 `from_pretrained` 的默认
    连接超时（~10s）拖慢整套测试。
    """
    import socket

    for host in ("modelscope.cn", "huggingface.co"):
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def _live_spec(spec_cfg: dict[str, Any]) -> ModelSpec:
    """真实加载（需要 transformers）。失败则抛 ImportError/OSError。

    ImportError 是本函数的"信号出口"之一：scan_model 捕获它切换到 fallback；
    另一个是 OSError（网络不可达 / hub 离线）。其它异常继续向外抛，
    避免静默产出错误估计。

    模型源：优先 modelscope.cn（`model_id` 如 "Qwen/Qwen3-4B" 可直接用），
    其 AutoConfig API 与 HuggingFace 对齐；不可达时退回 huggingface.co。
    """
    # offline 快速探测：hub 不可达时直接抛 OSError → scan_model 走 fallback，
    # 不等待 from_pretrained 的默认 ~10s 连接超时（T00 与 CLI 测试会显著变快）。
    if not _hub_reachable():
        raise OSError(
            f"模型 hub 不可达（offline）：无法加载 {spec_cfg['model_id']}，走 fallback"
        )

    revision = spec_cfg.get("revision", "main")
    # 第一优先：modelscope.cn（本项目模型源，见 README 快速开始）
    try:
        from modelscope import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(spec_cfg["model_id"], revision=revision)
    except Exception:  # noqa: BLE001  modelscope 缺失 / 模型不在其上
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(spec_cfg["model_id"], revision=revision)
    spec = _from_config_dict(spec_cfg["model_id"], revision, cfg.to_dict())
    # attention_implementation / dtype 不在 AutoConfig 中，从 yaml 配置补填
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
    # fallback 检查矩阵（按 model_id 子串匹配，仅覆盖已知 Qwen3 尺寸）：
    #   4B   → 36 层 / 20 heads / 8 kv_heads / head_dim 128
    #   1.7B → 28 层 / 16 heads / 8 kv_heads / head_dim 128
    #   其它 → 通用 LLaMA 风格默认值（保守估计，产物标注 source="fallback"）
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


def _hf_reachable(timeout: float = 1.0) -> bool:
    """HF hub 是否可达的快速探测（offline 开发机/CI 无需等待连接超时）。

    仅做一次轻量 TCP 握手（不下载任何内容）；失败返回 False。
    探测时间受 `timeout` 约束（默认 1s），确保离线环境下
    scan_model 的 fallback 路径不被 `from_pretrained` 的默认
    连接超时（~10s）拖慢整套测试。
    """
    return _hub_reachable(timeout)


def scan_model(spec_cfg: dict[str, Any]) -> ModelSpec:
    """优先用 AutoConfig；ImportError/OSError 时 fallback 到已知架构。

    策略分层：真实读取 > 硬编码估计。fallback 产物带
    extras={"source": "fallback"} 标记，消费方可据此识别数据来源。

    为什么捕获 OSError：装有 transformers 但 HF 网络不可达（离线开发机）
    时 `AutoConfig.from_pretrained` 抛 OSError（ConnectionError/HTTPError），
    应同样回退到已知架构估计，而不是让 T00 崩溃。离线时先经 _hf_reachable
    快速探测，避免每次等待 ~10s 连接超时。

    T00 溯源防串跑：扫描完成后计算 tokenizer_hash（SHA-256），
    写入 ModelSpec.tokenizer_hash 以供后续任务防串跑校验。
    """
    try:
        spec = _live_spec(spec_cfg)
    except (ImportError, OSError):
        spec = _fallback_spec(spec_cfg)
    # T00 溯源：计算 tokenizer_config.json 的 SHA-256 哈希
    # 离线环境无 HF 缓存时返回 "unavailable:<reason>"，不崩溃
    spec.tokenizer_hash = _tokenizer_hash(spec_cfg["model_id"])
    return spec


def compatibility_report(t: ModelSpec, s: ModelSpec) -> dict[str, Any]:
    """对比两份 ModelSpec，给出 §10/§12/§13 分流结论。

    判定规则：
        matched_kv AND same_tokenizer  → G1_MATCHED_KV
        matched_kv AND diff tokenizer   → G1_MATCHED_KV_DIFF_TOKENIZER
        diff_kv    AND same_tokenizer   → G2_MISMATCHED_HEAD_DIM
        diff_kv    AND diff tokenizer   → G3_CROSS_FAMILY
    """
    # 检查矩阵：matched_kv × same_tokenizer 两个布尔量组合出 4 个 verdict。
    #   matched_kv      = kv_heads 与 head_dim 同时一致（G1 的硬条件）
    #   same_tokenizer  ≈ vocab_size 相同（粗略判断；真正审计在 T03+）
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
    """T00 入口：扫描 + 判定 + 输出（CLI 分发到 run_compat_scan(cfg, run_dir)）。

    扫描步骤（与 §28 对齐）：
        1. scan_model 分别读取 teacher / student 架构（AutoConfig 优先，
           transformers 缺失时用 fallback 估计）；
        2. compatibility_report 对比两份 ModelSpec，得 G1/G2/G3 verdict；
        3. 落盘 model_compatibility.json（两端 spec + verdict + 溯源字段）、
           metrics.json（status=OK + verdict）、summary.md（人类可读摘要）。
    """
    teacher_cfg = cfg["teacher"]
    student_cfg = cfg["student"]
    # 步骤 1：分别扫描两端模型架构（含 tokenizer_hash 溯源）
    t = scan_model(teacher_cfg)
    s = scan_model(student_cfg)
    # 步骤 2：判定兼容性分流（G1/G2/G3）
    compat = compatibility_report(t, s)

    # 步骤 3a：model_compatibility.json = 两端完整 spec + 兼容性判定
    # __dict__ 包含 revision / rope_theta / tokenizer_hash 等溯源字段
    payload = {
        "teacher": t.__dict__,
        "student": s.__dict__,
        "compatibility": compat,
    }
    write_json(run_dir / "model_compatibility.json", payload)
    # 步骤 3b：metrics.json —— CLI 据此取 status/verdict（T13 聚合也读 verdict）
    write_json(
        run_dir / "metrics.json",
        {"status": "OK", "task": "T00", "verdict": compat["verdict"]},
    )

    summary = (
        f"# T00 Compatibility Scanner\n\n"
        f"- Teacher: {t.model_id} (revision={t.revision})\n"
        f"- Student: {s.model_id} (revision={s.revision})\n"
        f"- Verdict: **{compat['verdict']}**\n\n"
        f"{compat['note']}\n\n"
        "## Teacher\n"
        f"- layers={t.num_layers}, hidden={t.hidden_size}, "
        f"heads={t.num_attention_heads}, kv_heads={t.num_kv_heads}, "
        f"head_dim={t.head_dim}, rope_theta={t.rope_theta}\n\n"
        "## Student\n"
        f"- layers={s.num_layers}, hidden={s.hidden_size}, "
        f"heads={s.num_attention_heads}, kv_heads={s.num_kv_heads}, "
        f"head_dim={s.head_dim}, rope_theta={s.rope_theta}\n"
    )
    # 步骤 3c：summary.md 人类可读摘要（verdict + 两端关键维度 + 溯源字段）
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")

    return {
        "status": "PASS",
        "metrics": {"verdict": compat["verdict"], "matched_kv": compat["matched_kv"]},
        "summary": summary,
    }
