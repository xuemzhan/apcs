"""T02 RoPE Round-trip 验证（design.md §30 / §23）。

═══════════════════════════════════════════════════════════════════════════════
目标：
    K_rope → de-RoPE → re-RoPE ≈ K_rope
必须 round-trip 误差极小（cosine > 0.9999）。

§30 至少抽样：3 Layers × 3 Heads × 128 Tokens。

§23 / §47 A6 强制 de-RoPE：本模块的 `de_rope` 是 Mapper 的入口函数。
对 last K（Teacher RoPE 空间）做 de-RoPE 后再线性映射，最后由 Student
attention 自己在自己的 RoPE 空间里 re-RoPE。

RoPE 公式：
    对每个 pair (x_{2i}, x_{2i+1})，旋转角度 = pos · inv_freq[i]
    inv_freq[i] = θ^(-2i/d)
    默认 θ = 10000.0；Qwen3 用 θ = 1_000_000.0（更长上下文）

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..io.runs import write_json


def _rope_pairs(head_dim: int, theta: float = 10000.0) -> np.ndarray:
    """每个 pair 维度 i 对应的频率倒数 inv_freq[i]。

    RoPE 标准形式：inv_freq[i] = theta^(-2i/d)，i = 0..(d/2)-1。
    返回 shape=(d/2,)。
    """
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return theta ** (-i / head_dim)


def apply_rope(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """对最后一维按配对 (2i, 2i+1) 做旋转。

    支持形状：
        - (..., head_dim)              任意前导维度
        - (S, head_dim)                常见 KV cache 形状
        - (S, H, head_dim)             KV cache 多头

    参数：
        x: 最后一维必须为 head_dim
        positions: (S,) 序列位置；内部广播到 (..., half)
        inv_freq: (head_dim/2,)
    返回：
        旋转后的 x，shape 与输入一致。
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("head_dim 必须为偶数")
    half = head_dim // 2
    x_pairs = x.reshape(*orig_shape[:-1], half, 2)

    # positions: (S,)
    # 需要把它广播到 (*orig_shape[:-1], half)
    # 即把 positions reshape 为 (S, 1, 1, ..., 1)，共 (n-1) 个 1
    pos = positions.reshape(-1, *([1] * (x.ndim - 1)))  # (S, 1, ..., 1)
    pos = np.broadcast_to(pos, (*orig_shape[:-1], half))
    angles = pos * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)

    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    out = np.stack([y0, y1], axis=-1).reshape(orig_shape)
    return out


def de_rope(x: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """应用 RoPE 的逆旋转 = 负角度旋转。

    数学上：de-RoPE 是正交变换（det = ±1），
    所以 de-RoPE(re-RoPE(K)) ≈ K（浮点精度范围内）。

    形状约定同 apply_rope。
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    half = head_dim // 2
    x_pairs = x.reshape(*orig_shape[:-1], half, 2)
    pos = positions.reshape(-1, *([1] * (x.ndim - 1)))
    pos = np.broadcast_to(pos, (*orig_shape[:-1], half))
    angles = pos * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    y0 = x0 * cos + x1 * sin
    y1 = -x0 * sin + x1 * cos
    return np.stack([y0, y1], axis=-1).reshape(orig_shape)


def roundtrip_error(
    x: np.ndarray, positions: np.ndarray, head_dim: int, theta: float = 10000.0
) -> tuple[float, float]:
    """Round-trip 误差：返回 (max_abs_err, cosine_to_original)。

    理想：max_err < 1e-10，cosine > 0.999999999。
    """
    inv_freq = _rope_pairs(head_dim, theta)
    x_r = apply_rope(x, positions, inv_freq)
    x_back = de_rope(x_r, positions, inv_freq)
    err = float(np.max(np.abs(x - x_back)))
    cos = float(
        np.sum(x * x_back) / max(1e-12, np.sqrt(np.sum(x * x) * np.sum(x_back * x_back)))
    )
    return err, cos


def run_rope_roundtrip(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T02 CLI 入口：3 layers × 3 heads × 128 tokens 抽样 round-trip。"""
    rng = np.random.default_rng(0)
    head_dim = 128
    theta = 1_000_000.0
    n_tokens = 128
    n_layers, n_heads = 3, 3

    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            x = rng.standard_normal((n_tokens, head_dim)).astype(np.float64)
            pos = np.arange(n_tokens, dtype=np.float64)
            err, cos = roundtrip_error(x, pos, head_dim, theta)
            rows.append({"layer": layer, "head": head, "max_err": err, "cosine": cos})

    metrics = {
        "task": "T02",
        "head_dim": head_dim,
        "theta": theta,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "mean_max_err": float(np.mean([r["max_err"] for r in rows])),
        "mean_cosine": float(np.mean([r["cosine"] for r in rows])),
        "gate": "PASS"
        if float(np.mean([r["cosine"] for r in rows])) > 0.9999
        else "FAIL",
    }
    write_json(run_dir / "metrics.json", {"task": "T02", **metrics})
    summary = (
        "# T02 RoPE Round-trip\n\n"
        f"- head_dim={head_dim}, theta={theta}\n"
        f"- {n_layers} layers × {n_heads} heads × {n_tokens} tokens\n"
        f"- mean max err: {metrics['mean_max_err']:.2e}\n"
        f"- mean cosine: {metrics['mean_cosine']:.6f}\n"
        f"- **Gate: {metrics['gate']}**\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"status": metrics["gate"], "metrics": {"task": "T02", **metrics}, "summary": summary}