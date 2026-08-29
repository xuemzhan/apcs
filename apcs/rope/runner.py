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

RoPE 原理（为什么有 theta / pos / 频率）：
    - 旋转向量：把 head_dim 维向量切成 d/2 个二维平面 (x_{2i}, x_{2i+1})，
      每个平面内按角度 pos·inv_freq[i] 做旋转（2×2 正交旋转矩阵）。
    - pos（位置索引）：同一平面不同 token 的旋转角不同 → 相对位置编码；
      两个 token 向量内积只依赖相对距离 Δpos（旋转角之差），且旋转不改变模长。
    - inv_freq（频率表）：d/2 个平面用递减频率 θ^(-2i/d)。
      低频（i 小）旋转慢、波长长 → 编码粗粒度/长距离依赖；
      高频（i 大）旋转快、波长短 → 编码细粒度/短距离依赖。
    - theta（基数）：整体缩放频率尺度。θ 越大频率越小、旋转越慢，波长更长，
      更适配长上下文（Qwen3 用 1e6；原版 RoPE 用 1e4）。
      约束：若最大平面旋转角在上下文范围内不超过 π，各相对位置可唯一区分。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..io.runs import write_json


def _rope_pairs(head_dim: int, theta: float = 10000.0) -> np.ndarray:
    """每个 pair 维度 i 对应的频率倒数 inv_freq[i]。

    RoPE 标准形式：inv_freq[i] = theta^(-2i/d)，i = 0..(d/2)-1。
    返回 shape=(d/2,)。

    说明：
        - i 遍历"偶数下标对"（每对 (2i, 2i+1) 共用一个频率），共 d/2 个。
        - i 越大指数越负 → inv_freq 越小 → 旋转越慢 → 波长越长，
          形成"低频管远、高频管近"的多尺度位置编码。
        - theta 是旋转编码的基数（周期尺度）：默认 10000（RoPE/LLaMA 标准），
          Qwen3 放大到 1e6 以支持更长上下文（§23）。
    """
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return theta ** (-i / head_dim)


def _locate_positions_axis(
    orig_shape: tuple[int, ...], positions: np.ndarray
) -> int:
    """定位 positions (S,) 对齐的序列维（◆ apply_rope/de_rope 修复的公共 helper）。

    规则（KV 张量约定：dim0=层、dim1=序列）：
        - ndim ≤ 3（(S,D)/(S,H,D)）：优先 dim0；
        - ndim ≥ 4（(L,S,H,D)）：优先 dim1 —— L==S 时按上述约定取 dim1；
        - 首选轴不匹配时在其余前导轴中搜索；全部不匹配 → raise。
    """
    n_pos = int(np.asarray(positions).reshape(-1).shape[0])
    preferred = 1 if len(orig_shape) >= 4 else 0
    if orig_shape[preferred] == n_pos:
        return preferred
    for axis, sz in enumerate(orig_shape[:-1]):
        if sz == n_pos:
            return axis
    raise ValueError(
        f"positions 长度 {n_pos} 与输入形状 {orig_shape} 的任何前导维都不匹配"
    )


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
    # 把最后一维重排为 (half, 2)：x_pairs[..., i, 0]=x_{2i}, [..., i, 1]=x_{2i+1}
    # 即把 d 维向量拆成 d/2 个二维平面，每个平面内做独立旋转。
    x_pairs = x.reshape(*orig_shape[:-1], half, 2)

    # positions: (S,) —— 对齐到序列维。
    # ◆ 修复：此前硬编码对齐 dim0，4D 输入 (L, S, H, D) 时 S 在 dim1，
    #   广播形状不匹配直接崩溃。现自动定位长度匹配的序列维（优先 dim0）。
    pos_axis = _locate_positions_axis(orig_shape, positions)
    shape = [1] * x.ndim  # pos ndim == x.ndim，末维 1 与 inv_freq 广播
    shape[pos_axis] = positions.shape[0]
    pos = positions.reshape(*shape)  # 序列维为 S，其余为 1
    pos = np.broadcast_to(pos, (*orig_shape[:-1], half))
    # 旋转角 = 位置 pos × 频率 inv_freq：同一 token 的第 i 个平面旋转 pos·inv_freq[i] 弧度
    angles = pos * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)

    # 2×2 旋转矩阵 R(θ)=[[cosθ,-sinθ],[sinθ,cosθ]] 作用于 (x0,x1)：
    # y0 = x0·cosθ - x1·sinθ ; y1 = x0·sinθ + x1·cosθ（保模长的正交变换）
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
    # 与 apply_rope 相同的拆平面/广播/角度计算（shape 约定一致）
    x_pairs = x.reshape(*orig_shape[:-1], half, 2)
    # ◆ 同 apply_rope 的序列维定位修复（4D (L,S,H,D) 场景）
    pos_axis = _locate_positions_axis(orig_shape, positions)
    shape = [1] * x.ndim
    shape[pos_axis] = positions.shape[0]
    pos = positions.reshape(*shape)
    pos = np.broadcast_to(pos, (*orig_shape[:-1], half))
    angles = pos * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    # 逆旋转 = 用负角度旋转：cos(-θ)=cosθ, sin(-θ)=-sinθ，
    # 即 R(-θ)=[[cosθ,sinθ],[-sinθ,cosθ]]：
    # y0 = x0·cosθ + x1·sinθ ; y1 = -x0·sinθ + x1·cosθ
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
    theta = 1_000_000.0  # Qwen3 的 RoPE 基数（§23），放大到 1e6 以支持更长上下文
    n_tokens = 128
    n_layers, n_heads = 3, 3  # §30 要求至少 3 Layers × 3 Heads × 128 Tokens 抽样

    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            # 每个 (layer, head) 独立采样随机向量序列：x shape (S, head_dim)
            x = rng.standard_normal((n_tokens, head_dim)).astype(np.float64)
            # 位置即 token 下标 0..S-1（取浮点以与角度乘法匹配）
            pos = np.arange(n_tokens, dtype=np.float64)
            # apply → de 应逐位还原（正交变换，误差仅来自浮点精度）
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
        # §30 Gate 判定：平均 cosine > 0.9999 才允许 PASS
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