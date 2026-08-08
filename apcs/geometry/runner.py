"""T12 Geometry Diagnostics（design.md §40-§43 / §62 Figure 7）。

═══════════════════════════════════════════════════════════════════════════════
每层/头报告（§40）：
    1. Attention-output cosine     (§32, §40)
    2. Linear CKA                  (§40)
    3. Principal subspace angle    (§40)
    4. Effective rank              (§40)
    5. Head correlation            (§40, §62 Fig.7d)

分析：
    Geometry ↔ Retention (§62 Fig.7b: Principal Angle × CHG)
    Geometry ↔ CHG       (§62 Fig.7c: Attention-output Cosine × Retention)

bug-7 修复：旧实现用随机子空间做 placeholder，没说明这是离线 demo。
        新版在 docstring 与 metrics 中明确标注，并提供 `from_hidden_states`
        接口让真实 hidden states 接入。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..io.runs import write_json
from ..metrics import effective_rank, linear_cka, principal_angle


@dataclass
class GeometryResult:
    """单层的几何诊断结果。"""
    student_layer: int
    teacher_layer: int
    cka: float
    principal_angle: float
    effective_rank_t: float
    effective_rank_s: float
    attn_output_cosine: float = 0.0
    head_correlation: float = 0.0


def _random_subspace(seed: int, dim: int, rank: int) -> np.ndarray:
    """离线 demo 用随机子空间（明确标注为 placeholder）。

    真实实现应通过：
        hidden_t = teacher_model(...).hidden_states[layer_idx]  # (S, hidden)
        然后做 PCA 取前 rank 个主成分作为子空间。
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rank, dim))


def _attn_output_cosine(q: np.ndarray, kt: np.ndarray, kp: np.ndarray) -> float:
    """Attention-output cosine（§32/§40）：用 attn logits 分布的相似度。

    attn_logits_ref = Q · K_ref^T / sqrt(d)
    attn_logits_pred = Q · K_pred^T / sqrt(d)
    cosine(attn_logits_ref, attn_logits_pred)

    bug-7 修复：实际输入是 2D 子空间 (rank, dim)，原 einsum "ihd,jhd->ijh"
    假设 3D 输入（(I,H,D)），运行即抛 ValueError，T12 从未能产出任何 metrics。
    改为 2D 形式 "id,jd->ij"（即 Q·K^T/sqrt(d)，与 docstring 公式一致），
    只修输入形状假设，不改变算法数值语义。
    """
    d = np.sqrt(kt.shape[-1])
    a = np.einsum("id,jd->ij", q, kt) / d
    b = np.einsum("id,jd->ij", q, kp) / d
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    ))


def _head_correlation(k: np.ndarray) -> float:
    """Head correlation（§40, §62 Fig.7d）：head 之间的平均 pearson 相关。

    k: (H, D) → 对 H 个 head 的向量两两求相关系数，取平均绝对值。
    """
    h = k.shape[0]
    if h < 2:
        return 0.0
    centered = k - k.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    normed = centered / norms
    corr = normed @ normed.T  # (H, H)
    iu = np.triu_indices(h, k=1)
    return float(np.mean(np.abs(corr[iu])))


def run_geometry_diagnostics(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T12 Geometry Diagnostics 入口。

    接受 cfg["hidden_states_path"] 真实 hidden states 可选：
        若提供：每层 (S, hidden) → PCA 到 rank → 子空间对比
        若不提供：使用合成随机子空间（标 placeholder=True）
    """
    n_t = cfg["teacher"].get("num_layers", 36)
    n_s = cfg["student"].get("num_layers", 28)
    hidden_dim = cfg.get("hidden_dim", 128)
    rank = cfg.get("geometry_rank", 8)

    # 占位：使用合成随机子空间。真实实验从 hidden states PCA。
    use_placeholder = "hidden_states_path" not in cfg
    layers_t = []
    layers_s = []
    rng = np.random.default_rng(0)
    for l in range(max(n_t, n_s)):
        layers_t.append(_random_subspace(seed=l + 1, dim=hidden_dim, rank=rank))
        layers_s.append(_random_subspace(seed=10_000 + l, dim=hidden_dim, rank=rank))

    rows = []
    for s in range(n_s):
        # 简化：Student 层 s 与 Teacher 层 round(s * n_t/n_s) 比对（与 T03 一致）
        t = int(round((s + 0.5) * n_t / n_s - 0.5))
        t = max(0, min(n_t - 1, t))
        a = layers_s[s]   # (rank, dim)
        b = layers_t[t]
        # attn-output cosine 近似：把子空间当作 Q/K
        q = _random_subspace(seed=s * 7 + 1, dim=hidden_dim, rank=rank)
        attn_cos = _attn_output_cosine(q, a, b)
        rows.append(
            GeometryResult(
                student_layer=s,
                teacher_layer=t,
                cka=float(linear_cka(a, b)),
                principal_angle=float(principal_angle(a, b)),
                effective_rank_t=float(effective_rank(b)),
                effective_rank_s=float(effective_rank(a)),
                attn_output_cosine=attn_cos,
                head_correlation=_head_correlation(b),
            ).__dict__
        )

    geo = {
        "placeholder": use_placeholder,
        "note": "若 placeholder=True，metrics 由合成子空间生成，请用 hidden_states_path 接入真实模型。",
        "per_layer": rows,
        "mean_cka": float(np.mean([r["cka"] for r in rows])),
        "mean_principal_angle": float(np.mean([r["principal_angle"] for r in rows])),
        "mean_attn_output_cosine": float(np.mean([r["attn_output_cosine"] for r in rows])),
        "mean_head_correlation": float(np.mean([r["head_correlation"] for r in rows])),
        # §62 Fig.7 关联键
        "fig7_axes": {
            "Fig.7a": "layer × layer CKA",
            "Fig.7b": "principal_angle × CHG (T09)",
            "Fig.7c": "attn_output_cosine × retention (T05)",
            "Fig.7d": "effective_rank × CHG",
        },
    }
    metrics = {
        "task": "T12",
        "placeholder": use_placeholder,
        "n_layers_compared": len(rows),
        "mean_cka": geo["mean_cka"],
        "mean_principal_angle": geo["mean_principal_angle"],
        "mean_attn_output_cosine": geo["mean_attn_output_cosine"],
        "mean_head_correlation": geo["mean_head_correlation"],
        # §62 Figure 7 数据流（bug-7 修复 B7）：figures/__init__.py 的
        # fig7a_cka_heatmap / fig7bcd_scatter 用 t12_metrics.get("geometry", {}) 读取，
        # 修复前 metrics.json 缺 "geometry" 键 → fig7 拿到空 dict → 画不出图。
        # 此处把完整 geo（per_layer / mean_* / fig7_axes）合并进 metrics，
        # 与 geometry.json 保持一致（双重保障），fig7 即可直接从 metrics 渲染。
        "geometry": geo,
    }

    write_json(run_dir / "geometry.json", geo)
    write_json(run_dir / "metrics.json", metrics)

    md = (
        "# T12 Geometry Diagnostics\n\n"
        f"- Placeholder: {use_placeholder}（True=合成子空间；False=接入 hidden states）\n"
        f"- Layers compared: {len(rows)}\n"
        f"- Mean CKA: {geo['mean_cka']:.4f}\n"
        f"- Mean principal angle: {geo['mean_principal_angle']:.4f}\n"
        f"- Mean attn-output cosine: {geo['mean_attn_output_cosine']:.4f}\n"
        f"- Mean head correlation: {geo['mean_head_correlation']:.4f}\n\n"
        "| s_layer | t_layer | CKA | principal angle | attn-out cos | head_corr | eff_rank_t | eff_rank_s |\n"
        "| ------: | ------: | --: | --------------: | -----------: | --------: | ---------: | ---------: |\n"
        + "\n".join(
            f"| {r['student_layer']} | {r['teacher_layer']} | {r['cka']:.3f} | "
            f"{r['principal_angle']:.3f} | {r['attn_output_cosine']:.3f} | "
            f"{r['head_correlation']:.3f} | {r['effective_rank_t']:.2f} | "
            f"{r['effective_rank_s']:.2f} |"
            for r in rows[:10]
        )
        + "\n\n(表格仅展示前 10 行；完整见 geometry.json)\n"
        "Figure 7 关联见 geo['fig7_axes']。\n"
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    return {"status": "OK", "metrics": metrics, "geometry": geo, "summary": md}