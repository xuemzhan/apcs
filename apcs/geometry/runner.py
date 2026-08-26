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

几何视角（流形 / 投影）：
    - 每层 hidden states 可看作高维空间中一条"表示流形"的采样点；
      不同层对应不同深度的表示流形切片。
    - 投影：对每层表示做 PCA 取前 rank 个主成分 → 得到该层的 rank 维主子空间
      （Grassmann 流形上的一个点）。
    - principal angle：两个子空间在主方向上的夹角（近似测地距离），
      越小代表 Teacher/Student 在该层的几何越对齐。
    - linear CKA：对 scale 不敏感的子空间整体相似度（§40）。
    - effective rank：表示的内在维度（信息量 / 流形"厚度"代理）。
    - attn-output cosine：K 被替换后 attention 分布保持程度（§32/§40）。
    几何对齐 → Retention / CHG 的机制解释（§62 Fig.7）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io.runs import write_json
from ..metrics import effective_rank, linear_cka, principal_angle


@dataclass
class GeometryResult:
    """单层的几何诊断结果。

    字段语义（§40）：
        student_layer / teacher_layer : 被比对的 Student / Teacher 层索引
        cka                 : 线性 CKA（子空间相似度，对 scale 不敏感）
        principal_angle     : 主子空间夹角（弧度，越小越对齐）
        effective_rank_t/s  : Teacher / Student 表示的有效秩（内在维度）
        attn_output_cosine  : Q·K^T 注意力 logits 分布的余弦（§32/§40）
        head_correlation    : head 间平均相关（§62 Fig.7d）
    """
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
        然后做 PCA 取前 rank 个主成分作为子空间
        （即把表示流形投影到其 rank 维主子空间）。
    返回 (rank, dim) 的子空间基向量。
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
    # 注意力 logits：Q · K^T / sqrt(d)，与 Transformer attention 公式一致
    a = np.einsum("id,jd->ij", q, kt) / d
    b = np.einsum("id,jd->ij", q, kp) / d
    # 展平后求余弦：度量"参考 K"与"替换 K"对同一 Q 产生的注意力分布差异
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    ))


def _head_correlation(k: np.ndarray) -> float:
    """Head correlation（§40, §62 Fig.7d）：head 之间的平均 pearson 相关。

    k: (H, D) → 对 H 个 head 的向量两两求相关系数，取平均绝对值。
    """
    h = k.shape[0]
    if h < 2:
        return 0.0   # 只有 1 个 head，无两两组合
    centered = k - k.mean(axis=1, keepdims=True)   # 每 head 减去自身均值（Pearson 相关）
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    normed = centered / norms
    corr = normed @ normed.T  # (H, H) 相关矩阵
    iu = np.triu_indices(h, k=1)   # 严格上三角，排除对角线 self-correlation
    return float(np.mean(np.abs(corr[iu])))   # 两两相关取平均绝对值


def run_geometry_diagnostics(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T12 Geometry Diagnostics 入口。

    接受 cfg["hidden_states_path"] 真实 hidden states 可选：
        若提供：每层 (S, hidden) → PCA 到 rank → 子空间对比
        若不提供：使用合成随机子空间（标 placeholder=True）

    几何模型（§40-§43）：
        - 每层表示被投影到其 rank 维主子空间（Grassmann 流形上的点）。
        - principal_angle 度量两个子空间主方向夹角（近似测地距离）；
        - linear CKA 度量子空间整体相似度；
        - effective rank 度量表示的内在维度。
        这些几何量用于解释 Retention / CHG 的机制（§62 Fig.7）。
    """
    n_t = cfg["teacher"].get("num_layers", 36)
    n_s = cfg["student"].get("num_layers", 28)
    hidden_dim = cfg.get("hidden_dim", 128)
    rank = cfg.get("geometry_rank", 8)   # 投影子空间的维度（主成分个数）

    hidden_states_path = cfg.get("hidden_states_path")
    use_placeholder = hidden_states_path is None
    repr_t: list[np.ndarray] = []
    repr_s: list[np.ndarray] = []
    if use_placeholder:
        for l in range(max(n_t, n_s)):
            # placeholder 也使用“tokens × hidden”表示，与真实路径语义一致。
            repr_t.append(_random_subspace(seed=l + 1, dim=hidden_dim, rank=max(32, rank)))
            repr_s.append(_random_subspace(seed=10_000 + l, dim=hidden_dim, rank=max(32, rank)))
    else:
        path = Path(str(hidden_states_path))
        if not path.exists():
            raise FileNotFoundError(f"hidden_states_path 不存在: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if "teacher" not in payload or "student" not in payload:
                raise ValueError("hidden-states NPZ 必须包含 teacher 和 student 数组")
            hidden_t = np.asarray(payload["teacher"])
            hidden_s = np.asarray(payload["student"])
        if hidden_t.ndim != 3 or hidden_s.ndim != 3:
            raise ValueError("teacher/student hidden states 必须是 (layers, tokens, hidden)")
        if hidden_t.shape[2] != hidden_s.shape[2]:
            raise ValueError("T12 当前要求 Teacher/Student hidden 维度一致")
        if hidden_t.shape[1] != hidden_s.shape[1]:
            raise ValueError("T12 CKA 要求 Teacher/Student 使用对齐的 token 样本")
        n_t, n_s = int(hidden_t.shape[0]), int(hidden_s.shape[0])
        hidden_dim = int(hidden_t.shape[2])
        repr_t = [hidden_t[l] for l in range(n_t)]
        repr_s = [hidden_s[l] for l in range(n_s)]

    def pca_basis(x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[: min(rank, vt.shape[0])]

    rows = []
    for s in range(n_s):
        # 简化：Student 层 s 与 Teacher 层 round(s * n_t/n_s) 比对（与 T03 一致）
        t = int(round((s + 0.5) * n_t / n_s - 0.5))
        t = max(0, min(n_t - 1, t))   # 夹取到合法 Teacher 层范围
        a_raw = np.asarray(repr_s[s], dtype=np.float64)
        b_raw = np.asarray(repr_t[t], dtype=np.float64)
        a_basis = pca_basis(a_raw)
        b_basis = pca_basis(b_raw)
        # hidden-state artifact 不含真实 Q/K/V；此项仅保留为显式 proxy。
        q = _random_subspace(seed=s * 7 + 1, dim=hidden_dim, rank=len(a_basis))
        attn_cos = _attn_output_cosine(q, a_basis, b_basis)
        rows.append(
            GeometryResult(
                student_layer=s,
                teacher_layer=t,
                cka=float(linear_cka(a_raw, b_raw)),
                principal_angle=float(principal_angle(a_basis, b_basis)),
                effective_rank_t=float(effective_rank(b_raw)),
                effective_rank_s=float(effective_rank(a_raw)),
                attn_output_cosine=attn_cos,                     # attention 分布保持度
                head_correlation=_head_correlation(b_basis),
            ).__dict__
        )

    geo = {
        "placeholder": use_placeholder,
        "evidence_grade": "placeholder" if use_placeholder else "measured_representation",
        "note": "若 placeholder=True，metrics 由合成子空间生成，请用 hidden_states_path 接入真实模型。",
        "metric_semantics": {
            "cka": "centered token hidden states",
            "principal_angle": "PCA row subspaces",
            "effective_rank": "centered token hidden states",
            "attn_output_cosine": "random-Q subspace proxy; real Q/K/V required for task evidence",
            "head_correlation": "PCA-component correlation; not literal attention heads",
        },
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
        "evidence_grade": "placeholder" if use_placeholder else "measured_representation",
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
