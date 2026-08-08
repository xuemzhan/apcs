"""T03 Layer Alignment（design.md §21 / §31 / A7 消融）。

═══════════════════════════════════════════════════════════════════════════════
§21 必须比较四种 Layer Selection 策略（A7 消融）：
    1. Proportional         按层数比例均摊
    2. Last-layer           每个 Student 层只取最末 k 个 Teacher 层
    3. Data-driven top-k    基于相似度矩阵 top-k
    4. Geometry-aware top-k data-driven + 邻近层平滑

§21 强制约束：sum_i w_{l,i,h} = 1（权重和为 1）

§31 输出：
    - Layer Similarity Matrix
    - Top-k Mapping（每个 Student 层对应的 Teacher 层）
    - Attention-output Similarity（attn-output cosine）

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..metrics import cosine, linear_cka
from ..io.runs import write_json


def proportional_mapping(n_t: int, n_s: int) -> list[list[int]]:
    """按层数比例把 Teacher 层均摊给 Student 层。

    例：n_t=36, n_s=28 → Student layer 0..27 各取 ⌊36/28⌋ 或 ⌈36/28⌉ 个
    Teacher 层。
    """
    mapping: list[list[int]] = []
    for s in range(n_s):
        start = int(round(s * n_t / n_s))
        end = int(round((s + 1) * n_t / n_s))
        end = max(end, start + 1)
        end = min(end, n_t)
        mapping.append(list(range(start, end)))
    return mapping


def last_layer_mapping(n_t: int, n_s: int, k: int = 2) -> list[list[int]]:
    """每个 Student 层只取最末 k 个 Teacher 层（baseline 退化版）。"""
    tail = list(range(max(0, n_t - k), n_t))
    return [list(tail) for _ in range(n_s)]


def data_driven_topk(
    sim_matrix: np.ndarray, k: int = 2
) -> list[list[int]]:
    """基于相似度矩阵，每个 Student 层取 top-k Teacher 层。"""
    mapping: list[list[int]] = []
    for s in range(sim_matrix.shape[0]):
        idx = np.argsort(-sim_matrix[s])[:k]
        mapping.append(sorted(idx.tolist()))
    return mapping


def geometry_aware_topk(
    sim_matrix: np.ndarray, k: int = 2, alpha: float = 0.5
) -> list[list[int]]:
    """Geometry-aware：data-driven top-k + 邻近层平滑 (避免断层)。

    alpha 越大越平滑。对深层的相似度结构更有鲁棒性。
    """
    n_s = sim_matrix.shape[0]
    smoothed = sim_matrix.copy()
    if n_s > 1:
        for s in range(n_s):
            lo = max(0, s - 1)
            hi = min(n_s, s + 2)
            smoothed[s] = (1 - alpha) * sim_matrix[s] + alpha * sim_matrix[lo:hi].mean(0)
    return data_driven_topk(smoothed, k=k)


def synthetic_similarity(n_t: int, n_s: int, seed: int = 0) -> np.ndarray:
    """无真实 hidden states 时构造一个"对角带状"相似度矩阵做演示。

    真实实验应通过：
        sim[s, t] = cos(hidden_s_layer_s.mean(0), hidden_t_layer_t.mean(0))
    """
    rng = np.random.default_rng(seed)
    sim = rng.uniform(0, 0.3, size=(n_s, n_t))
    for s in range(n_s):
        center = (s + 0.5) * n_t / n_s
        for t in range(n_t):
            d = abs(t - center)
            sim[s, t] += float(np.exp(-d * d / (n_t / n_s) ** 2))
    return sim


def run_layer_alignment(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T03 入口：构造 4 种 mapping 策略并写出 layer_mapping.json。"""
    n_t = cfg["teacher"].get("num_layers", 36)
    n_s = cfg["student"].get("num_layers", 28)
    sim = synthetic_similarity(n_t, n_s)

    mappings = {
        "proportional": proportional_mapping(n_t, n_s),
        "last_layer": last_layer_mapping(n_t, n_s),
        "data_driven_topk": data_driven_topk(sim, k=2),
        "geometry_aware_topk": geometry_aware_topk(sim, k=2),
    }

    metrics = {
        "task": "T03",
        "n_teacher_layers": n_t,
        "n_student_layers": n_s,
        "strategy": list(mappings.keys()),
        "mapping_size_avg": {
            k: float(np.mean([len(v) for v in mp])) for k, mp in mappings.items()
        },
        "selected": "proportional",
        "note": "offline demo 用对角带状合成相似度矩阵；真实实验需 attn-output cosine。",
    }
    write_json(run_dir / "layer_mapping.json", mappings)
    write_json(run_dir / "metrics.json", metrics)
    summary = (
        "# T03 Layer Alignment\n\n"
        f"- Teacher layers: {n_t}\n"
        f"- Student layers: {n_s}\n"
        f"- Strategies: {list(mappings.keys())}\n"
        "- 默认 `proportional`；其余用于 A7 消融 (design.md §47)。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"status": "OK", "metrics": metrics, "summary": summary}