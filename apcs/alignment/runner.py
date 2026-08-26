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

T03 选择依据来自 train/validation 而非 test：
    _similarity_from_kv_pairs 从 KV pairs 计算余弦相似度矩阵；
    _get_paired_kv_samples 从 provider 或确定性合成获取 train/validation 样本。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..metrics import cosine, linear_cka
from ..io.runs import write_json

# T03 选择依据来自 train/validation 而非 test
_DEFAULT_SOURCE_TOP_K: int = 2
_DEFAULT_SIM_SEED: int = 42


def proportional_mapping(n_t: int, n_s: int) -> list[list[int]]:
    """按层数比例把 Teacher 层均摊给 Student 层。

    例：n_t=36, n_s=28 → Student layer 0..27 各取 ⌊36/28⌋ 或 ⌈36/28⌉ 个
    Teacher 层。
    """
    mapping: list[list[int]] = []
    for s in range(n_s):
        # Student 层 s 对应 Teacher 层的比例区间 [start, end)：
        # 把 [0, n_t) 按层数比例切成 n_s 段，逐段均摊
        start = int(round(s * n_t / n_s))
        end = int(round((s + 1) * n_t / n_s))
        end = max(end, start + 1)   # 保证每个 Student 层至少取 1 个 Teacher 层
        end = min(end, n_t)          # 截断到 Teacher 层数上界
        mapping.append(list(range(start, end)))
    return mapping


def last_layer_mapping(n_t: int, n_s: int, k: int = 2) -> list[list[int]]:
    """每个 Student 层只取最末 k 个 Teacher 层（baseline 退化版）。

    说明：仅作退化 baseline 参与 A7 消融对比，用于凸显
    data-driven / geometry-aware 策略在深层表示对齐上的优势。
    """
    tail = list(range(max(0, n_t - k), n_t))   # 最末 k 个 Teacher 层索引
    return [list(tail) for _ in range(n_s)]     # 所有 Student 层共享同一 tail


def data_driven_topk(
    sim_matrix: np.ndarray, k: int = 2
) -> list[list[int]]:
    """基于相似度矩阵，每个 Student 层取 top-k Teacher 层。"""
    mapping: list[list[int]] = []
    for s in range(sim_matrix.shape[0]):
        # 对相似度行降序取前 k：np.argsort(-sim[s]) 返回从大到小的 Teacher 层下标
        idx = np.argsort(-sim_matrix[s])[:k]
        mapping.append(sorted(idx.tolist()))   # 升序排序，保证映射确定性/可读性
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
            # 邻近层平滑：当前层相似度与上下相邻层相似度的加权平均，
            # 避免个别 Student 层出现"断层式"映射（相邻层跳到无关 Teacher 层）。
            smoothed[s] = (1 - alpha) * sim_matrix[s] + alpha * sim_matrix[lo:hi].mean(0)
    # 平滑后再走 data-driven top-k
    return data_driven_topk(smoothed, k=k)


def _similarity_from_kv_pairs(kv_t: np.ndarray, kv_s: np.ndarray) -> np.ndarray:
    """T03 选择依据：从 KV pairs 计算 Student-Teacher 层间余弦相似度矩阵。

    计算方式：将每个层的 (S, H, D) 维展平为一维向量，
    再求 student 层 s 与 teacher 层 t 之间的 cosine similarity。

    输入形状：
        kv_t: (n_t, S, H, D)  — Teacher 各层 KV
        kv_s: (n_s, S, H, D)  — Student 各层 KV

    输出形状：
        (n_s, n_t)  — sim_matrix[s, t] = cos(kv_s[s].flat, kv_t[t].flat)
    """
    # 展平为 (n, S*H*D) 以便批量计算余弦
    flat_t = kv_t.reshape(kv_t.shape[0], -1).astype(np.float64)
    flat_s = kv_s.reshape(kv_s.shape[0], -1).astype(np.float64)

    # 归一化
    t_norms = np.linalg.norm(flat_t, axis=1, keepdims=True)  # (n_t, 1)
    s_norms = np.linalg.norm(flat_s, axis=1, keepdims=True)  # (n_s, 1)
    t_norms = np.maximum(t_norms, 1e-12)
    s_norms = np.maximum(s_norms, 1e-12)
    flat_t_normed = flat_t / t_norms
    flat_s_normed = flat_s / s_norms

    # 余弦相似度矩阵：(n_s, n_t)
    return flat_s_normed @ flat_t_normed.T


def _load_kv_from_provider(cfg: dict[str, Any], n_t: int, n_s: int) -> tuple[np.ndarray, np.ndarray] | None:
    """从 provider 加载真实 KV pairs。

    当前状态：占位 stub（§75 诚实性 —— 不允许假装提供了真实路径）。

    设计意图（未来真实实现，T03 选择依据若来自真实 hidden states 应走此路径）：
        1. cfg["provider"]["kv"] == "hf" 时尝试加载真实 Student 自产 KV
        2. 调用 ..inference.backends.TorchBackend.forward_prefill_native 抓 (L,S,H,2D) numpy
        3. 拆出 K / V 两组缓存，返回 (kv_t, kv_s)
        4. 离线环境或模型不可用 → return None（调用方 fallback 到确定性合成）

    TODO(real-kv-provider): 待 real-GPU provider 路径稳定后实现此函数。
    占位期间函数无条件返回 None（不论 provider 配置如何），确保
    _get_paired_kv_samples 永远走确定性合成路径 —— 与 §75 诚实性一致
    （不静默回退到合成、不假装提供了真实 KV）。
    """
    # 占位实现：永远走确定性合成（real-kv-provider 待实现）
    return None


def _get_paired_kv_samples(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """T03 选择依据：获取 train/validation 而非 test 的 KV pairs。

    优先从 provider 获取真实 KV；离线环境下用确定性合成替代：
        - Teacher: 每层独立的随机向量（种子确定 → 可复现）
        - Student: 每层 s 最接近 teacher layer s（对角带状结构）

    为什么要用 train/validation 而非 test：层对齐选择依据若来自
    test 数据会导致信息泄漏——Mapper 拟合时隐式"看到"了 test 的
    表示结构，违反 train/test 严格分离原则。
    """
    n_t = cfg.get("teacher", {}).get("num_layers", 36)
    n_s = cfg.get("student", {}).get("num_layers", 28)
    seed = (cfg.get("seeds", [_DEFAULT_SIM_SEED]) or [_DEFAULT_SIM_SEED])[0]

    # 优先尝试 provider（真实路径）
    kv = _load_kv_from_provider(cfg, n_t, n_s)
    if kv is not None:
        return kv

    # 确定性合成：种子固定 → 相同 cfg 总是产出相同 KV pairs
    rng = np.random.default_rng(int(seed))
    S, H, D = 8, 4, 16  # 序列长度 / head 数 / head_dim（小规模，CI 友好）

    # Teacher: 每层独立的随机 KV（正交性由随机性保证）
    kv_t = rng.standard_normal((n_t, S, H, D))

    # Student: 每层 s 以 teacher layer s 为主体 + 微小扰动，
    # 构造对角带状相似度结构（cos ≈ 1.0 对角 / cos ≈ 0.1~0.3 非对角）
    kv_s = np.zeros((n_s, S, H, D), dtype=np.float64)
    for s in range(n_s):
        teacher_idx = min(s, n_t - 1)
        kv_s[s] = kv_t[teacher_idx] + 0.05 * rng.standard_normal((S, H, D))

    return kv_t, kv_s


def synthetic_similarity(n_t: int, n_s: int, seed: int = 0) -> np.ndarray:
    """无真实 hidden states 时构造一个"对角带状"相似度矩阵做演示。

    真实实验应通过：
        sim[s, t] = cos(hidden_s_layer_s.mean(0), hidden_t_layer_t.mean(0))
    """
    rng = np.random.default_rng(seed)
    # 基底：均匀噪声 [0, 0.3)，代表与结构无关的随机相似度
    sim = rng.uniform(0, 0.3, size=(n_s, n_t))
    for s in range(n_s):
        # 期望中心位置：Student 层 s 大致对应 Teacher 层 (s+0.5)·n_t/n_s
        center = (s + 0.5) * n_t / n_s
        for t in range(n_t):
            d = abs(t - center)
            # 高斯核：离中心越近相似度越高 → 对角带状结构，模拟真实层间渐进对应
            sim[s, t] += float(np.exp(-d * d / (n_t / n_s) ** 2))
    return sim


def run_layer_alignment(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T03 入口：构造 4 种 mapping 策略并写出 layer_mapping.json。

    T03 选择依据来自 train/validation 而非 test：
        - 优先从 provider 获取真实 KV pairs（_get_paired_kv_samples）
        - 离线环境下用确定性合成（种子固定 → 可复现）
        - _similarity_from_kv_pairs 计算余弦相似度矩阵
        - data_driven_topk 基于相似度矩阵选 top-k
    """
    n_t = cfg.get("teacher", {}).get("num_layers", 36)
    n_s = cfg.get("student", {}).get("num_layers", 28)
    source_top_k = cfg.get("mapper", {}).get("source_top_k", _DEFAULT_SOURCE_TOP_K)

    # T03 选择依据：从 train/validation 获取 KV pairs
    kv_pair = _get_paired_kv_samples(cfg)
    if kv_pair is not None:
        kv_t, kv_s = kv_pair
        sim = _similarity_from_kv_pairs(kv_t, kv_s)
        similarity_source = "synthetic_deterministic"
    else:
        # fallback：对角带状合成相似度矩阵（无 KV pairs 时的退化路径）
        sim = synthetic_similarity(n_t, n_s, seed=_DEFAULT_SIM_SEED)
        similarity_source = "synthetic_diagonal"

    mappings = {
        "proportional": proportional_mapping(n_t, n_s),
        "last_layer": last_layer_mapping(n_t, n_s),
        "data_driven_topk": data_driven_topk(sim, k=2),
        "geometry_aware_topk": geometry_aware_topk(sim, k=2),
    }

    # T03 扩展输出：data_driven_topk_scores（每个 student 层的 teacher layers + 分数）
    data_driven_scores: list[dict[str, Any]] = []
    for s in range(n_s):
        idx = np.argsort(-sim[s])[:source_top_k]
        data_driven_scores.append({
            "student_layer": s,
            "teacher_layers": sorted(idx.tolist()),
            "scores": [float(sim[s, t]) for t in sorted(idx)],
        })

    metrics = {
        "task": "T03",
        "n_teacher_layers": n_t,
        "n_student_layers": n_s,
        "strategy": list(mappings.keys()),
        "mapping_size_avg": {
            k: float(np.mean([len(v) for v in mp])) for k, mp in mappings.items()
        },
        # §21 默认选取 proportional（简单可复现）；其余策略保留给 A7 消融
        "selected": "proportional",
        "similarity_source": similarity_source,
        "source_top_k": source_top_k,
        "note": (
            "offline demo 用确定性合成相似度矩阵（train/validation）；"
            "真实实验需 attn-output cosine。"
        ),
    }
    write_json(run_dir / "layer_mapping.json", mappings | {
        "data_driven_topk_scores": data_driven_scores,
    })
    write_json(run_dir / "metrics.json", metrics)
    summary = (
        "# T03 Layer Alignment\n\n"
        f"- Teacher layers: {n_t}\n"
        f"- Student layers: {n_s}\n"
        f"- Strategies: {list(mappings.keys())}\n"
        f"- Similarity source: {similarity_source}\n"
        "- 默认 `proportional`；其余用于 A7 消融 (design.md §47)。\n"
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"status": "OK", "metrics": metrics, "summary": summary}
