"""RAT: Residual-Anchored Translator（残差锚定翻译器，v1.4）。

════════════════════════════════════════════════════════════════════════════════
架构依据（v1.3 复盘 + Qwen3 架构事实）：

1. KV 是残差流的线性投影：V = W_V·h。跨模型 KV 翻译的本质是残差空间翻译。
   RAT 把 per-head 映射分解为
       W_h = ( W_V^{s,h} · R · pinv(W_V^{t,h}) )^T      （行向量约定 out = src @ W）
   其中 R 是两模型 token-embedding 的正交 Procrustes 对齐（2560→2048）——
   Qwen3 家族共享词表（vocab=151936），embedding 几何是**免费的跨模型对齐锚**，
   不需要任何配对数据。该解析核给出一个架构级的初始化，
   残差（目标 − 解析预测）由校准数据的**闭式低秩修正**吸收（无梯度训练）。

2. QK-Norm 事实（已验证 transformers 源码）：Qwen3 在 RoPE 前对 Q/K 做逐头
   RMSNorm —— 缓存 K 是"归一化后+旋转后"状态（类球面几何）；V 无归一化
   （实测教师/学生 V 范数比 7.4×）。因此：
       - K 路径：de-RoPE 后的正交几何，解析核 + 小秩修正即可；
       - V 路径：尺度失配为主，低秩修正 + 逐头校准尺度吸收。

3. GQA 头对应是排列问题：教师 32Q/8KV vs 学生 16Q/8KV，head h↔h 的
   identity 假设无依据。RAT 用校准数据的头均值余弦相似度 + 匈牙利算法
   求最优头排列（默认开启）。

4. 注意力汇：首 token（position 0）承载不成比例的注意力质量，其翻译误差
   被放大。RAT 可选将 position 0 的输出 KV 直接替换为校准集学生自 KV
   的 position-0 均值（不翻译）。

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_KIND_WEIGHT_KEY = {"K": "k_proj", "V": "v_proj"}


def _resolve_model_dir(model_ref: str) -> Path:
    """model_id 或本地路径 → 本地 snapshot 目录（与 evaluator 的 modelscope 缓存约定一致）。"""
    p = Path(os.path.expanduser(model_ref))
    if p.is_dir():
        return p
    cache = Path(os.path.expanduser(
        f"~/.cache/modelscope/models/{model_ref.replace('/', '--')}/snapshots/master"
    ))
    if cache.is_dir():
        return cache
    raise FileNotFoundError(f"RAT: 找不到模型目录: {model_ref}")


def _load_tensor(model_dir: Path, key: str) -> np.ndarray:
    """从（可能分片的）safetensors 中按 key 部分加载一个张量 → float32 numpy。"""
    from safetensors import safe_open

    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
        shard = wmap.get(key)
        if shard is None:
            raise KeyError(f"RAT: 权重 {key} 不在模型索引中")
        with safe_open(model_dir / shard, framework="pt") as sf:
            return sf.get_tensor(key).float().numpy()
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as sf:
            return sf.get_tensor(key).float().numpy()
    raise FileNotFoundError(f"RAT: {model_dir} 下无 safetensors 权重")


def embedding_procrustes(e_t: np.ndarray, e_s: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """残差空间对齐：求 R（hidden_t → hidden_s）最小化 ||e_t R − e_s||² + λ|R|²。

    e_t: (vocab, hidden_t)，e_s: (vocab, hidden_s)，共享词表逐行对应。
    闭式：R = (e_t^T e_t + λI)^{-1} e_t^T e_s（岭回归；e_t 满列秩时
    λ→0 精确恢复任意线性关系 R0，不受正交约束限制 —— 此前的
    列正交 Procrustes 公式对非平衡 Stiefel 问题不成立，已修正）。
    """
    a = e_t.astype(np.float64)
    b = e_s.astype(np.float64)
    g = a.T @ a                                   # (hidden_t, hidden_t)
    lam_eff = lam * max(1.0, float(np.trace(g)) / g.shape[0])
    return np.linalg.solve(
        g + lam_eff * np.eye(g.shape[0]), a.T @ b
    ).astype(np.float32)


def hungarian_head_matching(t_head_stats: np.ndarray, s_head_stats: np.ndarray) -> np.ndarray:
    """头排列求解：最大化教师头/学生头统计轮廓的相似度总和。

    t_head_stats: (H_t, K)，s_head_stats: (H_s, K) —— 每头的统计轮廓
    （推荐用展平的协方差矩阵而非均值：V 的头均值≈小随机向量，
    余弦相似度退化为噪声；协方差对头几何更可辨识）。
    返回 perm: (H_s,) —— 学生头 j 对应教师头 perm[j]。
    """
    from scipy.optimize import linear_sum_assignment

    tn = t_head_stats - t_head_stats.mean(axis=1, keepdims=True)
    sn = s_head_stats - s_head_stats.mean(axis=1, keepdims=True)
    tn = tn / (np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9)
    sn = sn / (np.linalg.norm(sn, axis=1, keepdims=True) + 1e-9)
    sim = tn @ sn.T  # (H_t, H_s)
    row, col = linear_sum_assignment(-sim)
    perm = np.zeros(sim.shape[1], dtype=np.int64)
    perm[col] = row
    return perm


def _head_covariance_profile(kv: np.ndarray, layer: int) -> np.ndarray:
    """第 layer 层各 KV 头的协方差轮廓 (H, D*D)（按样本×位置展平）。"""
    H, D = kv.shape[2], kv.shape[3]
    x = kv[layer].reshape(-1, H, D)
    return np.stack([
        np.cov(x[:, h, :].T).reshape(-1) for h in range(H)
    ])


class RATMapper:
    """残差锚定翻译器：解析核（权重+embedding锚） + 闭式低秩修正。

    接口与其它 mapper 兼容（fit / fit_batch / transform / n_params），
    另需在 fit 前调用 setup_weights(teacher_dir, student_dir) 加载权重并
    构建 embedding 锚。权重加载失败时优雅降级：解析核置零，
    退化为纯低秩 ridge 校正（诚实告警）。
    """

    def __init__(
        self,
        lam: float = 1e-3,
        rank: int = 16,
        head_match: bool = True,
        sink_override: bool = True,
        use_embedding_anchor: bool = True,
    ):
        self.lam = lam
        self.rank = int(rank)
        self.head_match = bool(head_match)
        self.sink_override = bool(sink_override)
        self.use_embedding_anchor = bool(use_embedding_anchor)

        self.W: dict[tuple[str, int, int], np.ndarray] = {}
        self.perm: dict[int, np.ndarray] = {}       # student layer → teacher head perm
        self._sink: dict[str, np.ndarray] = {}      # kind → (L_s, H, D) position-0 学生统计
        self._analytic_ready = False
        self._teacher_dir: Path | None = None
        self._student_dir: Path | None = None
        self._R: np.ndarray | None = None           # embedding 锚 (hidden_t, hidden_s)
        self._teacher_meta: dict[str, Any] = {}
        self._student_meta: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 权重与解析核
    # ------------------------------------------------------------------

    def setup_weights(self, teacher_dir: str | Path, student_dir: str | Path) -> None:
        """加载两侧模型目录引用；embedding 锚延迟到首次 fit 时构建（避免重复开销）。"""
        self._teacher_dir = _resolve_model_dir(str(teacher_dir))
        self._student_dir = _resolve_model_dir(str(student_dir))
        logger.info("[RAT] weights dirs: t=%s s=%s", self._teacher_dir, self._student_dir)

    def _ensure_anchor(self) -> None:
        """构建 embedding Procrustes 锚 R（一次）。失败则降级为 None（纯校正模式）。"""
        if self._analytic_ready:
            return
        try:
            if self.use_embedding_anchor:
                e_t = _load_tensor(self._teacher_dir, "model.embed_tokens.weight")
                e_s = _load_tensor(self._student_dir, "model.embed_tokens.weight")
                if e_t.shape[0] != e_s.shape[0]:
                    raise ValueError(
                        f"RAT: 词表不一致 t={e_t.shape[0]} s={e_s.shape[0]}，embedding 锚不可用"
                    )
                self._R = embedding_procrustes(e_t, e_s)
                logger.info("[RAT] embedding anchor ready: R %s", self._R.shape)
            else:
                self._R = None
        except Exception as e:  # noqa: BLE001
            self._R = None
            logger.warning("[RAT] embedding anchor 构建失败，退化为纯校正模式: %s", e)
        self._analytic_ready = True

    def _analytic_maps_for_layer(
        self, kind: str, t_layer: int, s_layer: int, D: int
    ) -> dict[tuple[int, int], np.ndarray] | None:
        """解析核：一次加载 (t_layer, s_layer) 两侧投影，返回 {(t_head, s_head): W}。

        W = ( W^{s,h} · R · pinv(W^{t,h}) )^T（行向量约定 out = src @ W）。
        失败 → None（调用方退化为纯校正模式）。
        """
        if self._R is None:
            return None
        try:
            wt = _load_tensor(
                self._teacher_dir,
                f"model.layers.{t_layer}.self_attn.{_KIND_WEIGHT_KEY[kind]}.weight",
            )  # (H*D_t, hidden_t)
            ws = _load_tensor(
                self._student_dir,
                f"model.layers.{s_layer}.self_attn.{_KIND_WEIGHT_KEY[kind]}.weight",
            )  # (H*D_s, hidden_s)
            H_t, H_s = wt.shape[0] // D, ws.shape[0] // D
            R = self._R.astype(np.float64)
            out: dict[tuple[int, int], np.ndarray] = {}
            for t_head in range(H_t):
                w_t_h = wt[t_head * D:(t_head + 1) * D, :].astype(np.float64)
                # 行向量约定：h_t(row) = V_t(row) @ pinv(w_t).T；h_s = h_t @ R；
                # V_s(row) = h_t @ R @ w_s.T  ⇒  W = pinv(w_t).T @ R @ w_s.T
                p = np.linalg.pinv(w_t_h)        # (hidden_t, D)
                mid = p.T @ R                    # (D, hidden_s)
                for s_head in range(H_s):
                    w_s_h = ws[s_head * D:(s_head + 1) * D, :].astype(np.float64)
                    out[(t_head, s_head)] = (mid @ w_s_h.T).astype(np.float32)  # (D, D)
            return out
        except Exception as e:  # noqa: BLE001
            logger.debug("[RAT] analytic maps failed for %s l%d→l%d: %s",
                         kind, t_layer, s_layer, e)
            return None

    # ------------------------------------------------------------------
    # 拟合（标准接口 + fit_batch）
    # ------------------------------------------------------------------

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> None:
        self.fit_batch([(kv_t, kv_s)], layer_map, kv_kind=kv_kind,
                       positions=positions, de_rope_fn=de_rope_fn)

    def fit_batch(
        self,
        samples: list[tuple[np.ndarray, np.ndarray]],
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> None:
        """闭式拟合：头匹配 → 解析核 → 低秩 ridge 修正（SVD 截断）→ sink 统计。"""
        if not samples:
            raise ValueError("RATMapper.fit_batch: samples 不能为空")
        self._ensure_anchor()
        kv_s0 = samples[0][1]
        L_s, _, H, D = kv_s0.shape
        L_t = samples[0][0].shape[0]

        # ---- 头匹配（逐学生层，用首教师层；协方差轮廓比头均值更可辨识）----
        for s in range(L_s):
            t_layer = layer_map[s][0] if layer_map[s] else min(s, L_t - 1)
            if self.head_match:
                t_prof = _head_covariance_profile(samples[0][0], t_layer)
                s_prof = _head_covariance_profile(samples[0][1], s)
                if t_prof.shape[0] == s_prof.shape[0]:
                    self.perm[s] = hungarian_head_matching(t_prof, s_prof)
                else:
                    self.perm[s] = np.arange(min(s_prof.shape[0], t_prof.shape[0]))
            else:
                self.perm[s] = np.arange(H)

        # ---- 逐 (s, h)：解析核 + 低秩 ridge 修正 ----
        for s in range(L_s):
            teachers = layer_map[s] if s < len(layer_map) else [min(s, L_t - 1)]
            t_primary = teachers[0]
            perm_s = self.perm.get(s, np.arange(H))
            # 一次加载该层对的解析核（含所有头组合），失败 → None（纯校正）
            analytic = self._analytic_maps_for_layer(kv_kind, t_primary, s, D)
            src_rows = {h: [] for h in range(H)}
            res_rows = {h: [] for h in range(H)}
            for kv_t, kv_s in samples:
                S_i = min(kv_t.shape[1], kv_s.shape[1])
                pos = (
                    np.arange(S_i, dtype=np.float64)
                    if positions is None else positions[:S_i]
                )
                for t in teachers:
                    k_raw = kv_t[t, :S_i]
                    if de_rope_fn is not None and kv_kind == "K":
                        k_raw = de_rope_fn(k_raw, pos)
                    for h in range(H):
                        th = int(perm_s[h]) if h < len(perm_s) else h
                        src = k_raw[:, th, :].astype(np.float64)  # (S_i, D)
                        w0 = (
                            analytic.get((th, h)) if analytic else None
                        )
                        base = src @ w0 if w0 is not None else np.zeros_like(src)
                        tgt = kv_s[s, :S_i, h, :].astype(np.float64)
                        src_rows[h].append(src)
                        res_rows[h].append(tgt - base)
            for h in range(H):
                x = np.concatenate(src_rows[h])   # (N, D)
                r = np.concatenate(res_rows[h])   # (N, D)
                # ridge 修正：Δ = solve(x^T x + λI, x^T r)，再 SVD 截断到 rank
                g = x.T @ x
                lam_eff = self.lam * max(1.0, np.trace(g) / D)  # λ 随数据尺度归一
                delta = np.linalg.solve(g + lam_eff * np.eye(D), x.T @ r)
                if self.rank < D:
                    u, sv, vt = np.linalg.svd(delta, full_matrices=False)
                    keep = min(self.rank, int((sv > 1e-10).sum()))
                    delta = (u[:, :keep] * sv[:keep]) @ vt[:keep]
                th = int(perm_s[h]) if h < len(perm_s) else h
                w0 = analytic.get((th, h)) if analytic else None
                self.W[(kv_kind, s, h)] = (
                    (w0 if w0 is not None else 0.0) + delta
                ).astype(np.float32)

        # ---- sink 统计（学生 position 0 均值，(L_s, H, D)）----
        if self.sink_override:
            self._sink[kv_kind] = np.stack(
                [kv_s[:, 0, :, :] for _, kv_s in samples]
            ).mean(axis=0).astype(np.float32)

    @property
    def n_params(self) -> int:
        return int(sum(w.size for w in self.W.values()))

    # ------------------------------------------------------------------
    # 变换
    # ------------------------------------------------------------------

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """逐 (s, h)：de-RoPE → 头重排 → (src @ W) ；可选 sink 覆盖 position 0。"""
        from .math import _require_kv_kind

        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "RATMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H, D), dtype=np.float32)
        for s in range(L_s):
            teachers = layer_map[s] if s < len(layer_map) else [min(s, L_t - 1)]
            perm_s = self.perm.get(s, np.arange(H))
            acc = np.zeros((S, H, D), dtype=np.float64)
            for t in teachers:
                k_raw = kv_t[t]
                if de_rope_fn is not None and kv_kind == "K":
                    k_raw = de_rope_fn(k_raw, positions)
                for h in range(H):
                    th = int(perm_s[h]) if h < len(perm_s) else h
                    w = self.W[(kv_kind, s, h)]
                    acc[:, h, :] += k_raw[:, th, :].astype(np.float64) @ w.astype(np.float64)
            out[s] = (acc / len(teachers)).astype(np.float32)
        # sink 覆盖：position 0 用学生统计（该 kind 必须已 fit）
        if self.sink_override and kv_kind in self._sink:
            out[:, 0, :, :] = self._sink[kv_kind]
        return out
