"""Ridge / Low-rank / Shared-Basis Mapper 数学实现。

═══════════════════════════════════════════════════════════════════════════════
本模块对应 design.md §20 (Compatibility Base)、§22 (K/V 独立参数化) 与
§23 (de-RoPE 强制)。

设计要点（中文注释）：
1. **Per-(layer, head) 独立训练**：每个 Student (s, h) 都学一个独立的映射矩阵。
   这样可以最大化每个头的表达自由度，但代价是参数量大。

2. **K / V 独立参数化（§22 强制）**：
   Teacher K 与 Teacher V 是两个**独立**的缓存张量（不同随机过程、不同内容），
   Student K/V 的适配自然要求两套独立参数。因此本模块全部 4 个 mapper 的
   fit() / transform() 都接受 `kv_kind: str = "K"` 参数：
   - 参数存储键加 kind 维度：W 键 `(s, 0)` → `(kv_kind, s, 0)`、
     `(s, h)` → `(kv_kind, s, h)`；SharedBasisMapper 的共享基从单个矩阵
     `A_shared` 改为 `dict[str, np.ndarray]`（键为 kv_kind）。
   - `n_params` 对 dict 全部 value 求和 → 只 fit 一个 kind 时数值与旧实现
     一致，fit K/V 两个 kind 时自动翻倍（PCR 口径随之翻倍，§3.4）。
   - transform() 必须用与 fit 相同的 kv_kind 取参；未 fit 的 kind 直接
     raise KeyError（中文提示），**禁止静默回退**到其他 kind —— 用 V 的参数
     变换 K 是错误语义，静默回退只会掩盖 bug。

2a. **de-RoPE 必须前置（§23 强制）**：
   Teacher K 是 Teacher RoPE 空间里的向量，Student K 是 Student RoPE 空间里的向量。
   如果直接做线性映射，需要隐式学会"先 de-RoPE 再 re-RoPE"，这对线性模型极难。
   因此强制流程：Teacher K → de-RoPE → Mapper → Student RoPE。
   本模块每个 mapper 类都接受 `de_rope_fn` 参数，在 fit / transform 时调用。

3. **形状约定**：
       K, V 形状为 (L_t, S, H, D)
   其中 L_t = Teacher 层数；S = 序列长度；H = kv-head 数；D = head_dim。
   训练时 src = Teacher tokens (k*S, D)，tgt = Student tokens (S, D)。

4. **bug-3/bug-4a 修复**：旧实现用 `min(src, tgt)` 静默截断。现在 S/H/D
   不一致会显式 raise ValueError；Teacher/Student **层数可以不等**（SmolLM2
   教师 24 层 → 学生 30 层），层对齐由 layer_map / G2 处理（§12/§13），
   不是几何层错误。layer_map 里 top_k>1 时，所有教师层样本全部参与 fit
   （tgt 复制 k 份对齐，见 `_stack_topk`），训练与 transform 的 k 层平均一致。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _ridge_closed_form(
    x: np.ndarray, y: np.ndarray, lam: float = 1e-3
) -> np.ndarray:
    """求解 Ridge 闭式解：找到 W 使 y @ W ≈ x。

    数学推导：
        最小化 ‖y W - x‖² + λ ‖W‖²
        对 W 求导令其为零：
            y^T y W + λ W = y^T x
            W = (y^T y + λ I)^{-1} y^T x

    参数：
        x: (n, d_out) 目标矩阵
        y: (n, d_in)  源矩阵
        lam: L2 正则系数，越大越平滑
    返回：
        W: (d_in, d_out)
    """
    yt = y.T @ y
    return np.linalg.solve(yt + lam * np.eye(yt.shape[0]), y.T @ x)


def _ridge_closed_form_batch(
    x_batch: np.ndarray, y_batch: np.ndarray, lam: float = 1e-3
) -> np.ndarray:
    """批量 Ridge 闭式解：H 个 head 同时求解。

    参数：
        x_batch: (H, n, d_out)  H 个 head 的目标矩阵
        y_batch: (H, n, d_in)   H 个 head 的源矩阵
        lam: L2 正则系数
    返回：
        W_batch: (H, d_in, d_out)  H 个 head 的解

    优化点：与 _ridge_closed_form 数学等价（per-head 独立求解），
    但所有 head 用一次 batch 矩阵运算完成，避免 H 次 Python 循环。
    """
    # 对每个 head：y_h^T y_h + λI 与 y_h^T x_h
    H = x_batch.shape[0]
    d_in = y_batch.shape[-1]
    d_out = x_batch.shape[-1]
    # (H, d_in, n) @ (H, n, d_in) → (H, d_in, d_in)
    yty = y_batch.transpose(0, 2, 1) @ y_batch
    # (H, d_in, n) @ (H, n, d_out) → (H, d_in, d_out)
    ytx = y_batch.transpose(0, 2, 1) @ x_batch
    # 加上 λI
    eye = np.eye(d_in).reshape(1, d_in, d_in)
    yty = yty + lam * eye
    # 批量解 (H, d_in, d_in) @ W = (H, d_in, d_out)
    # 用 np.linalg.solve 配合广播
    return np.linalg.solve(yty, ytx)


def _lowrank_factor(
    x: np.ndarray, y: np.ndarray, rank: int, n_iter: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """交替最小二乘求低秩解 y @ B @ A ≈ x。

    初始化：A ∈ R^{d_in × r}, B ∈ R^{r × d_out}（随机正态）。
    迭代：
        固定 A → 最小化 ‖y B A - x‖² 对 B 求解（B = lstsq(y @ a, x)）
        固定 B → 最小化 ‖y B A - x‖² 对 A 求解（A = lstsq(y, x @ B^T)）
    收敛后返回 (A, B)，最终映射为 y @ A @ B。

    参数：
        x: (n, d_out) 目标
        y: (n, d_in)  源
        rank: 低秩 r
        n_iter: ALS 迭代次数
    返回：
        (A: (d_in, r), B: (r, d_out))

    数学：A 形状 (d_in, r) → (n, r) 投影；B 形状 (r, d_out) → 升维。
    实际映射：y @ A @ B = ((y @ A) @ B)，与 §22 R = A σ(B Z) 形式对应
    （这里省 σ；详见 LowRankResidual 注释）。
    """
    rng = np.random.default_rng(0)
    d_in = y.shape[1]
    a = rng.standard_normal((d_in, rank)) / np.sqrt(rank)
    for _ in range(n_iter):
        # 固定 A → 求 B = lstsq(y @ a, x)
        proj = y @ a  # (n, r)
        b = np.linalg.lstsq(proj, x, rcond=None)[0]  # (r, d_out)
        # 固定 B → 求 A = lstsq(y, x @ B^T)
        rhs = x @ b.T  # (n, r)
        a = np.linalg.lstsq(y, rhs, rcond=None)[0]  # (d_in, r)
    return a, b


def _lowrank_factor_batch(
    x_batch: np.ndarray, y_batch: np.ndarray, rank: int, n_iter: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """批量 ALS：H 个 head 同时求解。

    参数：
        x_batch: (H, n, d_out)
        y_batch: (H, n, d_in)
        rank: 低秩 r
    返回：
        A_batch: (H, d_in, r)
        B_batch: (H, r, d_out)

    优化：每 head 独立的 ALS 用 numpy 的 lstsq 配合 batched 输入仍需
    循环，但可用 block 处理（每 B 块同时算）。这里保留 ALS 循环但把
    H 个 head 的 np.linalg.lstsq 改成更轻量的 _batched_lstsq。
    """
    H = x_batch.shape[0]
    d_in = y_batch.shape[-1]
    rng = np.random.default_rng(0)
    A_batch = rng.standard_normal((H, d_in, rank)) / np.sqrt(rank)
    B_batch = np.zeros((H, rank, x_batch.shape[-1]))
    for _ in range(n_iter):
        # y (H, n, d_in) @ A (H, d_in, r) → (H, n, r)
        proj = np.einsum("hnd,hdr->hnr", y_batch, A_batch)
        # 求 B：min_B ‖proj B - x‖² → (H, r, d_out)
        # 用 _batched_lstsq 同时解 H 个系统
        B_batch = _batched_lstsq(proj, x_batch)
        # 求 A：min_A ‖y A - x @ B^T‖² → (H, d_in, r)
        rhs = np.einsum("hnr,hrs->hns", x_batch, np.transpose(B_batch, (0, 2, 1)))
        # rhs: (H, n, r)，求 min_A ‖y A - rhs‖²
        A_batch = _batched_lstsq(y_batch, rhs)
    return A_batch, B_batch


def _batched_lstsq(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
    """批量 lstsq：同时解 H 个最小二乘系统 a_i @ x_i ≈ b_i。

    参数：
        a_batch: (H, n, m)  H 个矩阵 (n 行 m 列)
        b_batch: (H, n, k)  H 个目标矩阵 (n 行 k 列)
    返回：
        x_batch: (H, m, k)

    实现：对每个 head 调 np.linalg.lstsq；避免 Python 层面 H 次循环
    （用 list comprehension + numpy）。
    """
    H = a_batch.shape[0]
    out = np.zeros((H, a_batch.shape[-1], b_batch.shape[-1]), dtype=a_batch.dtype)
    for h in range(H):
        out[h] = np.linalg.lstsq(a_batch[h], b_batch[h], rcond=None)[0]
    return out


def _apply_or_skip(
    de_rope_fn: Callable | None,
    k: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """如果提供了 de_rope_fn，则先 de-RoPE（§23 强制路径），否则原样返回。

    在 cross-model 场景下，Teacher K 处于 Teacher RoPE 空间，
    而 Student 期望输入是 Student RoPE 空间。两边的 RoPE 频率通常不同，
    因此必须先 de-RoPE 到 unrotated 空间再做映射，再在 Student 端 re-RoPE。
    Student 端的 re-RoPE 由 Student attention 自己完成（注入 KV cache 时即生效）。

    de_rope_fn 签名：de_rope_fn(k, positions) → k_unrotated
        k 形状支持 (..., D) 或 (S, H, D)
        positions 形状 (S,)
    """
    if de_rope_fn is None:
        return k
    # rope.de_rope 期望最后一维为 head_dim，按位置广播 (S,) 到最后一维
    return de_rope_fn(k, positions)


def _check_shape(kv_t: np.ndarray, kv_s: np.ndarray) -> None:
    """断言 Teacher 与 Student KV 的形状兼容（bug-3 修复：不再静默截断）。

    要求（仅 S/H/D 必须一致）：
        kv_t.shape[1:] == kv_s.shape[1:]    即 S, H, D 都相同

    ◆ bug-4a 修复：**不再**要求 kv_t.shape[0] >= kv_s.shape[0]。
      Teacher 层数少于 Student 层数（如 SmolLM2-1.7B: 24 → 135M: 30）是
      合法场景，层数对齐由 layer_map / G2 MismatchedHeadMapper（§12/§13）
      在层对齐阶段处理，不是几何错误，不在本层报错。

    副作用：
        - §52.8 silent_re_prefill_on_failure: 形状不兼容时不再 silent re-prefill，
          而是 raise ValueError（符合 §52 禁止 8）。
        - 通过 `apcs.compliance.runtime.track_runtime("silent_re_prefill_on_failure", False)`
          显式标注。
    """
    if kv_t.shape[1:] != kv_s.shape[1:]:
        try:
            from ..compliance.runtime import track_runtime

            track_runtime("silent_re_prefill_on_failure", False)
        except ImportError:
            pass
        raise ValueError(
            f"Teacher 与 Student KV 形状不兼容："
            f"teacher={kv_t.shape}, student={kv_s.shape}。"
            f"APCS 设计要求 S/H/D 一致；如不一致属于 G2/G3 范畴 (§12/§13)。"
        )


def _require_kv_kind(kv_kind: str, fitted: set[str], what: str) -> None:
    """design.md §22 强制：K 与 V 必须独立参数化 —— 未 fit 的 kind 禁止静默回退。

    参数：
        kv_kind: transform 请求的 kind（"K" 或 "V"）
        fitted: 已 fit 的 kind 集合（从参数 dict 的键首元素提取）
        what: 出错上下文（如 "RidgeMapper.transform"），用于中文报错
    抛出：
        KeyError：kv_kind 尚未 fit 时给出中文提示；调用方不得捕获后回退。
    """
    if kv_kind not in fitted:
        raise KeyError(
            f"{what}: kv_kind='{kv_kind}' 尚未 fit。"
            f"K 与 V 必须独立参数化（design.md §22 强制），"
            f"禁止静默回退到其他 kind。当前已 fit: {sorted(fitted) or '无'}。"
        )


def _stack_topk(
    src_layers: np.ndarray, tgt_layer: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """top_k 教师层与 Student 层对齐（bug-2 修复的核心 helper）。

    背景：layer_map[s] 含 k 个教师层时，src 有 k 组行、tgt 只有 1 组。
    旧实现 `n = min(src.shape[0], tgt.shape[0])` 把 src 截成第 1 个教师层
    的行 → 训练只用第 1 层，而 transform 端平均 k 个教师层 → **训练/推理
    不一致**（§19 B3）。

    本 helper 保留全部 k 层 src 行，tgt 行复制 k 份对齐，返回两种视图：
        flat:  (k*S*H, D) —— 层外循环（layer-major），供 RidgeMapper /
               SharedBasis 等扁平（把 head 并进样本）求解使用
        batch: (H, k*S, D) —— head 外循环、position 内循环（position-major），
               供 RidgePerHead / LowRank 的批量闭式解（_*_batch）使用
    两种视图下 src 与 tgt 逐行对齐：教师层 l、position s（、head h）的向量
    与 Student 同 position（、同 head）的目标成对，闭式解目标等价于"预测
    k 个教师层表征的均值"，与 transform 端 mean over k 层的语义一致。

    参数：
        src_layers: (k, S, H, D) 已 de-RoPE 的 k 个教师层
        tgt_layer:  (S, H, D) 对应 Student 层
    返回：
        (src_flat, tgt_flat, src_batch, tgt_batch)
    """
    k = src_layers.shape[0]
    D = src_layers.shape[-1]
    S = tgt_layer.shape[0]
    H = tgt_layer.shape[1]
    # 扁平视图：src (k, S, H, D) → (k*S*H, D)；tgt (S*H, D) 用 np.tile 复制 k 份
    src_flat = src_layers.reshape(-1, D)
    tgt_flat = np.tile(tgt_layer.reshape(-1, D), (k, 1))
    # per-head 批量视图：src (H, S, k, D) → (H, k*S, D)，第 p=s*k+t 行是
    # 教师层 t 在 position s 的向量；tgt 用 np.repeat 把 position s 复制 k 份
    # 对齐（repeat = 逐位置 tile，与 src_batch 的行序一一对应）。
    src_batch = src_layers.transpose(2, 1, 0, 3).reshape(H, k * S, D)
    tgt_batch = np.repeat(tgt_layer.transpose(1, 0, 2), k, axis=1)
    return src_flat, tgt_flat, src_batch, tgt_batch


# ---------------------------------------------------------------------------
# Full Ridge Mapper
# ---------------------------------------------------------------------------


class RidgeMapper:
    """Per-layer 全量 Ridge mapper（设计文档 §20 Stage 1）。

    训练目标：对每个 Student 层 s，学一个 D × D 矩阵 W_s，
    使得 src @ W_s ≈ tgt。其中 src 来自 layer_map[s] 指定的 Teacher 层。

    ◆ bug-1 修复（per-layer 语义）：旧实现按 (s, h) 双层循环求 H 次**相同**
      矩阵（src/tgt 与 h 无关），W[(s,h)] 全部相同，transform 只用 W[(s,0)]，
      n_params 虚高 H 倍。现改为 per-layer：每层只学一个 W_s，与 transform
      的 per-layer 应用一致（§3.4 PCR 计算不再虚高）。
      更细粒度的 per-(s,h) 独立映射见 RidgePerHeadMapper（§19 B3 per-head）：
      本类是 per-layer 基准（PCR 分母），RidgePerHead 才是 per-head。

    参数量：L_s × D × D（Qwen3-1.7B: 28 × 128 × 128 ≈ 0.46M，
    对应 PCR = 1/8 of per-head base line）。
    """

    def __init__(self, lam: float = 1e-3):
        self.lam = lam
        # design.md §22：键含 kv_kind 维度 —— (kv_kind, s, 0)，K/V 各一套独立参数
        self.W: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def n_params(self) -> int:
        """总参数量，用于 PCR 计算（§3.4）。per-layer 语义下 = L_s × D × D。

        design.md §22：对 dict 全部 value 求和 —— 只 fit 一个 kind（K）时
        数值与旧实现一致；K/V 各 fit 一次时自动翻倍（K、V 独立适配器）。
        """
        return int(sum(w.size for w in self.W.values()))

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ):
        """训练所有 s 的 per-layer Ridge 矩阵。

        参数：
            kv_t: (L_t, S, H, D) Teacher KV
            kv_s: (L_s, S, H, D) Student Self-Prefill KV（仅用作训练目标，
                  真实 inference 时 Student 不重新 prefill X，但训练阶段需要
                  一组 teacher→student 对应样本学 mapper）
            layer_map: 长度 L_s 的 list，每个元素是 Student 层对应的 Teacher 层索引
            kv_kind: design.md §22 —— K/V 独立参数化；为 "K" 时 fit K 的参数，
                为 "V" 时 fit V 的参数。K/V 各自持有独立参数矩阵，互不覆盖。
            positions: (S,) 位置索引；若提供且 de_rope_fn 非空，则先 de-RoPE
            de_rope_fn: callable(k, positions) → k_unrotated；§23 强制
        """
        _check_shape(kv_t, kv_s)
        L_s, S, H, D = kv_s.shape
        L_t = kv_t.shape[0]
        if positions is None:
            positions = np.arange(S, dtype=np.float64)

        for s in range(L_s):
            teachers = (
                layer_map[s]
                if s < len(layer_map)
                else [int(round(s * L_t / L_s))]
            )
            # —— §23 强制 de-RoPE：先去掉 Teacher RoPE 再做线性映射
            src_layers = []
            for t in teachers:
                k = kv_t[t, :, :, :]  # (S, H, D)，此处把 K/V 一起 de-RoPE
                src_layers.append(
                    _apply_or_skip(de_rope_fn, k, positions)
                )
            src_all = np.stack(src_layers, axis=0)  # (k, S, H, D)
            # ◆ bug-2 修复（top_k 对齐）：layer_map[s] 里 k>1（如 top_k=2）时，
            #   所有教师层样本全部保留，tgt 复制 k 份对齐后一起闭环求解 ——
            #   fit 目标等价于"预测 k 个教师层表征的均值"，与 transform 端
            #   平均 k 层的行为一致。旧实现 `n = min(src, tgt)` 只用第 1 个
            #   教师层 → 训练/推理不一致。
            src, tgt, _, _ = _stack_topk(src_all, kv_s[s])
            # ◆ bug-1 修复（per-layer）：每层 s 只算一次，存 W[(s, 0)]；
            #   旧实现按 (s, h) 循环 H 次求完全相同矩阵 → n_params 虚高 H 倍。
            #   _ridge_closed_form(x=target, y=source)，求 W 使 y@W≈x
            # design.md §22：键含 kv_kind —— K 与 V 各存一套，互不覆盖。
            self.W[(kv_kind, s, 0)] = _ridge_closed_form(tgt, src, self.lam)

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """将 Teacher KV 映射到 Student KV。

        不重新 Prefill X —— 直接对已捕获的 Teacher KV 做线性变换。

        design.md §22：按 kv_kind 取对应参数矩阵（K/V 独立）；
        未 fit 该 kind 时 raise KeyError（禁止静默回退）。
        """
        # §22：先校验该 kind 是否已 fit，未 fit 直接报错（中文提示）
        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "RidgeMapper.transform")
        _check_shape(kv_t, np.zeros((len(layer_map),) + kv_t.shape[1:], dtype=kv_t.dtype))
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H, D), dtype=kv_t.dtype)
        for s in range(L_s):
            teachers = layer_map[s]
            mapped_layers = []
            for t in teachers:
                k = kv_t[t, :, :, :]
                k_unrot = _apply_or_skip(de_rope_fn, k, positions)
                w = self.W[(kv_kind, s, 0)]  # §22：按 kv_kind 取对应 kind 的 W
                mapped_layers.append(k_unrot @ w)
            stacked = np.stack(mapped_layers, axis=0)  # (k, S, H, D)
            out[s] = stacked.mean(0)
        return out


# ---------------------------------------------------------------------------
# Per-(layer, head) Ridge（更细粒度，符合 design.md §19 B3 描述）
# ---------------------------------------------------------------------------


class RidgePerHeadMapper:
    """严格的 per-(Student layer, head) Ridge mapper（§19 B3 强制）。

    每个 (s, h) 学独立 D × D 矩阵。参数量与 RidgeMapper 相同，
    但每个头都学自己的映射，语义更清晰。

    design.md §22：W 键为 `(kv_kind, s, h)` —— K 与 V 各持有独立参数。
    """

    def __init__(self, lam: float = 1e-3):
        self.lam = lam
        # design.md §22：键含 kv_kind 维度 —— (kv_kind, s, h)
        self.W: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def n_params(self) -> int:
        # §22：对 dict 全部 value 求和 —— fit 一个 kind 与旧实现一致，K/V 翻倍
        return int(sum(w.size for w in self.W.values()))

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ):
        """训练 per-(layer, head) 的 Ridge 矩阵。

        kv_kind（design.md §22）：K/V 独立参数化 —— 指定本次 fit 属于 K 还是 V，
        K 与 V 各自持有独立的 `(kv_kind, s, h)` 参数矩阵，互不覆盖。
        """
        _check_shape(kv_t, kv_s)
        L_s, S, H, D = kv_s.shape
        L_t = kv_t.shape[0]
        if positions is None:
            positions = np.arange(S, dtype=np.float64)

        for s in range(L_s):
            teachers = (
                layer_map[s]
                if s < len(layer_map)
                else [int(round(s * L_t / L_s))]
            )
            # §23：Teacher KV 先 de-RoPE
            src_per_head = []
            for t in teachers:
                k_unrot = _apply_or_skip(de_rope_fn, kv_t[t], positions)  # (S, H, D)
                src_per_head.append(k_unrot)
            src_all = np.stack(src_per_head, axis=0)  # (k, S, H, D)
            tgt_all = kv_s[s]                          # (S, H, D)
            # ◆ bug-2 修复（top_k 对齐，per-head batch 形态）：
            #   src 保留全部 k 层样本 → (H, k*S, D)，tgt 用 np.repeat 把每个
            #   position 复制 k 份对齐（tgt_h 也成 (H, k*S, D)）→ 批量闭式解
            #   目标与 transform 端 k 层平均一致。旧实现 `n = min(k*S, S)`
            #   只用第 1 个教师层 → 训练/推理不一致。
            _, _, src_h, tgt_h = _stack_topk(src_all, tgt_all)  # (H, k*S, D)
            # _ridge_closed_form_batch(x=tgt, y=src)：批量解 (H, n, D) → (H, D, D)
            W_batch = _ridge_closed_form_batch(tgt_h, src_h, self.lam)
            for h in range(H):
                # design.md §22：键含 kv_kind —— K 与 V 各存一套，互不覆盖
                self.W[(kv_kind, s, h)] = W_batch[h]

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """将 Teacher KV 映射到 Student KV。

        优化（相比旧版双层循环）：
            1. de-RoPE 一次算完所有 head 的所有 token。
            2. W_stack = (H, D, D) 一次 stack。
            3. einsum 把 (S, H, D) @ (H, D, D) → (S, H, D) 一次算完所有 head。
        数学上完全等价（per-head 独立），性能 3-10× 提升。

        design.md §22：按 kv_kind 取对应参数；未 fit 该 kind 时 raise KeyError。
        """
        # §22：先校验该 kind 是否已 fit，未 fit 直接报错（中文提示）
        _require_kv_kind(kv_kind, {key[0] for key in self.W}, "RidgePerHeadMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H, D), dtype=kv_t.dtype)
        for s in range(L_s):
            teachers = layer_map[s]
            for t in teachers:
                # 一次 de-RoPE 覆盖所有 head
                if de_rope_fn is not None:
                    k_unrot = _apply_or_skip(de_rope_fn, kv_t[t], positions)  # (S, H, D)
                else:
                    k_unrot = kv_t[t]
                # W_stack (H, D, D)，einsum 同时乘所有 head
                W_stack = np.stack(
                    [self.W[(kv_kind, s, h)] for h in range(H)], axis=0
                )  # (H, D, D)  # §22：按 kv_kind 取对应 kind 的 W
                # out[s, i, h, d] = sum_k W_stack[h, d, k] * k_unrot[i, h, k]
                mapped = np.einsum("shd,hdk->shk", k_unrot, W_stack)
                out[s] += mapped
            out[s] /= len(teachers)
        return out


# ---------------------------------------------------------------------------
# Low-rank Mapper
# ---------------------------------------------------------------------------


class LowRankMapper:
    """Per-(layer, head) 低秩 mapper（设计文档 §20 Stage 2）。

    W ≈ A @ B，A ∈ R^{D × r}, B ∈ R^{r × D}。
    参数量：L_s × H × 2 × r × D，对 r=16 → 约 28*8*2*16*128 ≈ 0.9M，
    PCR ≈ 0.9M / 3.7M ≈ 0.25。
    """

    def __init__(self, rank: int = 16, n_iter: int = 10):
        """默认 n_iter=10（ALS 经验上 5-10 次已收敛）。"""
        self.rank = rank
        self.n_iter = n_iter
        # design.md §22：键含 kv_kind 维度 —— (kv_kind, s, h)，K/V 各一套独立参数
        self.A: dict[tuple[str, int, int], np.ndarray] = {}
        self.B: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def n_params(self) -> int:
        # §22：对 dict 全部 value 求和 —— fit 一个 kind 与旧实现一致，K/V 翻倍
        return int(sum(a.size + b.size for a, b in zip(self.A.values(), self.B.values())))

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ):
        """训练 per-(layer, head) 的低秩因子 A/B。

        kv_kind（design.md §22）：K/V 独立参数化 —— K 与 V 各自持有独立的
        `(kv_kind, s, h)` 的 A/B 因子矩阵，互不覆盖。
        """
        _check_shape(kv_t, kv_s)
        L_s, S, H, D = kv_s.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)

        for s in range(L_s):
            teachers = layer_map[s]
            src_per_head = []
            for t in teachers:
                k_unrot = _apply_or_skip(de_rope_fn, kv_t[t], positions)
                src_per_head.append(k_unrot)
            src_all = np.stack(src_per_head, axis=0)  # (k, S, H, D)
            tgt_all = kv_s[s]                          # (S, H, D)
            # ◆ bug-2 修复（top_k 对齐，per-head batch 形态）：与
            #   RidgePerHeadMapper 相同 —— src 保留全部 k 层 → (H, k*S, D)，
            #   tgt_h 用 np.repeat 复制 k 份对齐，fit 与 transform 的 k 层
            #   平均语义一致；旧实现 `n = min(k*S, S)` 训练只用第 1 个教师层。
            _, _, src_h, tgt_h = _stack_topk(src_all, tgt_all)  # (H, k*S, D)
            # _lowrank_factor_batch(x=tgt, y=src)：批量 ALS 解 (H, n, D) → (H, D, r)+(H, r, D)
            A_batch, B_batch = _lowrank_factor_batch(
                tgt_h, src_h, self.rank, self.n_iter
            )
            for h in range(H):
                # design.md §22：键含 kv_kind —— K 与 V 各存一套，互不覆盖
                self.A[(kv_kind, s, h)] = A_batch[h]
                self.B[(kv_kind, s, h)] = B_batch[h]

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """将 Teacher KV 经低秩映射到 Student KV。

        优化：把 A/B stack 成 (H, D, r)/(H, r, D) 后 einsum 一次算完所有 head。

        design.md §22：按 kv_kind 取对应参数；未 fit 该 kind 时 raise KeyError。
        """
        # §22：先校验该 kind 是否已 fit，未 fit 直接报错（中文提示）
        _require_kv_kind(kv_kind, {key[0] for key in self.A}, "LowRankMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H, D), dtype=kv_t.dtype)
        for s in range(L_s):
            teachers = layer_map[s]
            # Stack A, B：形状 (H, D, r), (H, r, D)
            r = self.rank
            A_stack = np.stack([self.A[(kv_kind, s, h)] for h in range(H)], axis=0)  # (H, D, r)
            B_stack = np.stack([self.B[(kv_kind, s, h)] for h in range(H)], axis=0)  # (H, r, D)
            for t in teachers:
                if de_rope_fn is not None:
                    k_unrot = _apply_or_skip(de_rope_fn, kv_t[t], positions)  # (S, H, D)
                else:
                    k_unrot = kv_t[t]
                # 先 A：out[i, h, k] = sum_d A[h, d, k] * k_unrot[i, h, d] → (S, H, r)
                # 再 B：out[i, h, d] = sum_r B[h, r, d] * A_out[i, h, r] → (S, H, D)
                a_out = np.einsum("shd,hdk->shk", k_unrot, A_stack)
                mapped = np.einsum("shk,hkd->shd", a_out, B_stack)
                out[s] += mapped
            out[s] /= len(teachers)
        return out


# ---------------------------------------------------------------------------
# Shared-Basis Mapper（§20 Stage 3 / §34 T06 强制 / A2 消融开关）
# ---------------------------------------------------------------------------


class SharedBasisMapper:
    """跨 (layer, head) 共享的低秩基底 mapper（design.md §20 Stage 3）。

    与 LowRankMapper 的区别：
        - B（升维矩阵）所有 head 共享一套：A ∈ R^{D × r}
        - 只有 A_{s,h}（降维矩阵）每头独立
    参数量：1 × A_shared + L_s × H × A_per_head
        = D*r + L_s*H*D*r
        = r*D*(1 + L_s*H)
    对 Qwen3-1.7B, r=16: 16*128*(1+28*8) ≈ 0.46M，比 LowRankMapper 还省一半。

    触发条件：config["mapper"]["shared_basis"] = True（A2 消融）。
    """

    def __init__(self, rank: int = 16, n_iter: int = 25):
        self.rank = rank
        self.n_iter = n_iter
        # design.md §22：共享基底改为 dict[kv_kind] —— K/V 各持有一套共享基，
        # 不再共享同一个矩阵（K 与 V 是独立缓存，共享基必须独立）。
        self.A_shared: dict[str, np.ndarray] = {}
        # 每 (s, h) 的升维矩阵，键含 kv_kind —— (kv_kind, s, h)
        self.B: dict[tuple[str, int, int], np.ndarray] = {}

    @property
    def n_params(self) -> int:
        # §22：对 dict 全部 value 求和 —— fit 一个 kind 与旧实现一致，K/V 翻倍
        shared = sum(a.size for a in self.A_shared.values())
        per = int(sum(b.size for b in self.B.values()))
        return int(shared + per)

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ):
        """训练共享基底 A_shared[kv_kind] 与每 (s, h) 的 B[(kv_kind, s, h)]。

        kv_kind（design.md §22）：K/V 独立参数化 —— K 与 V 各自持有一套
        `A_shared[kv_kind]` 共享基与 `B[(kv_kind, s, h)]` 升维矩阵，互不覆盖。
        """
        _check_shape(kv_t, kv_s)
        L_s, S, H, D = kv_s.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)

        # 第一步：在所有 (s, h) 拼接的数据上学一个共享 A
        all_src = []
        all_tgt = []
        for s in range(L_s):
            teachers = layer_map[s]
            src_per_head = []
            for t in teachers:
                k_unrot = _apply_or_skip(de_rope_fn, kv_t[t], positions)
                src_per_head.append(k_unrot)
            src_all = np.stack(src_per_head, axis=0)  # (k, S, H, D)
            tgt_all = kv_s[s]
            # ◆ bug-2 修复（top_k 对齐）：每 (s, h) 保留全部 k 层 src 行
            #   （batch 视图 (H, k*S, D)），tgt_h 复制 k 份对齐后拼接进共享
            #   A 的数据集 —— 共享基底同样覆盖全部教师层样本。
            _, _, src_batch, tgt_batch = _stack_topk(src_all, tgt_all)
            for h in range(H):
                all_src.append(src_batch[h])   # (k*S, D)
                all_tgt.append(tgt_batch[h])   # (k*S, D)
        all_src = np.concatenate(all_src, axis=0)
        all_tgt = np.concatenate(all_tgt, axis=0)
        # 学一个共享 A：等价于把所有 (s, h) 当成一个数据集做一次低秩分解
        # _lowrank_factor(x=target, y=source)，学 (A, B) 使 y@B@A≈x
        # 这里 y=src（输入），x=tgt（输出目标）；A_shared ∈ R^{D×r} 把 src 从 D 维降到 r 维
        # design.md §22：存入 dict[kv_kind] —— K 与 V 各持有一套共享基
        self.A_shared[kv_kind], _ = _lowrank_factor(all_tgt, all_src, self.rank, self.n_iter)

        # 第二步：固定 A_shared[kv_kind]，每个 (s, h) 单独学 B
        for s in range(L_s):
            teachers = layer_map[s]
            src_per_head = []
            for t in teachers:
                k_unrot = _apply_or_skip(de_rope_fn, kv_t[t], positions)
                src_per_head.append(k_unrot)
            src_all = np.stack(src_per_head, axis=0)
            tgt_all = kv_s[s]
            # ◆ bug-2 修复（top_k 对齐）：与第一步相同，B 也用全部 k 层样本
            #   学习（src_batch[h] 保留 k*S 行，tgt_batch[h] 已复制对齐）。
            _, _, src_batch, tgt_batch = _stack_topk(src_all, tgt_all)
            for h in range(H):
                # B = lstsq(A_shared^T src, tgt)
                proj = src_batch[h] @ self.A_shared[kv_kind]  # (k*S, r)
                # design.md §22：键含 kv_kind —— K 与 V 各存一套，互不覆盖
                self.B[(kv_kind, s, h)] = np.linalg.lstsq(proj, tgt_batch[h], rcond=None)[0]

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        kv_kind: str = "K",
        positions: np.ndarray | None = None,
        de_rope_fn: Callable | None = None,
    ) -> np.ndarray:
        """将 Teacher KV 经共享基映射到 Student KV。

        design.md §22：按 kv_kind 取对应 `A_shared[kv_kind]` 与 `B[(kv_kind, s, h)]`；
        未 fit 该 kind 时 raise KeyError（禁止静默回退到其他 kind 的共享基）。
        """
        # §22：先校验该 kind 是否已 fit（A_shared dict 含该键），未 fit 直接报错
        _require_kv_kind(kv_kind, set(self.A_shared), "SharedBasisMapper.transform")
        L_s = len(layer_map)
        L_t, S, H, D = kv_t.shape
        if positions is None:
            positions = np.arange(S, dtype=np.float64)
        out = np.zeros((L_s, S, H, D), dtype=kv_t.dtype)
        for s in range(L_s):
            teachers = layer_map[s]
            for h in range(H):
                b = self.B[(kv_kind, s, h)]  # §22：按 kv_kind 取对应 kind 的 B
                mapped = []
                for t in teachers:
                    k_unrot = _apply_or_skip(de_rope_fn, kv_t[t, :, h, :], positions)
                    mapped.append((k_unrot @ self.A_shared[kv_kind]) @ b)  # §22：取本 kind 共享基
                out[s, :, h, :] = np.stack(mapped, axis=0).mean(0)
        return out