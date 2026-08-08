"""§12 G2 Mismatched Head / Dimension Projection Mapper。

═══════════════════════════════════════════════════════════════════════════════
当 Teacher 与 Student 的 KV 配置不一致时（§12 G2），不能直接 per-(s, h) 学
Ridge。需要先做投影对齐：

    1. Head Projection P_H    (H_T → H_S)
       把 Teacher 的 H_T 个 kv-head 映射到 Student 的 H_S 个 head。
       典型策略：均值池化 / 线性层 / 固定模式（如 H_T=8 → H_S=4 时两两合并）。

    2. Dimension Projection P_d    (D_T → D_S)
       head_dim 不一致时（如 D_T=128 → D_S=64），需要学一个 (D_T × D_S) 矩阵。
       典型策略：截断 / 线性投影 / PCA 降维。

本模块实现 `MismatchedHeadMapper`：同时支持 head 数和 head_dim 不同的场景。
设计思路：
    Step 1: 对 Teacher 每层做 P_d 投影到 Student head_dim
    Step 2: 用 P_H 把 H_T 个 head 映射到 H_S 个 head
    Step 3: 然后按匹配情形用 Ridge / LowRank 完成 token-维映射

§19 B3 强制 per-head 训练：MismatchedHeadMapper 默认 head 维已经对齐，
因此后续可直接用 RidgePerHeadMapper。

使用示例：
    mapper = MismatchedHeadMapper(
        n_t_layers=36, n_s_layers=28,
        n_t_heads=8, n_s_heads=8,        # 头数相同，不需要 P_H
        d_t=128, d_s=128,                 # head_dim 相同，不需要 P_d
        p_h_strategy="mean",
        p_d_strategy="truncate",
    )
    mapper.fit(kv_t, kv_s, layer_map, use_inner="ridge")
    pred = mapper.transform(kv_t, layer_map)

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from .math import (
    LowRankMapper,
    RidgePerHeadMapper,
    SharedBasisMapper,
    _check_shape,
    _lowrank_factor_batch,
    _ridge_closed_form,
    _ridge_closed_form_batch,
)


class HeadProjection:
    """§12 P_H: Head 数 H_T → H_S 的投影。

    支持三种策略：
        - "mean":  把 H_T 个 head 平均池化到 H_S 个 head（H_T > H_S）
        - "repeat": 把每个 head 复制到目标位置（H_T < H_S）
        - "linear": 学一个 (H_T × H_S) 矩阵（最灵活，需要校准数据）

    参数：
        n_t_heads: Teacher head 数
        n_s_heads: Student head 数
        strategy: "mean" / "repeat" / "linear"
        seed: linear 策略的随机种子
    """

    def __init__(
        self,
        n_t_heads: int,
        n_s_heads: int,
        strategy: str = "mean",
        seed: int = 0,
    ):
        assert n_t_heads > 0 and n_s_heads > 0
        self.n_t = n_t_heads
        self.n_s = n_s_heads
        self.strategy = strategy
        self.W: np.ndarray | None = None  # linear 策略下存 (H_T, H_S)
        if strategy == "linear":
            rng = np.random.default_rng(seed)
            self.W = rng.standard_normal((n_t_heads, n_s_heads)) / np.sqrt(
                max(n_t_heads, n_s_heads)
            )

    def fit(self, kv_t_per_head: np.ndarray, kv_s_per_head: np.ndarray):
        """linear 策略：用数据学 W。

        kv_t_per_head: (n_samples, H_T)  — Teacher 每个 head 在样本上的某个标量
        kv_s_per_head: (n_samples, H_S)
        学 W: (H_T, H_S) 使 kv_t_per_head @ W ≈ kv_s_per_head
        """
        if self.strategy != "linear":
            return
        if kv_t_per_head.shape[1] != self.n_t or kv_s_per_head.shape[1] != self.n_s:
            raise ValueError(
                f"per-head 数据形状不匹配：teacher={kv_t_per_head.shape}, "
                f"student={kv_s_per_head.shape}"
            )
        # _ridge_closed_form(x=target, y=source)，求 W 使 y @ W ≈ x
        self.W = _ridge_closed_form_batch(
            kv_s_per_head.reshape(1, *kv_s_per_head.shape),
            kv_t_per_head.reshape(1, *kv_t_per_head.shape),
            lam=1e-3,
        )[0]

    def project(self, k_t: np.ndarray) -> np.ndarray:
        """把 (..., H_T, D) 的 Teacher head 投影到 (..., H_S, D)。

        注意：mean/repeat 是 per-token 的 head 维变换；
        linear 是学到的矩阵，作用于 head 维。
        """
        h = k_t.shape[-2]
        d = k_t.shape[-1]
        if h != self.n_t:
            raise ValueError(f"Teacher head 数 ({h}) 与 P_H 期望 ({self.n_t}) 不一致")

        if self.strategy == "mean":
            # 把 H_T 个 head 按比例合并到 H_S 个 head
            if self.n_t == self.n_s:
                return k_t
            # 每组 ceil(H_T/H_S) 或 floor
            ratio = self.n_t / self.n_s
            out = np.zeros((*k_t.shape[:-2], self.n_s, d), dtype=k_t.dtype)
            for i in range(self.n_s):
                start = int(round(i * ratio))
                end = int(round((i + 1) * ratio))
                end = min(max(end, start + 1), self.n_t)
                out[..., i, :] = k_t[..., start:end, :].mean(axis=-2)
            return out
        if self.strategy == "repeat":
            if self.n_t == self.n_s:
                return k_t
            # 把每个 Teacher head 复制 (n_s / n_t) 次
            repeat = self.n_s // self.n_t
            extra = self.n_s - repeat * self.n_t
            blocks = [k_t] * repeat
            if extra > 0:
                blocks.append(k_t[..., :extra, :])
            return np.concatenate(blocks, axis=-2)
        if self.strategy == "linear":
            # k_t: (..., H_T, D), W: (H_T, H_S)
            # 沿 head 维做线性组合：(H_T, D) @ (H_T, H_S) → 不对
            # 正确做法：把 head 维当样本维，W 作用在 head 维
            # (..., H_T, D) → (..., H_T, D)，对最后一维 H_T 做 W @ k
            # (H_T, D) @ (H_T, H_S) 不行：D 维不一样
            # 实际做法：对每个 token，独立做 (1, H_T) @ (H_T, H_S) = (1, H_S)
            return np.einsum("...hd,ts->...sd", k_t, self.W)
        raise ValueError(f"未知 strategy: {self.strategy}")


class DimensionProjection:
    """§12 P_d: head_dim D_T → D_S 的投影。

    支持策略：
        - "truncate": 截断到 min(D_T, D_S)（D_T ≥ D_S 时）
        - "pad":       零填充到 max(D_T, D_S)（D_T < D_S 时）
        - "linear":    学一个 (D_T, D_S) 线性矩阵
        - "pca":       PCA 投影（需要校准数据）

    参数：
        d_t: Teacher head_dim
        d_s: Student head_dim
        strategy: "truncate" / "pad" / "linear"
    """

    def __init__(self, d_t: int, d_s: int, strategy: str = "linear"):
        self.d_t = d_t
        self.d_s = d_s
        self.strategy = strategy
        self.W: np.ndarray | None = None
        if strategy == "linear":
            self.W = np.zeros((d_t, d_s))

    def fit(self, kv_t: np.ndarray, kv_s: np.ndarray):
        """linear 策略：用 Teacher 与 Student 同位置 token 学 (D_T, D_S) 矩阵。

        kv_t: (..., D_T)  Teacher 投影前
        kv_s: (..., D_S)  Student 对应位置
        """
        if self.strategy != "linear":
            return
        if kv_t.shape[-1] != self.d_t:
            raise ValueError(f"Teacher D ({kv_t.shape[-1]}) != {self.d_t}")
        if kv_s.shape[-1] != self.d_s:
            raise ValueError(f"Student D ({kv_s.shape[-1]}) != {self.d_s}")
        # 把前导维 flatten 成 (n, D)
        x = kv_s.reshape(-1, self.d_s)
        y = kv_t.reshape(-1, self.d_t)
        # 学 W 使 y @ W ≈ x → W ∈ R^{D_T × D_S}
        self.W = _ridge_closed_form(x, y, lam=1e-3)

    def project(self, k: np.ndarray) -> np.ndarray:
        """把 (..., D_T) 投影到 (..., D_S)。"""
        if k.shape[-1] != self.d_t:
            raise ValueError(f"输入 D ({k.shape[-1]}) != {self.d_t}")

        if self.strategy == "truncate":
            d = min(self.d_t, self.d_s)
            out = k[..., :d]
            if self.d_s > d:
                pad = np.zeros((*k.shape[:-1], self.d_s - d), dtype=k.dtype)
                return np.concatenate([out, pad], axis=-1)
            return out
        if self.strategy == "pad":
            if self.d_t >= self.d_s:
                return k[..., : self.d_s]
            pad = np.zeros((*k.shape[:-1], self.d_s - self.d_t), dtype=k.dtype)
            return np.concatenate([k, pad], axis=-1)
        if self.strategy == "linear":
            assert self.W is not None, "linear 策略需先 fit()"
            return k @ self.W
        raise ValueError(f"未知 strategy: {self.strategy}")


class MismatchedHeadMapper:
    """§12 G2 完整 Mapper：P_d → P_H → Ridge。

    适用场景：
        Teacher 与 Student 的 H 或 D 不一致。
        完整流程：
            1. kv_t (L_t, S, H_T, D_T) → P_d → (L_t, S, H_T, D_S)
            2. → P_H → (L_t, S, H_S, D_S)
            3. → RidgePerHeadMapper → (L_s, S, H_S, D_S) = kv_s

    参数：
        n_t_layers, n_s_layers: Teacher/Student 层数
        n_t_heads, n_s_heads:   Teacher/Student head 数
        d_t, d_s:               Teacher/Student head_dim
        p_h_strategy:           P_H 策略 (mean/repeat/linear)
        p_d_strategy:           P_d 策略 (truncate/pad/linear)
    """

    def __init__(
        self,
        n_t_layers: int,
        n_s_layers: int,
        n_t_heads: int,
        n_s_heads: int,
        d_t: int,
        d_s: int,
        p_h_strategy: str = "mean",
        p_d_strategy: str = "truncate",
        seed: int = 0,
    ):
        self.n_t_layers = n_t_layers
        self.n_s_layers = n_s_layers
        self.n_t_heads = n_t_heads
        self.n_s_heads = n_s_heads
        self.d_t = d_t
        self.d_s = d_s
        self.p_h = HeadProjection(n_t_heads, n_s_heads, strategy=p_h_strategy, seed=seed)
        self.p_d = DimensionProjection(d_t, d_s, strategy=p_d_strategy)
        self.inner: RidgePerHeadMapper | LowRankMapper | None = None

    def fit(
        self,
        kv_t: np.ndarray,
        kv_s: np.ndarray,
        layer_map: list[list[int]],
        use_inner: str = "ridge",
        rank: int = 16,
        positions: np.ndarray | None = None,
        de_rope_fn=None,
    ):
        """训练 P_d（若 linear）+ P_H（若 linear）+ 内层 mapper。"""
        # 先 fit linear 策略的投影（如果启用）
        if self.p_d.strategy == "linear":
            # 把所有层的所有 token flatten 学投影
            kv_t_flat = kv_t.reshape(-1, self.d_t)
            # 拿 Student 的目标维度对应位置
            kv_s_match = kv_s[: kv_t.shape[0]].reshape(-1, self.d_s)
            n = min(kv_t_flat.shape[0], kv_s_match.shape[0])
            self.p_d.fit(kv_t_flat[:n], kv_s_match[:n])
        if self.p_h.strategy == "linear":
            # 用某 token 维的标量聚合做 linear head 投影
            kv_t_h = kv_t[..., : self.d_s].mean(axis=-1)  # (L_t, S, H_T)
            kv_s_h = kv_s[..., :].mean(axis=-1)  # (L_s, S, H_S)
            # 对齐 L 维
            n = min(kv_t_h.shape[0], kv_s_h.shape[0])
            self.p_h.fit(kv_t_h[:n].reshape(-1, self.n_t_heads),
                          kv_s_h[:n].reshape(-1, self.n_s_heads))

        # 投影 Teacher 到 (..., H_S, D_S)
        kv_t_proj = self._project_teacher(kv_t, positions, de_rope_fn)

        # 内层 mapper
        if use_inner == "ridge":
            self.inner = RidgePerHeadMapper(lam=1e-3)
        elif use_inner == "lowrank":
            self.inner = LowRankMapper(rank=rank)
        else:
            raise ValueError(f"未知 use_inner: {use_inner}")
        self.inner.fit(kv_t_proj, kv_s, layer_map,
                       positions=positions, de_rope_fn=de_rope_fn)

    def _project_teacher(
        self,
        kv_t: np.ndarray,
        positions: np.ndarray | None = None,
        de_rope_fn=None,
    ) -> np.ndarray:
        """应用 P_d + P_H + de-RoPE（若启用）。

        流程：先 de-RoPE（如果启用了 de_rope_fn 且是 linear 投影前的最后一维 = D_T），
        然后 P_d 把 D_T → D_S，再 P_H 把 H_T → H_S。
        """
        L_t, S, H_T, D_T = kv_t.shape
        # 1) de-RoPE（若启用）
        if de_rope_fn is not None and positions is not None:
            k = kv_t
            k = de_rope_fn(k, positions)
        else:
            k = kv_t
        # 2) P_d: (L_t, S, H_T, D_T) → (L_t, S, H_T, D_S)
        if self.d_t != self.d_s:
            k = self.p_d.project(k)
        # 3) P_H: (L_t, S, H_T, D_S) → (L_t, S, H_S, D_S)
        if self.n_t_heads != self.n_s_heads:
            k = self.p_h.project(k)
        return k

    def transform(
        self,
        kv_t: np.ndarray,
        layer_map: list[list[int]],
        positions: np.ndarray | None = None,
        de_rope_fn=None,
    ) -> np.ndarray:
        assert self.inner is not None, "必须先 fit()"
        kv_t_proj = self._project_teacher(kv_t, positions, de_rope_fn)
        return self.inner.transform(kv_t_proj, layer_map,
                                   positions=positions, de_rope_fn=de_rope_fn)