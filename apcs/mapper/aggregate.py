"""校准样本聚合：把 n_calib 次 `fit()` 合并为一次聚合 fit（§32/§33 校准）。

◆ bug-3 修复：旧 T04/T05/T06 循环 `for i in range(n_calib): ridge.fit(...)`
  每次 fit 覆盖 W → 最终只保留最后一组样本的影响（其他样本白算）。
  修复：先收集全部校准样本，再用本模块**聚合**后只 fit 一次。

两种数学等价的写法（任务 §32 方案 A/B）：
    方案 A（concat）：把全部样本沿 S 维拼接成一个大 KV，只调一次 fit。
        W = (Y^T Y + λI)^{-1} Y^T X，Y = [Y_1; ...; Y_n]（行拼接）
        → G = Σ_i Y_i^T Y_i，B = Σ_i Y_i^T X_i —— 这正是方案 B。
    方案 B（Gram 聚合，本模块默认）：在每个 (layer, head) 上累加 Gram 矩阵
        G[(s,h)] = Σ_i src_i^T src_i    （(D,D)，不随样本数增长）
        B[(s,h)] = Σ_i src_i^T tgt_i    （(D,D)）
        W[(s,h)] = solve(G[(s,h)] + λI, B[(s,h)])
    内存 O(L_s × H × D²)，与 n_calib 无关 —— §32 目标"100-500 样本"下
    concat 方案（n_calib=128, S=2048 时约 1GB）在真实 GPU 场景不可行，
    但两种方案数学上逐位等价（见 test_calibration_aggregate_uses_all_samples
    的等价性断言）。

约束（与 math.py 的 fit 完全一致，不重复实现）：
    - de-RoPE 由调用方传入 de_rope_fn（§23 强制路径）
    - layer_map[s] 缺省回退 `[int(round(s * L_t / L_s))]`（与 fit 相同）
    - top_k 对齐：src 保留全部 k 层，tgt 用 np.tile/np.repeat 复制 k 份
      （bug-2 修复语义，见 math._stack_topk）

支持两种 mapper：
    - RidgeMapper：per-layer 语义（bug-1 修复），W[(kv_kind, s, 0)]，flat 视图
    - RidgePerHeadMapper：per-(s, h) 语义，W[(kv_kind, s, h)]，batch 视图
    其余类型（LowRank / SharedBasis，ALS 无法 Gram 聚合）请用
    concat_kv_samples 后一次 fit，见 runner.py T06 的注释。

design.md §22（K/V 独立参数化）：`fit_ridge_aggregate` 接受 `kv_kind` 参数，
Gram 累加到 `(kv_kind, s, h)`（RidgeMapper 用 `(kv_kind, s, 0)`）键下 ——
K 与 V 各持有一套独立参数矩阵，先 fit("K") 再 fit("V") 互不覆盖。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import numpy as np

from .math import (
    RidgeMapper,
    RidgePerHeadMapper,
    _apply_or_skip,
    _stack_topk,
)


def concat_kv_samples(
    samples: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """方案 A：把多个校准样本沿 S 维拼接成一个"大 KV"。

    参数：
        samples: 可迭代的 (kv_t, kv_s) 对，每个 (L_t, S, H, D) / (L_s, S, H, D)
    返回：
        (kv_t_big, kv_s_big)：沿 S 维拼接，(L_t, n*S, H, D) / (L_s, n*S, H, D)

    配合一次 fit 时，positions 需用 np.tile(positions, n) 拼接（S 扩展）；
    拼接不改变层/头/维度，只增加样本行 —— 这正是方案 A 的语义。

    拼接语义：n 个 (kv_t, kv_s) 对沿 S 维首尾相接成一个"长序列"，数学上
    等价于"把全部样本当作一条超长序列做一次 fit"。Ridge 闭式解由 Gram 累加
    证明两种写法逐位等价；ALS 类 mapper（LowRank / SharedBasis）无闭式解，
    concat 是唯一严格写法。
    内存注意：拼接后行数为 n*S，随样本数线性增长 —— 样本多/序列长时请用
    fit_ridge_aggregate（方案 B，内存 O(L_s×H×D²) 与样本数无关）。
    """
    pairs = list(samples)
    # 边界：空样本集无意义（后续 fit 会退化），显式报错而非静默返回空 KV
    if not pairs:
        raise ValueError("concat_kv_samples: samples 不能为空")
    kv_t_list = [kv_t for kv_t, _ in pairs]
    kv_s_list = [kv_s for _, kv_s in pairs]
    # axis=1 = S 维：层/头/维度保持不变，只把多个样本的序列行首尾相接。
    # 各样本 S/H/D 不一致会在 np.concatenate 处抛 ValueError（G1 要求几何一致）
    return (
        np.concatenate(kv_t_list, axis=1),
        np.concatenate(kv_s_list, axis=1),
    )


def fit_ridge_aggregate(
    mapper: RidgeMapper | RidgePerHeadMapper,
    samples: Iterable[tuple[np.ndarray, np.ndarray]],
    layer_map: list[list[int]],
    kv_kind: str = "K",
    positions: np.ndarray | None = None,
    de_rope_fn: Callable[..., np.ndarray] | None = None,
) -> None:
    """方案 B：Gram 聚合后一次 fit，原地填充 mapper.W（与 fit() 等价）。

    参数：
        mapper: RidgeMapper（per-layer）或 RidgePerHeadMapper（per-head）；
            其他类型 raise TypeError（ALS 无法按 Gram 聚合）。
        samples: 全部校准样本 (kv_t, kv_s)（共享同一 W_t/W_s —— 同一
            Teacher/Student 模型对不同 prompt 的真实语义）。
        layer_map: 长度 L_s 的 list，每个元素是 Student 层对应的 Teacher 层索引
        kv_kind: design.md §22 —— K/V 独立参数化。本次聚合 fit 的 Gram 累加到
            `(kv_kind, s, h)`（RidgeMapper 为 `(kv_kind, s, 0)`）键下；
            K 与 V 各持有一套独立参数矩阵，先 fit("K") 再 fit("V") 互不覆盖。
        positions / de_rope_fn: 与 math.fit 同语义。

    数学：Ridge 闭式解 W = (Y^T Y + λI)^{-1} Y^T X 对样本是可加的：
        把样本行拼接后 fit ⇒ G = Σ_i Y_i^T Y_i、B = Σ_i Y_i^T X_i。
    因此本函数与"concat 后一次 fit"逐位等价（测试断言 maxdiff < 1e-4）。
    """
    # 只实例化一次：传入 generator 时，先 list(samples)[0] 再遍历
    # samples 会把后续样本静默丢掉。
    pairs = list(samples)
    if not pairs:
        raise ValueError("fit_ridge_aggregate: samples 不能为空")
    kv_t0, kv_s0 = pairs[0]
    L_s, S, H, D = kv_s0.shape
    L_t = kv_t0.shape[0]
    if positions is not None and len(positions) != S:
        raise ValueError(
            f"positions 长度 {len(positions)} 与首个样本序列长度 {S} 不一致"
        )

    # per-head 语义（RidgePerHeadMapper）→ 每个 (s, h) 一套参数；
    # per-layer 语义（RidgeMapper）→ 每层一套参数（h 固定为 0）
    # TaskAwareRidgeMapper 的 Phase A 是 per-head ridge（.W/.lam 同构），
    # 鸭子类型识别以避免循环 import
    per_head = isinstance(mapper, RidgePerHeadMapper) or hasattr(mapper, "fit_task_aware")
    if not (per_head or isinstance(mapper, RidgeMapper)):
        raise TypeError(
            "fit_ridge_aggregate 仅支持 RidgeMapper / RidgePerHeadMapper / "
            f"TaskAwareRidgeMapper；got {type(mapper).__name__}。"
            "ALS/Affine 类 mapper 请用 fit_batch 或 concat_kv_samples"
            "拼接后一次 fit（见 runner.py T06 注释）。"
        )

    # 每 (s, h)（或 per-layer 的 (s, 0)）累加 Gram：
    #   G += src^T src（(D, D)），B += src^T tgt（(D, D)）
    # design.md §22：键含 kv_kind 维度 —— K/V 的 Gram 互不混淆
    G = {(kv_kind, s, h): np.zeros((D, D)) for s in range(L_s) for h in (range(H) if per_head else [0])}
    B = {(kv_kind, s, h): np.zeros((D, D)) for s in range(L_s) for h in (range(H) if per_head else [0])}

    # 逐样本累加 Gram：samples 共享同一 W_t/W_s（同一模型对），每个样本的
    # 贡献是独立的 (D,D) 外积累加 —— 与"concat 后一次 fit"逐位等价
    for kv_t, kv_s in pairs:
        # 真实 prompt 是变长的。positions=None 时按当前样本生成位置，
        # 避免为了 Gram 聚合而把所有样本裁到全局最短序列。
        sample_seq = int(kv_s.shape[1])
        sample_positions = (
            np.arange(sample_seq, dtype=np.float64) if positions is None else positions
        )
        if kv_t.shape[1] != sample_seq:
            raise ValueError(
                "Teacher/Student 样本在聚合前必须已裁成同一序列长度"
            )
        if positions is not None and len(positions) != sample_seq:
            raise ValueError(
                "显式 positions 只适用于定长样本；变长样本请传 positions=None"
            )
        for s in range(L_s):
            # Student 层 s 对应的 Teacher 层索引；越界时按比例回退（与 fit 一致）
            teachers = (
                layer_map[s]
                if s < len(layer_map)
                else [int(round(s * L_t / L_s))]
            )
            # 对每个教师层先 de-RoPE（§23 强制路径），再 stack 成 (k, S, H, D)
            src_layers = [
                _apply_or_skip(de_rope_fn, kv_t[t], sample_positions) for t in teachers
            ]
            src_all = np.stack(src_layers, axis=0)  # (k, S, H, D)
            if per_head:
                # per-head：batch 视图 (H, k*S, D)，要求 src/tgt head 数一致（G1 §10）
                _, _, src_h, tgt_h = _stack_topk(src_all, kv_s[s])  # (H, k*S, D)
                for h in range(H):
                    sh, th = src_h[h], tgt_h[h]
                    G[(kv_kind, s, h)] += sh.T @ sh
                    B[(kv_kind, s, h)] += sh.T @ th
            else:
                # per-layer：flat 视图 (k*S*H, D)，把 head 并入样本行
                src_f, tgt_f, _, _ = _stack_topk(src_all, kv_s[s])  # (k*S*H, D)
                G[(kv_kind, s, 0)] += src_f.T @ src_f
                B[(kv_kind, s, 0)] += src_f.T @ tgt_f

    # 一次求解全部 (s, h) 的 W = (G + λI)^{-1} B（键已含 kv_kind）。
    # λI 岭正则：保证 G+λI 对称正定、必可逆（数值稳定性；λ 与 math.fit 一致）
    eye = np.eye(D)
    for key, g in G.items():
        mapper.W[key] = np.linalg.solve(g + mapper.lam * eye, B[key])
