"""MismatchedHeadMapper（§12 G2）单元测试。

bug-8 回归：MismatchedHeadMapper._project_teacher 已经把 Teacher K de-RoPE 到
unrotated 空间，却又把 de_rope_fn 传给 inner mapper → 第二次 de-RoPE 在
unrotated 表示上叠加位置相关旋转，破坏值。
"""
from __future__ import annotations

import numpy as np

from apcs.mapper.math import RidgePerHeadMapper
from apcs.mapper.mismatched import MismatchedHeadMapper
from apcs.rope.runner import _rope_pairs, apply_rope, de_rope


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """展平后计算余弦相似度（held-out 保留度的统一度量）。"""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _mismatched_toy(n_t=3, n_s=2, S=32, seed=0, noise=0.01, theta=10.0):
    """构造有结构 toy：共享 latent Z + 每层不同线性变换 W_t / W_s。

    teacher: (n_t, S, H_T=4, D=16) —— 已 apply RoPE（Teacher 旋转空间）
    student: (n_s, S, H_S=2, D=16) —— Student 端目标
    student head h = mean(teacher heads 2h, 2h+1) → P_H="mean" 可精确对齐。
    """
    rng = np.random.default_rng(seed)
    D, H_T, H_S = 16, 4, 2
    Z_t = rng.standard_normal((S, H_T, D))
    Z_s = np.stack([Z_t[:, 2 * h : 2 * h + 2, :].mean(axis=1) for h in range(H_S)], axis=1)
    Wt = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_t)]
    Ws = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_s)]
    kv_t_plain = np.stack(
        [(Z_t @ Wt[l]) + noise * rng.standard_normal(Z_t.shape) for l in range(n_t)]
    )
    kv_s = np.stack(
        [(Z_s @ Ws[l]) + noise * rng.standard_normal(Z_s.shape) for l in range(n_s)]
    ).astype(np.float32)
    inv_freq = _rope_pairs(D, theta=theta)
    positions = np.arange(S, dtype=np.float64)
    kv_t = np.stack(
        [
            apply_rope(kv_t_plain[l].astype(np.float64), positions, inv_freq).astype(np.float32)
            for l in range(n_t)
        ]
    )
    return kv_t, kv_s, inv_freq


def _de_rope_layer_aware(k: np.ndarray, positions: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """de-RoPE 兼容 (L,S,H,D) 整批 与 (S,H,D) 单层 两种调用形状。"""
    if k.ndim == 4:
        return np.stack([de_rope(k[l], positions, inv_freq) for l in range(k.shape[0])])
    return de_rope(k, positions, inv_freq)


def test_bug8_single_de_rope_beats_double_de_rope():
    """bug-8 回归：Teacher K 只允许 de-RoPE 一次。

    Given: 有结构 toy（共享 latent + RoPE 过的 Teacher K，H_T=4 → H_S=2），
           train / held-out 位置各占一半（S=32，前 16 训练、后 16 预测）
    When:  MismatchedHeadMapper.fit 用前 16 个位置训练，transform 后 16 个位置
    Then:  单次 de-RoPE（公共 API）把 Train/Test 放到同一 unrotated 帧，
           per-(s,h) Ridge W 位置无关 → held-out cosine 显著 > 0.5；
           双重 de-RoPE（模拟 bug-8）残留位置相关旋转 → 显著劣于单次。
    """
    kv_t, kv_s, inv_freq = _mismatched_toy(seed=0)
    layer_map = [[0], [1]]
    S = kv_t.shape[1]
    train_pos = np.arange(S // 2)
    test_pos = np.arange(S // 2, S)
    de_rope_fn = lambda k, p: _de_rope_layer_aware(k, p, inv_freq)

    # ---- 修复路径（公共 API）：_project_teacher 单次 de-RoPE，inner 不再 de-RoPE ----
    m = MismatchedHeadMapper(
        n_t_layers=3,
        n_s_layers=2,
        n_t_heads=4,
        n_s_heads=2,
        d_t=16,
        d_s=16,
        p_h_strategy="mean",
        p_d_strategy="truncate",
    )
    m.fit(kv_t[:, train_pos], kv_s[:, train_pos], layer_map,
          positions=train_pos, de_rope_fn=de_rope_fn)
    pred = m.transform(kv_t[:, test_pos], layer_map, positions=test_pos,
                       de_rope_fn=de_rope_fn)
    cos_single = _cosine(pred, kv_s[:, test_pos])

    # ---- 模拟 bug-8：inner mapper 再收到 de_rope_fn → 第二次 de-RoPE ----
    m2 = MismatchedHeadMapper(
        n_t_layers=3,
        n_s_layers=2,
        n_t_heads=4,
        n_s_heads=2,
        d_t=16,
        d_s=16,
        p_h_strategy="mean",
        p_d_strategy="truncate",
    )
    proj_tr = m2._project_teacher(kv_t[:, train_pos], train_pos, de_rope_fn)
    proj_te = m2._project_teacher(kv_t[:, test_pos], test_pos, de_rope_fn)
    inner = RidgePerHeadMapper(lam=1e-3)
    inner.fit(proj_tr, kv_s[:, train_pos], layer_map,
              positions=train_pos, de_rope_fn=de_rope_fn)
    pred_double = inner.transform(proj_te, layer_map, positions=test_pos,
                                  de_rope_fn=de_rope_fn)
    cos_double = _cosine(pred_double, kv_s[:, test_pos])

    assert cos_single > 0.5, (
        f"单次 de-RoPE 应保留跨位置信号：cos={cos_single:.3f} > 0.5；"
        f"若接近 0 说明 inner mapper 被二次 de-RoPE（bug-8 未修复）"
    )
    assert cos_double < cos_single - 0.3, (
        f"双重 de-RoPE 应显著劣于单次：double={cos_double:.3f} vs single={cos_single:.3f}"
    )
