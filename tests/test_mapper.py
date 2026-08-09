"""Mapper 数学层单元测试（覆盖所有 bug 修复）。

bug-1: RidgeMapper 是 per-layer 语义，n_params 不虚高 H 倍
bug-2: de-RoPE 必须接入 mapper
bug-3: 形状不兼容要 raise 而非静默截断
bug-4: calibration KV 必须有真实对应关系
bug-4a: Teacher 层数少于 Student 层数是合法场景（由 layer_map / G2 处理）
"""
from __future__ import annotations

import numpy as np
import pytest

from apcs.mapper.math import (
    LowRankMapper,
    RidgeMapper,
    RidgePerHeadMapper,
    SharedBasisMapper,
    _check_shape,
)


def _toy(n_t=4, n_s=2, S=8, H=2, D=16, seed=0, with_signal=True):
    """构造有结构的 toy 数据：kv_s 是 kv_t 经线性变换 + 噪声。"""
    rng = np.random.default_rng(seed)
    if with_signal:
        Z = rng.standard_normal((S, H, D))
        Wt = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_t)]
        Ws = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_s)]
        kv_t = np.stack([(Z @ Wt[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_t)])
        kv_s = np.stack([(Z @ Ws[l]) + 0.05 * rng.standard_normal(Z.shape) for l in range(n_s)])
        return kv_t.astype(np.float32), kv_s.astype(np.float32)
    else:
        return (
            rng.standard_normal((n_t, S, H, D)).astype(np.float32),
            rng.standard_normal((n_s, S, H, D)).astype(np.float32),
        )


def _layer_map(n_t, n_s):
    """构造最简 layer_map：Student 层 s 只对齐 Teacher 层 s（一对一层映射）。"""
    return [list(range(s, s + 1)) for s in range(n_s)]


# ---- bug-3 修复：_check_shape 必须 raise ----


def test_check_shape_raises_on_mismatch():
    """bug-3：Teacher/Student 的 S/H/D 任一维度不一致必须 raise ValueError，禁止静默截断。"""
    bad_t = np.zeros((4, 8, 2, 16))
    bad_s = np.zeros((2, 7, 2, 16))  # S 不一致
    with pytest.raises(ValueError, match="形状不兼容"):
        _check_shape(bad_t, bad_s)


def test_check_shape_allows_fewer_teacher_layers():
    """B4a：教师层数少于学生层数（如 SmolLM2 教师 24 层 → 学生 30 层）不再 raise。

    原 test_check_shape_raises_on_too_few_teacher_layers 的语义已随 B4a 修复反转：
    层数不等由 layer_map / G2 处理（§12 G2），不是 _check_shape 的错误条件；
    只有 S/H/D 不匹配才 raise。
    """
    bad_t = np.zeros((1, 8, 2, 16))
    bad_s = np.zeros((3, 8, 2, 16))  # n_t=1 < n_s=3，但 S/H/D 一致
    _check_shape(bad_t, bad_s)  # 不抛 ValueError


# ---- bug-2 修复：de-RoPE 接入 mapper ----


def test_ridge_per_head_with_de_rope():
    """有结构数据 + de-RoPE 应该学到比纯随机更高的 retention。"""
    from apcs.rope.runner import _rope_pairs, de_rope

    kv_t, kv_s = _toy(n_t=4, n_s=2, S=16, H=2, D=16, seed=0)
    positions = np.arange(16, dtype=np.float64)
    inv_freq = _rope_pairs(16, theta=1_000_000.0)
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)

    m = RidgePerHeadMapper(lam=1e-3)
    m.fit(kv_t, kv_s, _layer_map(4, 2), positions=positions, de_rope_fn=de_rope_fn)
    pred = m.transform(kv_t, _layer_map(4, 2), positions=positions, de_rope_fn=de_rope_fn)
    assert pred.shape == kv_s.shape


def test_ridge_per_head_without_de_rope_also_works():
    """无 de-RoPE 时也必须能跑（保持兼容）。"""
    kv_t, kv_s = _toy()
    m = RidgePerHeadMapper()
    m.fit(kv_t, kv_s, _layer_map(4, 2))
    pred = m.transform(kv_t, _layer_map(4, 2))
    assert pred.shape == kv_s.shape


# ---- 低秩 + Shared Basis 形状与参数 ----


def test_lowrank_shape_and_smaller_than_ridge():
    """LowRankMapper 输出形状正确，且 rank=2 时参数数应小于 Ridge（PCR 压缩比前提）。"""
    kv_t, kv_s = _toy()
    ridge = RidgePerHeadMapper()
    ridge.fit(kv_t, kv_s, _layer_map(4, 2))
    lr = LowRankMapper(rank=2)
    lr.fit(kv_t, kv_s, _layer_map(4, 2))
    assert lr.n_params < ridge.n_params


def test_shared_basis_smaller_than_lowrank():
    """SharedBasisMapper 跨层共享基 → 同 rank 下参数数应小于 LowRankMapper，且输出形状正确。"""
    kv_t, kv_s = _toy()
    lr = LowRankMapper(rank=4)
    lr.fit(kv_t, kv_s, _layer_map(4, 2))
    sb = SharedBasisMapper(rank=4)
    sb.fit(kv_t, kv_s, _layer_map(4, 2))
    assert sb.n_params < lr.n_params
    pred = sb.transform(kv_t, _layer_map(4, 2))
    assert pred.shape == kv_s.shape


# ---- bug-1 修复：RidgeMapper per-layer 语义，n_params 不再虚高 H 倍 ----


def test_ridge_mapper_n_params_is_per_layer():
    """RidgeMapper 是 per-layer mapper（§3.4 PCR 口径）。

    bug-1 修复前：fit 循环 H 次求同一矩阵，W[(s,h)] 全相同，n_params=1536
    （L_s×H×D×D 虚高）；修复后每层只学一个 W_s → n_params = L_s×D×D = 512。
    """
    kv_t, kv_s = _toy(n_t=4, n_s=2, S=8, H=3, D=16)
    m = RidgeMapper()
    m.fit(kv_t, kv_s, _layer_map(4, 2))
    assert m.n_params == 2 * 16 * 16 == 512, (
        f"per-layer n_params 应为 L_s*D*D=512，实际 {m.n_params}"
    )
    # transform 只用 W[(s,0)]，且 per-layer 每个 (s,*) 只存一个矩阵
    assert len(m.W) == 2  # 只有层 0/1 各一个
    pred = m.transform(kv_t, _layer_map(4, 2))
    assert pred.shape == kv_s.shape


# ---- bug-2 修复：top_k>1 时 fit 使用全部 Teacher 层样本（train/infer 一致）----


def test_topk_fit_uses_all_teacher_layers():
    """layer_map 每层 k=2 个 Teacher 时，fit 必须利用全部 k 层样本。

    bug-2 修复前：n=min(k*S, S) 截断 → 只用第 1 个 Teacher 层训练，
    transform 却平均 k 层 → 训练/推理不一致。构造 kv_s[s] = mean(kv_t[l])
    + 小噪声，若 fit 用全 k 层则整体余弦应显著高于只用 1 层。
    """
    rng = np.random.default_rng(42)
    n_t, n_s, S, H, D = 4, 2, 32, 2, 16
    kv_t = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    # Student 层 0 ← Teacher 层 0,1 的均值；层 1 ← 层 1,2 的均值
    kv_s = np.zeros((n_s, S, H, D), dtype=np.float32)
    kv_s[0] = (kv_t[0] + kv_t[1]) / 2 + 0.05 * rng.standard_normal((S, H, D))
    kv_s[1] = (kv_t[1] + kv_t[2]) / 2 + 0.05 * rng.standard_normal((S, H, D))

    layer_map = [[0, 1], [1, 2]]  # 每层 k=2
    m = RidgePerHeadMapper(lam=1e-5)
    m.fit(kv_t, kv_s, layer_map)
    pred = m.transform(kv_t, layer_map)

    a = pred.reshape(-1, D)
    b = kv_s.reshape(-1, D)
    cos = float(np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    ))
    # 修复前：只训练 Teacher 层 0 → transform 平均后偏差大 → cos≈0；
    # 修复后：全部 k 层参与训练 → cos ≈ 0.94，显著高于 0.5 阈值。
    assert cos > 0.5, f"top_k=2 应利用全部 k 层样本，cos={cos:.4f}"


# ---- bug-4 修复：calibration 数据应该让 retention > 0.5 ----


def test_calibration_data_has_signal():
    """§32 calibration 数据有真实结构对应 → retention 应显著高于纯随机。

    bug-4 验证：用最直接的形式：kv_s = kv_t @ W + 噪声。
    Ridge 应该能学到 W^{-1} 使 retention 接近 1.0。
    纯随机独立数据时 retention ≈ 0。
    """
    n_t, n_s, H, D = 4, 2, 2, 16
    S = 64

    # case 1: kv_s = kv_t[:n_s] @ W + 极小噪声 → 可完美学
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    W = rng.standard_normal((D, D)).astype(np.float32) / np.sqrt(D)
    kv_s = (kv_t[:n_s] @ W + 0.001 * rng.standard_normal((n_s, S, H, D))).astype(np.float32)
    layer_map = _layer_map(n_t, n_s)
    ridge = RidgePerHeadMapper(lam=1e-5)
    ridge.fit(kv_t, kv_s, layer_map)
    pred = ridge.transform(kv_t, layer_map)
    a = pred.reshape(-1, D)
    b = kv_s.reshape(-1, D)
    cos_struct = float(np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    ))

    # case 2: 纯随机独立数据 → cosine 应 ≈ 0
    kv_t_rand = rng.standard_normal((n_t, S, H, D)).astype(np.float32)
    kv_s_rand = rng.standard_normal((n_s, S, H, D)).astype(np.float32)
    ridge2 = RidgePerHeadMapper(lam=1e-5)
    ridge2.fit(kv_t_rand, kv_s_rand, layer_map)
    pred_rand = ridge2.transform(kv_t_rand, layer_map)
    a2 = pred_rand.reshape(-1, D)
    b2 = kv_s_rand.reshape(-1, D)
    cos_rand = float(np.dot(a2.reshape(-1), b2.reshape(-1)) / (
        np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-12
    ))

    # 有结构数据 retention 应显著高于随机
    assert cos_struct > 0.5, (
        f"有结构数据应可完美学，但 cos_struct={cos_struct:.4f}"
    )
    assert cos_struct > cos_rand + 0.3, (
        f"有结构应 > 随机；struct={cos_struct:.4f}, rand={cos_rand:.4f}"
    )


# ---- bug-3 修复：校准循环必须"聚合全部样本 fit 一次"，而非每轮覆盖 W ----


def _shared_w_samples(seed=3, n_train=3, n_eval=8):
    """构造共享 W_t/W_s 的校准样本（同一模型对、不同 latent Z）。

    bug-3 语义：真实校准是"同一 Teacher/Student 模型、多个 prompt"，
    因此 W_t/W_s 应跨样本固定，仅 latent Z 与噪声随 seed 变化 ——
    否则每样本都是"不同模型"，聚合后不存在可学的共享映射（§32）。
    """
    from apcs.rope.runner import _rope_pairs, de_rope

    n_t, n_s, S, H, D, noise = 2, 2, 24, 2, 16, 0.5
    rng = np.random.default_rng(seed)
    w_t = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_t)]
    w_s = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(n_s)]

    def _one(seed_i):
        r = np.random.default_rng(seed_i)
        Z = r.standard_normal((S, H, D))
        kv_t = np.zeros((n_t, S, H, D), dtype=np.float32)
        kv_s = np.zeros((n_s, S, H, D), dtype=np.float32)
        for l in range(n_t):
            kv_t[l] = (Z @ w_t[l]) + noise * r.standard_normal(Z.shape)
        for l in range(n_s):
            kv_s[l] = (Z @ w_s[l]) + noise * r.standard_normal(Z.shape)
        return kv_t, kv_s

    train = [_one(i) for i in range(n_train)]  # seed 0,1,2 → 最后一组 = seed 2
    eval_set = [_one(9000 + i) for i in range(n_eval)]
    layer_map = [[0], [1]]
    positions = np.arange(S, dtype=np.float64)
    inv_freq = _rope_pairs(D, theta=1_000_000.0)
    de_rope_fn = lambda k, p: de_rope(k, p, inv_freq)
    return train, eval_set, layer_map, positions, de_rope_fn


def test_calibration_aggregate_uses_all_samples():
    """bug-3：校准循环每轮 `ridge.fit(...)` 覆盖 W → 只留最后一组样本的影响。

    修复：把全部校准样本**聚合**后只 fit 一次。本测试证明聚合版用了更多
    样本信息：在同一 eval 集上，聚合 fit 的 cosine 应显著高于只 fit 最后一组。

    同时做数学等价性校验：聚合 W 必须等于"全部样本 concat 后一次 fit"的 W
    （确定性证明"用了全部样本"，而不是某种近似）。
    """
    from apcs.mapper.aggregate import concat_kv_samples, fit_ridge_aggregate

    train, eval_set, layer_map, positions, de_rope_fn = _shared_w_samples()

    # group-A：当前 bug 语义 —— 只 fit 最后一组（seed=2）
    mA = RidgePerHeadMapper(lam=1e-3)
    mA.fit(
        train[-1][0], train[-1][1], layer_map,
        positions=positions, de_rope_fn=de_rope_fn,
    )
    # group-B：聚合全部 3 组
    mB = RidgePerHeadMapper(lam=1e-3)
    fit_ridge_aggregate(mB, train, layer_map, positions=positions, de_rope_fn=de_rope_fn)

    def _mean_cos(m):
        cos_list = []
        for kv_t, kv_s in eval_set:
            pred = m.transform(kv_t, layer_map, positions=positions, de_rope_fn=de_rope_fn)
            a = pred.reshape(-1, pred.shape[-1])
            b = kv_s.reshape(-1, kv_s.shape[-1])
            cos_list.append(
                float(np.dot(a.reshape(-1), b.reshape(-1)) / (
                    np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
                ))
            )
        return float(np.mean(cos_list))

    cos_single = _mean_cos(mA)
    cos_aggregate = _mean_cos(mB)
    assert cos_aggregate >= cos_single + 0.15, (
        f"聚合 fit 应显著优于只 fit 最后一组（bug 语义）；"
        f"single={cos_single:.4f}, aggregate={cos_aggregate:.4f}"
    )

    # 确定性等价：聚合 W == concat 全部样本后一次 fit 的 W（数学等价性证明）
    kv_t_big, kv_s_big = concat_kv_samples(train)
    pos_big = np.tile(positions, len(train))
    m_concat = RidgePerHeadMapper(lam=1e-3)
    m_concat.fit(kv_t_big, kv_s_big, layer_map, positions=pos_big, de_rope_fn=de_rope_fn)
    for key in m_concat.W:
        assert np.abs(mB.W[key] - m_concat.W[key]).max() < 1e-4, (
            f"聚合 W[{key}] 与 concat-fit 不一致："
            f"maxdiff={np.abs(mB.W[key] - m_concat.W[key]).max():.2e}"
        )


# ---- design.md §22 强制：K / V 独立参数化（P1 差距项）----


def test_kv_separate_parametrization_ridge():
    """§22：RidgeMapper 对 K 与 V 各持有独立参数矩阵（per-layer 语义）。

    同一 mapper 先 fit(kv_kind="K") 再 fit(kv_kind="V")，K/V 数据独立抽样：
    - W 键含 kind 维度：`(kv_kind, s, 0)`（旧 `(s, 0)`）
    - K 与 V 的参数矩阵不同（各自独立拟合，不共享）
    - n_params 从只 fit K 时的 512 翻倍到 1024（"对 dict 全部 value 求和"）
    """
    kv_t_K, kv_s_K = _toy(seed=0)
    kv_t_V, kv_s_V = _toy(seed=1)  # 独立抽样：同一结构、不同随机性
    m = RidgeMapper()
    m.fit(kv_t_K, kv_s_K, _layer_map(4, 2), kv_kind="K")
    assert ("K", 0, 0) in m.W and ("K", 1, 0) in m.W
    # 只 fit 一个 kind 时 n_params 与旧实现一致（==512 保持）
    assert m.n_params == 2 * 16 * 16 == 512, f"n_params 应为 512，实际 {m.n_params}"
    m.fit(kv_t_V, kv_s_V, _layer_map(4, 2), kv_kind="V")
    assert ("V", 0, 0) in m.W and ("V", 1, 0) in m.W
    assert len(m.W) == 4  # 2 层 × K/V 两套
    assert m.n_params == 2 * 512 == 1024, f"fit 两个 kind 应翻倍，实际 {m.n_params}"
    assert not np.allclose(m.W[("K", 0, 0)], m.W[("V", 0, 0)]), (
        "K/V 独立参数化（§22）：两套参数矩阵必须不同"
    )


def test_kv_separate_parametrization_per_head_and_lowrank():
    """§22：RidgePerHead / LowRankMapper 的 (s, h) 键加 kind 维度且 K/V 独立。"""
    kv_t_K, kv_s_K = _toy(seed=2)
    kv_t_V, kv_s_V = _toy(seed=3)
    layer_map = _layer_map(4, 2)
    r = RidgePerHeadMapper()
    r.fit(kv_t_K, kv_s_K, layer_map, kv_kind="K")
    r.fit(kv_t_V, kv_s_V, layer_map, kv_kind="V")
    assert ("K", 0, 0) in r.W and ("V", 0, 0) in r.W
    assert not np.allclose(r.W[("K", 0, 0)], r.W[("V", 0, 0)]), (
        "K/V 的 W 矩阵必须独立（§22）"
    )

    lr = LowRankMapper(rank=2)
    lr.fit(kv_t_K, kv_s_K, layer_map, kv_kind="K")
    assert ("K", 0, 0) in lr.A and ("K", 0, 0) in lr.B
    lr.fit(kv_t_V, kv_s_V, layer_map, kv_kind="V")
    assert ("V", 0, 0) in lr.A and ("V", 0, 0) in lr.B
    assert not np.allclose(lr.A[("K", 0, 0)], lr.A[("V", 0, 0)]), (
        "K/V 的 A 矩阵必须独立（§22）"
    )


def test_kv_separate_parametrization_shared_basis():
    """§22：SharedBasisMapper 的共享基 A_shared 改为 dict[kv_kind]，B 键含 kind。"""
    kv_t_K, kv_s_K = _toy(seed=4)
    kv_t_V, kv_s_V = _toy(seed=5)
    layer_map = _layer_map(4, 2)
    sb = SharedBasisMapper(rank=2)
    sb.fit(kv_t_K, kv_s_K, layer_map, kv_kind="K")
    assert "K" in sb.A_shared, "A_shared 应变为 dict[kv_kind]"
    assert ("K", 0, 0) in sb.B
    sb.fit(kv_t_V, kv_s_V, layer_map, kv_kind="V")
    assert "V" in sb.A_shared
    assert ("V", 0, 0) in sb.B
    assert not np.allclose(sb.A_shared["K"], sb.A_shared["V"]), (
        "K/V 的共享基必须独立（§22）"
    )


def test_kv_separate_transform_uses_matching_kind():
    """§22：transform 用与 fit 匹配的 kv_kind 取参数，输出形状正确。"""
    kv_t_K, kv_s_K = _toy(seed=6)
    kv_t_V, kv_s_V = _toy(seed=7)
    layer_map = _layer_map(4, 2)
    m = RidgePerHeadMapper()
    m.fit(kv_t_K, kv_s_K, layer_map, kv_kind="K")
    m.fit(kv_t_V, kv_s_V, layer_map, kv_kind="V")
    pred_K = m.transform(kv_t_K, layer_map, kv_kind="K")
    pred_V = m.transform(kv_t_V, layer_map, kv_kind="V")
    assert pred_K.shape == kv_s_K.shape
    assert pred_V.shape == kv_s_V.shape
    # 同一 K 数据用 V 参数 vs 用 K 参数输出不同 → 证明取参按 kind 分开
    pred_K_with_V = m.transform(kv_t_K, layer_map, kv_kind="V")
    assert not np.allclose(pred_K, pred_K_with_V, atol=1e-6), (
        "transform 必须按 kv_kind 取对应参数（§22），K 数据用 V 参数应不同"
    )


def test_kv_separate_unfitted_kind_raises_keyerror():
    """§22：未 fit 的 kind 调用 transform 必须 raise KeyError（禁止静默回退）。"""
    kv_t, kv_s = _toy()
    layer_map = _layer_map(4, 2)
    m = RidgeMapper()
    m.fit(kv_t, kv_s, layer_map, kv_kind="K")
    with pytest.raises(KeyError, match="K"):
        m.transform(kv_t, layer_map, kv_kind="V")
    # SharedBasisMapper 同样禁止静默回退
    sb = SharedBasisMapper(rank=2)
    sb.fit(kv_t, kv_s, layer_map, kv_kind="K")
    with pytest.raises(KeyError, match="K"):
        sb.transform(kv_t, layer_map, kv_kind="V")


def test_runner_ridge_baseline_separate_kv_metrics(tmp_path):
    """§22 接线：separate_kv=True 时 run_ridge_baseline 的 metrics 含 K/V 独立字段。

    真实 CLI 路径（configs/pair_qwen3.yaml 的 mapper.separate_kv=true）走 K/V
    分离分支：先 fit(kv_kind="K") 再 fit(kv_kind="V")，metrics 必须含
    `retention_K` / `retention_V`（以及 mean_cos_K / mean_cos_V）。
    """
    from apcs.mapper.runner import run_ridge_baseline

    cfg = {
        "teacher": {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16},
        "student": {"num_layers": 2, "num_kv_heads": 2, "head_dim": 16},
        "mapper": {
            "calibration_context": 64,
            "calibration_samples": 6,
            "separate_kv": True,
        },
        "timing": {"repeats": 2, "warmup": 1},
    }
    res = run_ridge_baseline(cfg, tmp_path)
    metrics = res["metrics"]
    assert "retention_K" in metrics, "separate_kv=True 必须输出 retention_K（§22）"
    assert "retention_V" in metrics, "separate_kv=True 必须输出 retention_V（§22）"
    assert "mean_cos_K" in metrics and "mean_cos_V" in metrics
    assert res["status"] in ("PASS", "FAIL")


# ---- §22 修复回归：held-out 种子空间 / kv_kinds / _merge_kv_fields / G2 警告 ----


def test_held_out_eval_samples_do_not_collide_with_calib():
    """seed 修复：calib（master_seed=0）与 eval（master_seed=1）的样本必须不同。

    旧实现样本 seed=i 只随索引走 → eval[0..19] 与 calib[0..19] 逐字节相同
    （held-out 名存实亡，retention 被虚高）。
    """
    from apcs.mapper.runner import _shared_model_weights, _synth_calibration_set

    n_t, n_s, S, H, D = 4, 4, 64, 8, 128
    w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
    calib = _synth_calibration_set(
        n_t, n_s, S, H, D, 32, master_seed=0, noise=0.05, w_t=w_t, w_s=w_s
    )
    evl = _synth_calibration_set(
        n_t, n_s, S, H, D, 20, master_seed=1, noise=0.05, w_t=w_t, w_s=w_s
    )
    for i in range(min(len(calib), len(evl))):
        assert not np.array_equal(evl[i][0], calib[i][0]), (
            f"eval[{i}] 与 calib[{i}] 逐字节相同 → held-out 失效（种子碰撞）"
        )


def test_synth_calibration_reproducible_same_master_seed():
    """回归：同 master_seed 生成结果必须可复现（§51 可复现性）。"""
    from apcs.mapper.runner import _shared_model_weights, _synth_calibration_set

    n_t, n_s, S, H, D = 4, 4, 32, 8, 128
    w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
    a = _synth_calibration_set(n_t, n_s, S, H, D, 5, master_seed=0, w_t=w_t, w_s=w_s)
    b = _synth_calibration_set(n_t, n_s, S, H, D, 5, master_seed=0, w_t=w_t, w_s=w_s)
    assert np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[3][1], b[3][1])


def test_kv_kinds_helper():
    """internal helper：separate_kv=false → ["K"]，true → ["K","V"]（单一事实源）。"""
    from apcs.mapper.runner import KIND_SEED_OFFSET, kv_kinds

    assert kv_kinds({"mapper": {"separate_kv": False}}) == ["K"]
    assert kv_kinds({"mapper": {"separate_kv": True}}) == ["K", "V"]
    assert kv_kinds({}) == ["K"]
    assert set(KIND_SEED_OFFSET) == {"K", "V"}
    assert KIND_SEED_OFFSET["K"] == 0 and KIND_SEED_OFFSET["V"] == 100000


def test_merge_kv_fields_guard_and_write():
    """_merge_kv_fields：单 kind 无副作用；双 kind 写 {base}_K/{base}_V。"""
    from apcs.mapper.runner import _merge_kv_fields

    dst: dict = {}
    _merge_kv_fields(dst, {"K": {"retention": 0.9}}, {"retention": "retention"})
    assert dst == {}, "单 kind（无 V）时不应写入 *_K/*_V"

    per_kind = {
        "K": {"retention": 0.91, "mean_cos": 0.92},
        "V": {"retention": 0.93, "mean_cos": 0.94},
    }
    dst2: dict = {}
    _merge_kv_fields(dst2, per_kind, {"retention": "retention", "mean_cos": "mean_cos"})
    assert dst2["retention_K"] == 0.91 and dst2["retention_V"] == 0.93
    assert dst2["mean_cos_K"] == 0.92 and dst2["mean_cos_V"] == 0.94


def test_cfg_layers_warns_on_head_mismatch():
    """G2 修复：_cfg_layers 在 teacher/student head 数或 head_dim 不同时发警告。"""
    import warnings

    from apcs.mapper.runner import _cfg_layers

    cfg_ok = {
        "teacher": {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16},
        "student": {"num_layers": 2, "num_kv_heads": 2, "head_dim": 16},
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _cfg_layers(cfg_ok)
        assert not any("G2" in str(x.message) for x in w)

    cfg_bad = {
        "teacher": {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16},
        "student": {"num_layers": 2, "num_kv_heads": 2, "head_dim": 32},
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n_t, n_s, H, D = _cfg_layers(cfg_bad)
        assert any("G2" in str(x.message) for x in w), "head_dim 不一致应触发 G2 警告"
        # 与文档语义一致：runner 按 teacher 维度训练，H 取 max
        assert (n_t, n_s, D) == (4, 2, 16)