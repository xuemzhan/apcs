"""协议 v1.1 单元测试（A1/A2/A4/B1/B2/B3/D1/D2 修复的回归验证）。

覆盖：
    - A1  指标：permutation_p / gold_prob_from_logits
    - D1  审计：cache_seq_len 从实际 cache 读取 + EvalPhaseCounters 真实断言
    - B1  持久化：KV/mapper 参数/manifest 落盘往返 + 篡改检测
    - D2  JointKVMapper：联合布局 K/V 独立参数化（与手动拆分等价）
    - B3  RoPE unrotated：held-out 内容上显著优于 rotated（W 吸收旋转）模式
    - A4  多数据集轮转采样
    - 配置项：InjectionEvaluator 读取 v1.1 开关
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apcs.inference.evaluator import (
    EvalPhaseCounters,
    InjectionEvaluator,
    cache_seq_len,
    gold_prob_from_logits,
)
from apcs.inference.kv_store import (
    load_kv_sample,
    load_mapper_params,
    load_store,
    save_kv_sample,
    save_mapper_params,
    write_manifest,
)
from apcs.mapper.joint import JointKVMapper
from apcs.mapper.math import RidgePerHeadMapper
from apcs.metrics import permutation_p
from apcs.rope.runner import _rope_pairs, apply_rope, de_rope


# ---------------------------------------------------------------------------
# A1: permutation_p / gold_prob_from_logits
# ---------------------------------------------------------------------------

def test_permutation_p_positive_diffs_significant():
    rng = np.random.default_rng(0)
    diffs = rng.normal(loc=0.5, scale=0.05, size=40).tolist()
    p = permutation_p(diffs, n_perm=500)
    assert p < 0.01, f"恒正差值应显著，got p={p}"


def test_permutation_p_symmetric_noise_not_significant():
    rng = np.random.default_rng(1)
    diffs = rng.normal(loc=0.0, scale=1.0, size=40).tolist()
    p = permutation_p(diffs, n_perm=500)
    assert p > 0.05, f"零均值噪声不应显著，got p={p}"


def test_gold_prob_from_logits():
    letter_ids = {"A": 32, "B": 33, "C": 34, "D": 35}
    logits = np.zeros(100)
    logits[33] = 10.0  # B 最强
    gp_b = gold_prob_from_logits(logits, "B", letter_ids)
    gp_a = gold_prob_from_logits(logits, "A", letter_ids)
    assert gp_b is not None and gp_b > 0.99
    assert gp_a is not None and gp_a < 0.01
    assert gold_prob_from_logits(logits, None, letter_ids) is None
    assert gold_prob_from_logits(logits, "E", letter_ids) is None


# ---------------------------------------------------------------------------
# D1: 真实 cache 审计
# ---------------------------------------------------------------------------

class _FakeCache:
    def __init__(self, seq_len: int):
        self._n = seq_len

    def get_seq_length(self) -> int:
        return self._n


def test_cache_seq_len_reads_actual_cache():
    assert cache_seq_len(_FakeCache(65)) == 65
    assert cache_seq_len(_FakeCache(0)) == 0


def test_counters_real_assertions():
    c = EvalPhaseCounters()
    c.query_past_len = 65
    c.inject_seq_len = 65
    c.student_prefill_new_tokens = 0
    assert c.is_verified
    # past_len 与 inject_seq_len 不一致 → 审计必须失败（此前手工赋值恒真）
    c.query_past_len = 30
    assert not c.is_verified
    c.query_past_len = 65
    c.student_prefill_new_tokens = 5  # forward 意外追加 token
    assert not c.is_verified


# ---------------------------------------------------------------------------
# B1: KV store 持久化往返 + 篡改检测
# ---------------------------------------------------------------------------

def _fit_small_mapper():
    rng = np.random.default_rng(0)
    L, S, H, D = 2, 16, 2, 8
    kv_t = rng.standard_normal((L, S, H, D))
    kv_s = rng.standard_normal((L, S, H, D))
    layer_map = [[0], [1]]
    m = RidgePerHeadMapper(lam=1e-3)
    m.fit(kv_t, kv_s, layer_map, kv_kind="K")
    m.fit(kv_t, kv_s, layer_map, kv_kind="V")
    return m, kv_t, kv_s, layer_map


def test_kv_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    k = rng.standard_normal((3, 20, 2, 8)).astype(np.float32)
    v = rng.standard_normal((3, 20, 2, 8)).astype(np.float32)
    save_kv_sample(tmp_path, "hs-001", k, v)
    k2, v2, S = load_kv_sample(tmp_path, "hs-001")
    assert S == 20
    np.testing.assert_allclose(k2, k, atol=1e-6)
    np.testing.assert_allclose(v2, v, atol=1e-6)

    m, _, _, _ = _fit_small_mapper()
    save_mapper_params(tmp_path, {"K": m, "V": m})
    m2 = RidgePerHeadMapper(lam=1e-3)
    m3 = RidgePerHeadMapper(lam=1e-3)
    assert load_mapper_params(tmp_path, {"K": m2, "V": m3})
    for key, w in m.W.items():
        kind, s, h = key
        target = m2 if kind == "K" else m3
        np.testing.assert_allclose(target.W[key], w, atol=1e-7)

    write_manifest(tmp_path, {"n_samples": 1})
    store = load_store(tmp_path, verify=True)
    assert store["has_mapper_params"]
    assert "hs-001" in store["kv_sample_ids"]


def test_kv_store_tamper_detected(tmp_path):
    rng = np.random.default_rng(0)
    save_kv_sample(tmp_path, "hs-002", rng.standard_normal((1, 8, 1, 4)), rng.standard_normal((1, 8, 1, 4)))
    write_manifest(tmp_path, {})
    # 篡改：改写 KV 文件
    kv_file = tmp_path / "kv" / "hs-002.npz"
    data = dict(np.load(kv_file))
    data["k"] = data["k"] + 1.0
    np.savez_compressed(kv_file, **data)
    with pytest.raises(ValueError, match="校验和不匹配"):
        load_store(tmp_path, verify=True)


# ---------------------------------------------------------------------------
# D2: JointKVMapper —— 联合布局与手动拆分等价、K/V 参数独立
# ---------------------------------------------------------------------------

def test_joint_kv_mapper_equivalent_to_manual_split():
    _, kv_t, kv_s, layer_map = _fit_small_mapper()
    # 手动拆分拟合
    mk_manual = RidgePerHeadMapper(lam=1e-3)
    mv_manual = RidgePerHeadMapper(lam=1e-3)
    dt = kv_t.shape[-1] // 2
    mk_manual.fit(kv_t[..., :dt], kv_s[..., :dt], layer_map, kv_kind="K")
    mv_manual.fit(kv_t[..., dt:], kv_s[..., dt:], layer_map, kv_kind="V")
    # JointKVMapper 拟合
    mk_j = RidgePerHeadMapper(lam=1e-3)
    mv_j = RidgePerHeadMapper(lam=1e-3)
    joint = JointKVMapper(mk_j, mv_j)
    joint.fit(kv_t, kv_s, layer_map)
    for key, w in mk_manual.W.items():
        np.testing.assert_allclose(mk_j.W[key], w, atol=1e-10)
    for key, w in mv_manual.W.items():
        np.testing.assert_allclose(mv_j.W[key], w, atol=1e-10)
    # K/V 参数独立（§22）：同输入下两组 W 不同
    assert not np.allclose(next(iter(mk_j.W.values())), next(iter(mv_j.W.values())))
    # transform 往返一致
    out_joint = joint.transform(kv_t, layer_map)
    out_manual = np.concatenate(
        [
            mk_manual.transform(kv_t[..., :dt], layer_map, kv_kind="K"),
            mv_manual.transform(kv_t[..., dt:], layer_map, kv_kind="V"),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(out_joint, out_manual, atol=1e-10)


# ---------------------------------------------------------------------------
# B3: unrotated RoPE —— held-out 内容上显著优于 rotated（W 吸收旋转）模式
# ---------------------------------------------------------------------------

def test_rope_4d_axis_alignment():
    """◆ 回归：apply_rope/de_rope 对 4D (L,S,H,D) 输入应把 positions 对齐到
    S（dim1），此前硬编码 dim0 导致广播崩溃。"""
    rng = np.random.default_rng(3)
    L, S, H, D = 3, 10, 2, 8
    inv_freq = _rope_pairs(D, theta=10_000.0)
    positions = np.arange(S, dtype=np.float64)
    x = rng.standard_normal((L, S, H, D))
    rt = de_rope(apply_rope(x, positions, inv_freq), positions, inv_freq)
    np.testing.assert_allclose(rt, x, atol=1e-10)
    # L == S 的歧义场景：约定 4D 时 S 在 dim1
    x2 = rng.standard_normal((S, S, H, D))
    rt2 = de_rope(apply_rope(x2, positions, inv_freq), positions, inv_freq)
    np.testing.assert_allclose(rt2, x2, atol=1e-10)


def test_unrotated_rope_generalizes_better_than_rotated():
    """构造可解的合成问题验证 §23 设计意图：

        teacher K = R(p) @ C；student K = R(p) @ (M @ C)

    - rotated（旧行为）：fit 目标是 student 旋转 K，W 必须吸收位置相关的
      R(p) → held-out 内容上误差大；
    - unrotated（v1.1）：fit 目标是 de-roped student K（= M@C），W 学纯
      内容映射，注入前 re-RoPE → held-out 内容上误差≈0。
    """
    rng = np.random.default_rng(7)
    S, H, D = 48, 1, 8
    inv_freq = _rope_pairs(D, theta=10_000.0)
    positions = np.arange(S, dtype=np.float64)
    M = rng.standard_normal((D, D)) + 2.0 * np.eye(D)  # 可逆内容映射

    def make_rotated_pair(content):
        teacher_rot = apply_rope(content, positions, inv_freq)  # R(p) @ C
        student_rot = apply_rope(content @ M, positions, inv_freq)  # R(p) @ M@C
        return teacher_rot[:, None, :], student_rot[:, None, :]  # (S,H,D)

    c_train = rng.standard_normal((S, D))
    c_hold = rng.standard_normal((S, D))
    kt_tr, ks_tr = make_rotated_pair(c_train)
    kt_ho, ks_ho = make_rotated_pair(c_hold)
    layer_map = [[0]]

    def teacher_deroped(k, p):
        return de_rope(k, p, inv_freq)

    # —— rotated（旧行为）：目标 = student 旋转 K ——
    m_rot = RidgePerHeadMapper(lam=1e-8)
    m_rot.fit(kt_tr[None], ks_tr[None], layer_map, kv_kind="K",
              positions=positions, de_rope_fn=teacher_deroped)
    pred_rot = m_rot.transform(kt_ho[None], layer_map, kv_kind="K",
                               positions=positions, de_rope_fn=teacher_deroped)[0]
    err_rot = float(np.linalg.norm(pred_rot - ks_ho))

    # —— unrotated（v1.1）：目标 = student de-roped K，变换后 re-RoPE ——
    ks_tr_unrot = de_rope(ks_tr, positions, inv_freq)
    m_unrot = RidgePerHeadMapper(lam=1e-8)
    m_unrot.fit(kt_tr[None], ks_tr_unrot[None], layer_map, kv_kind="K",
                positions=positions, de_rope_fn=teacher_deroped)
    pred_unrot = m_unrot.transform(kt_ho[None], layer_map, kv_kind="K",
                                   positions=positions, de_rope_fn=teacher_deroped)[0]
    pred_unrot = apply_rope(pred_unrot, positions, inv_freq)
    err_unrot = float(np.linalg.norm(pred_unrot - ks_ho))

    scale = float(np.linalg.norm(ks_ho))
    assert err_unrot < 0.05 * scale, f"unrotated held-out 误差应≈0，got {err_unrot / scale:.3f}x"
    assert err_unrot < 0.2 * err_rot, (
        f"unrotated({err_unrot:.4f}) 应显著优于 rotated({err_rot:.4f})"
    )


# ---------------------------------------------------------------------------
# A4: 多数据集轮转采样
# ---------------------------------------------------------------------------

def test_load_sample_rows_round_robin(monkeypatch):
    import apcs.inference.cli as cli_mod

    def fake_load(name, n, split, seed):
        return [
            SimpleNamespace(sample_id=f"{name}-{i}", context="c", query="q",
                            answer="A", split=split)
            for i in range(n)
        ]

    import apcs.data.hf_dataset as hf_ds
    monkeypatch.setattr(hf_ds, "load", fake_load)
    cfg = {
        "inject_eval": {"max_samples": 10, "seed": 0},
        "datasets": {"fidelity": ["dsA", "dsB"]},
    }
    rows = cli_mod._load_sample_rows(cfg)
    ids = [r.sample_id for r in rows]
    assert len(rows) == 10
    # 轮转交错：两个数据集交替出现
    assert ids[0].startswith("dsA") and ids[1].startswith("dsB")
    assert sum(i.startswith("dsA") for i in ids) == 5
    assert sum(i.startswith("dsB") for i in ids) == 5


# ---------------------------------------------------------------------------
# 配置项：InjectionEvaluator 读取 v1.1 开关
# ---------------------------------------------------------------------------

def _make_cfg(**ie_overrides):
    ie = {
        "max_samples": 8,
        "seed": 0,
        "calib_eval_split": True,
        "calib_samples": 4,
        "persist_kv": True,
        "export_hidden_states": False,
    }
    ie.update(ie_overrides)
    return {
        "inject_eval": ie,
        "teacher": {"model_id": "t", "num_layers": 4, "num_kv_heads": 2, "head_dim": 8},
        "student": {"model_id": "s", "num_layers": 2, "num_kv_heads": 2, "head_dim": 8},
        "mapper": {"type": "ridge", "rope_align": "unrotated",
                   "ridge_lambda_k": 1e-3, "ridge_lambda_v": 1e-3},
    }


def test_evaluator_reads_v11_flags(tmp_path):
    ev = InjectionEvaluator(_make_cfg(), tmp_path)
    assert ev._calib_split is True
    assert ev._n_calib == 4
    assert ev._persist_kv is True
    assert ev._rope_align == "unrotated"
    assert ev._export_hidden is False
    assert ev._kv_store_dir == tmp_path / "inject_eval" / "kv_store"
    # 关闭切分 → 回退旧行为（provenance 会如实标注 disjoint=False）
    ev2 = InjectionEvaluator(_make_cfg(calib_eval_split=False), tmp_path)
    assert ev2._calib_split is False


def test_evaluate_online_missing_store_raises(tmp_path):
    ev = InjectionEvaluator(_make_cfg(), tmp_path)
    with pytest.raises(FileNotFoundError):
        ev.evaluate_online(tmp_path / "nonexistent_store")


# ---------------------------------------------------------------------------
# P0.0/P0.5: 统一格式 helper 与 self_kv 对照模式
# ---------------------------------------------------------------------------

def test_unified_suffix_and_ids():
    from apcs.inference.evaluator import _unified_suffix_text

    q = "Continue the passage (A/B/C/D): A=x | B=y"
    s = _unified_suffix_text(q)
    assert s.endswith("\nAnswer:")
    assert s.startswith(q)


def test_unified_ids_split_tokenization():
    from types import SimpleNamespace as _NS

    class _Tok:
        def __call__(self, text, return_tensors=None):
            # 词表级"tokenizer"：按字符 id 化，便于断言拼接语义
            ids = np.array([hash(c) % 1000 for c in text])
            return _NS(input_ids=ids.reshape(1, -1))

        def encode(self, t, add_special_tokens=False):
            return [hash(t) % 1000]

    ev = InjectionEvaluator(_make_cfg(), "/tmp")
    row = SimpleNamespace(sample_id="x", context="CTX", query="QQ", answer="A")
    tok = _Tok()
    ids = ev._unified_ids(row, tok, "CTX")
    from apcs.inference.evaluator import _unified_suffix_text
    suffix_ids = tok(_unified_suffix_text(row.query)).input_ids.reshape(-1)
    ctx_ids = tok("CTX").input_ids.reshape(-1)
    # 分离 tokenize 再拼接：suffix 部分与单独 tokenize 完全一致
    np.testing.assert_array_equal(ids[: len(ctx_ids)], ctx_ids)
    np.testing.assert_array_equal(ids[len(ctx_ids):], suffix_ids)


def test_handoff_context_text_with_prefix():
    cfg = _make_cfg()
    cfg["teacher"]["prefill_prefix"] = "Read carefully.\n"
    ev = InjectionEvaluator(cfg, "/tmp")
    row = SimpleNamespace(sample_id="x", context="CTX", query="Q", answer="A")
    assert ev._handoff_context_text(row) == "Read carefully.\nCTX"
    # 无前缀 → 原 context
    ev2 = InjectionEvaluator(_make_cfg(), "/tmp")
    assert ev2._handoff_context_text(row) == "CTX"


# ---------------------------------------------------------------------------
# P0.1/P0.3: fit 路由 —— Affine 走 fit_batch（变长安全）
# ---------------------------------------------------------------------------

def test_fit_mappers_routes_affine_with_variable_lengths(tmp_path):
    """变长真实校准样本下 AffineMapper 经 fit_batch 拟合不再崩溃，
    且 transform 输出形状正确（P0.1：中心化截距 mapper）。"""
    from apcs.mapper.math import AffineMapper

    cfg = _make_cfg()
    cfg["mapper"]["type"] = "affine"
    ev = InjectionEvaluator(cfg, tmp_path)
    ev._build_mappers()
    assert isinstance(ev._mapper_k, AffineMapper)

    rng = np.random.default_rng(0)
    calib = [
        (rng.standard_normal((4, S, 2, 8)), rng.standard_normal((2, S, 2, 8)))
        for S in (10, 16, 24)  # 变长
    ]
    ev._fit_mappers({"K": calib, "V": calib})
    out = ev._mapper_k.transform(calib[0][0], ev._layer_map, kv_kind="K")
    assert out.shape == (2, 10, 2, 8)
    # bias 已写入（中心化截距）
    assert len(ev._mapper_k.bias) > 0


def test_fit_mappers_routes_shared_basis(tmp_path):
    from apcs.mapper.math import SharedBasisMapper

    ev = InjectionEvaluator(_make_cfg(), tmp_path)
    ev.cfg["mapper"]["type"] = "shared_basis"
    ev._build_mappers()
    assert isinstance(ev._mapper_k, SharedBasisMapper)
    rng = np.random.default_rng(1)
    calib = [
        (rng.standard_normal((4, S, 2, 8)), rng.standard_normal((2, S, 2, 8)))
        for S in (8, 12)
    ]
    ev._fit_mappers({"K": calib, "V": calib})
    out = ev._mapper_k.transform(calib[0][0], ev._layer_map, kv_kind="K")
    assert out.shape == (2, 8, 2, 8)


# ---------------------------------------------------------------------------
# P1.1/P1.3: TaskAware diag 参数化与逐层 α
# ---------------------------------------------------------------------------

def test_task_aware_diag_delta_and_layer_alpha(tmp_path):
    from apcs.mapper.task_aware import TaskAwareRidgeMapper

    m = TaskAwareRidgeMapper(
        lam=1e-3, task_loss_weight=0.7, task_delta_mode="diag",
        learn_layer_alpha=True,
    )
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((4, 12, 2, 8))
    kv_s = rng.standard_normal((2, 12, 2, 8))
    lm = [[0], [1]]
    m.fit(kv_t, kv_s, lm, kv_kind="K")
    assert m._fitted == {"K"}

    # 手动设置 diag 增量（模拟 Phase B 结果）
    key0 = ("K", 0, 0)
    m._gamma_np = {key0: np.full(8, 1.5)}
    m._beta_np = {key0: np.full(8, 0.1)}
    m._layer_alpha_np = np.array([0.5, 1.0])
    m.task_delta_mode = "diag"
    m.learn_layer_alpha = True

    base = m.W[key0]
    kv_in = rng.standard_normal((4, 12, 2, 8))
    out = m.transform(kv_in, lm, kv_kind="K")
    # 手工重算 (s0,h0): (x@W)*1.5+0.1 再 ×α0=0.5
    x = kv_in[0, :, 0, :]
    manual = ((x @ base) * 1.5 + 0.1) * 0.5
    np.testing.assert_allclose(out[0, :, 0, :], manual, atol=1e-6)


def test_task_aware_config_passthrough(tmp_path):
    cfg = _make_cfg()
    cfg["mapper"]["type"] = "task_aware"
    cfg["mapper"].update({
        "task_loss_weight": 0.5, "task_delta_mode": "diag",
        "learn_layer_alpha": True,
    })
    ev = InjectionEvaluator(cfg, tmp_path)
    ev._build_mappers()
    assert ev._mapper_k.task_loss_weight == 0.5
    assert ev._mapper_k.task_delta_mode == "diag"
    assert ev._mapper_k.learn_layer_alpha is True


# ---------------------------------------------------------------------------
# v1.3: AffineLayerMapper —— 偏置收益 × 每层参数缩减（头维并入样本行）
# ---------------------------------------------------------------------------

def test_affine_layer_recovers_affine_map():
    from apcs.mapper.math import AffineLayerMapper

    rng = np.random.default_rng(0)
    M = rng.standard_normal((8, 8))
    c = rng.standard_normal(8) * 3
    pairs = []
    for S in (10, 17, 23):  # 变长
        src = rng.standard_normal((4, S, 2, 8))
        pairs.append((src, src @ M + c))
    m = AffineLayerMapper(lam=1e-10)
    m.fit_batch(pairs, [[0], [1], [2], [3]], kv_kind="K")
    out = m.transform(pairs[0][0], [[0], [1], [2], [3]], kv_kind="K")
    err = float(np.abs(out - pairs[0][1][:, :10]).max())
    assert err < 1e-5, err
    # per-layer 参数量 = 每层 (D²+D)，比 per-head affine 少 H 倍
    assert m.n_params == 4 * (64 + 8)


def test_affine_layer_kv_independence():
    from apcs.mapper.math import AffineLayerMapper

    rng = np.random.default_rng(1)
    mk, mv = AffineLayerMapper(lam=1e-3), AffineLayerMapper(lam=1e-3)
    kv_t = rng.standard_normal((2, 12, 2, 8))
    kv_s_k = rng.standard_normal((2, 12, 2, 8))
    kv_s_v = rng.standard_normal((2, 12, 2, 8))
    lm = [[0], [1]]
    mk.fit_batch([(kv_t, kv_s_k)], lm, kv_kind="K")
    mv.fit_batch([(kv_t, kv_s_v)], lm, kv_kind="V")
    # 不同目标 ⇒ 参数不同；kind 键独立（§22）
    assert not np.allclose(mk.W[("K", 0, 0)], mv.W[("V", 0, 0)])
    with pytest.raises(KeyError):
        mk.transform(kv_t, lm, kv_kind="V")


# ---------------------------------------------------------------------------
# v1.4: RAT（残差锚定翻译器）与 Oracle 探针
# ---------------------------------------------------------------------------

def _build_synth_world(seed=0):
    """合成"模型对"：共享隐变量 z、已知跨模型映射 R0、embedding 锚可恢复 R0。"""
    rng = np.random.default_rng(seed)
    H, D = 4, 8
    hid_t, hid_s = 32, 24
    L_t, L_s = 4, 3
    R0 = rng.standard_normal((hid_t, hid_s))
    W_V_t = {l: rng.standard_normal((H * D, hid_t)) * 0.3 for l in range(L_t)}
    W_V_s = {l: rng.standard_normal((H * D, hid_s)) * 0.3 for l in range(L_s)}
    W_K_t = {l: rng.standard_normal((H * D, hid_t)) * 0.3 for l in range(L_t)}
    W_K_s = {l: rng.standard_normal((H * D, hid_s)) * 0.3 for l in range(L_s)}
    E_t = rng.standard_normal((64, hid_t))
    E_s = E_t @ R0
    wt, ws = {}, {}
    for l in range(L_t):
        wt[f"model.layers.{l}.self_attn.v_proj.weight"] = W_V_t[l]
        wt[f"model.layers.{l}.self_attn.k_proj.weight"] = W_K_t[l]
    for l in range(L_s):
        ws[f"model.layers.{l}.self_attn.v_proj.weight"] = W_V_s[l]
        ws[f"model.layers.{l}.self_attn.k_proj.weight"] = W_K_s[l]
    wt["model.embed_tokens.weight"] = E_t
    ws["model.embed_tokens.weight"] = E_s

    def make_pairs(n, S, offset=0):
        pairs = []
        for i in range(n):
            z = rng.standard_normal((S, hid_t)) + offset
            vt = np.stack([(z @ W_V_t[l].T).reshape(S, H, D) for l in range(L_t)])
            vs = np.stack([((z @ R0) @ W_V_s[l].T).reshape(S, H, D) for l in range(L_s)])
            pairs.append((vt, vs))
        return pairs

    lm = [[min(i, L_t - 1)] for i in range(L_s)]
    return dict(rng=rng, R0=R0, wt=wt, ws=ws, lm=lm, make_pairs=make_pairs,
                H=H, D=D, L_s=L_s)


def test_embedding_anchor_recovers_linear_map():
    from apcs.mapper.rat import embedding_procrustes

    w = _build_synth_world(seed=1)
    R = embedding_procrustes(
        w["wt"]["model.embed_tokens.weight"], w["ws"]["model.embed_tokens.weight"]
    )
    rel = float(np.abs(R - w["R0"]).max() / np.abs(w["R0"]).max())
    assert rel < 1e-4, rel


def test_rat_head_matching_recovers_alignment():
    from apcs.mapper.rat import hungarian_head_matching

    rng = np.random.default_rng(2)
    D = 8
    base = rng.standard_normal((4, D))          # 4 个"真实"头方向
    perm_true = np.array([2, 0, 3, 1])
    t_means = base.copy()
    s_means = base[perm_true] + rng.standard_normal((4, D)) * 1e-6  # 学生头=教师头的置换
    perm = hungarian_head_matching(t_means, s_means)
    np.testing.assert_array_equal(perm, perm_true)


def test_rat_beats_noanchor_under_rank_budget(tmp_path, monkeypatch):
    """核心假设：受限秩预算下，架构解析锚优于纯数据校正。"""
    import apcs.mapper.rat as rat_mod

    w = _build_synth_world(seed=3)
    monkeypatch.setattr(
        rat_mod, "_load_tensor",
        lambda md, key: (w["wt"] if "teacher" in str(md) else w["ws"])[key],
    )
    monkeypatch.setattr(rat_mod, "_resolve_model_dir", lambda p: p)
    (tmp_path / "teacher_dir").mkdir(exist_ok=True)
    (tmp_path / "student_dir").mkdir(exist_ok=True)
    calib = w["make_pairs"](24, 32)
    test = w["make_pairs"](8, 32)

    errs = {}
    for name, kw in [("rat", dict(rank=4)),
                     ("noanchor", dict(rank=4, use_embedding_anchor=False)),
                     ("analytic_only", dict(rank=0))]:
        m = rat_mod.RATMapper(lam=1e-4, head_match=True, sink_override=False, **kw)
        m.setup_weights(tmp_path / "teacher_dir", tmp_path / "student_dir")
        m.fit_batch(calib, w["lm"], kv_kind="V")
        errs[name] = float(np.mean([
            np.linalg.norm(m.transform(vt, w["lm"], kv_kind="V") - vs)
            / np.linalg.norm(vs) for vt, vs in test
        ]))
    assert errs["rat"] < errs["noanchor"], errs
    assert errs["analytic_only"] < errs["noanchor"], errs


def test_rat_interface_and_sink(tmp_path, monkeypatch):
    import apcs.mapper.rat as rat_mod

    w = _build_synth_world(seed=4)
    monkeypatch.setattr(
        rat_mod, "_load_tensor",
        lambda md, key: (w["wt"] if "teacher" in str(md) else w["ws"])[key],
    )
    monkeypatch.setattr(rat_mod, "_resolve_model_dir", lambda p: p)
    (tmp_path / "teacher_dir").mkdir(exist_ok=True)
    (tmp_path / "student_dir").mkdir(exist_ok=True)
    m = rat_mod.RATMapper(lam=1e-3, rank=8, head_match=True, sink_override=True)
    m.setup_weights(tmp_path / "teacher_dir", tmp_path / "student_dir")
    calib = w["make_pairs"](6, 16)
    m.fit_batch(calib, w["lm"], kv_kind="V")
    vt, vs = calib[0]
    out = m.transform(vt, w["lm"], kv_kind="V")
    assert out.shape == vs.shape
    # sink 覆盖：position 0 应等于学生校准均值
    sink0 = np.stack([kv_s[:, 0, :, :] for _, kv_s in calib]).mean(axis=0)
    np.testing.assert_allclose(out[:, 0, :, :], sink0, atol=1e-5)
    with pytest.raises(KeyError):
        m.transform(vt, w["lm"], kv_kind="K")  # 未 fit 的 kind 禁止静默回退


def test_probe_modes_gated(tmp_path):
    """探针模式必须在 probe_mode=true 时才可用（防误用于部署结论）。"""
    ev = InjectionEvaluator(_make_cfg(), tmp_path)
    assert ev._probe_mode is False
    ev2 = InjectionEvaluator(_make_cfg(probe_mode=True), tmp_path)
    assert ev2._probe_mode is True
