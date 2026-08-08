"""配置加载 + RoPE + Mapper + Alignment 单元测试。"""
from __future__ import annotations

import numpy as np

from apcs.alignment.runner import (
    data_driven_topk,
    geometry_aware_topk,
    last_layer_mapping,
    proportional_mapping,
    synthetic_similarity,
)
from apcs.io import load_config
from apcs.mapper.math import LowRankMapper, RidgeMapper
from apcs.rope.runner import apply_rope, de_rope, roundtrip_error


def test_load_config_with_placeholder(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "experiment:\n  name: foo\n  run_id: ${experiment.name}-${run.timestamp}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg["experiment"]["name"] == "foo"
    # timestamp 已替换
    assert "${run.timestamp}" not in cfg["experiment"]["run_id"]


def test_rope_roundtrip_perfect():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 128))
    pos = np.arange(16, dtype=np.float64)
    err, cos = roundtrip_error(x, pos, head_dim=128, theta=1_000_000.0)
    assert err < 1e-8
    assert cos > 0.999999


def test_rope_direct_and_inverse():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 128))
    pos = np.arange(4, dtype=np.float64)
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, 128, 2) / 128))
    y = apply_rope(x, pos, inv_freq)
    z = de_rope(y, pos, inv_freq)
    assert np.allclose(z, x, atol=1e-10)


def test_proportional_mapping_covers_teacher():
    mp = proportional_mapping(36, 28)
    flat = sorted({i for v in mp for i in v})
    assert flat[0] == 0
    assert flat[-1] == 35


def test_last_layer_mapping():
    mp = last_layer_mapping(36, 28, k=2)
    assert mp[0] == [34, 35]
    assert mp[-1] == [34, 35]


def test_data_driven_topk():
    sim = synthetic_similarity(36, 28)
    mp = data_driven_topk(sim, k=2)
    assert all(len(v) == 2 for v in mp)


def test_geometry_aware_topk_smooths():
    sim = synthetic_similarity(36, 28)
    mp = geometry_aware_topk(sim, k=2, alpha=0.5)
    assert all(len(v) == 2 for v in mp)


def _toy_kv(n_t=4, n_s=2, S=8, H=2, D=16, seed=0):
    rng = np.random.default_rng(seed)
    kt = rng.standard_normal((n_t, S, H, D))
    ks = kt[:n_s] + 0.1 * rng.standard_normal((n_s, S, H, D))
    return kt.astype(np.float32), ks.astype(np.float32)


def test_ridge_mapper_shape_and_size():
    kt, ks = _toy_kv()
    mp = proportional_mapping(4, 2)
    ridge = RidgeMapper(lam=1e-2)
    ridge.fit(kt, ks, mp)
    pred = ridge.transform(kt, mp)
    assert pred.shape == ks.shape
    assert ridge.n_params > 0


def test_lowrank_mapper_smaller_than_ridge():
    kt, ks = _toy_kv()
    mp = proportional_mapping(4, 2)
    ridge = RidgeMapper()
    ridge.fit(kt, ks, mp)
    lr = LowRankMapper(rank=2)
    lr.fit(kt, ks, mp)
    assert lr.n_params < ridge.n_params