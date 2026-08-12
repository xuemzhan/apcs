"""SourceLayerMixer shape-mismatch 修复回归测试（架构审查续轮）。

bug 现象：layer_map[s] 长度 < top_k 时，mixer.mix 在 tensordot 处抛
ValueError: shape-mismatch for sum —— 整轮 DAG 中 t08 直接挂掉，
其后所有 task 被 §72 阻断。

修复：用 w_eff = w[:k] 让权重与 block 头维一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from apcs.advantage.runner import SourceLayerMixer
from apcs.alignment.runner import proportional_mapping


def test_mix_handles_varying_k():
    """proportional_mapping 返回的 list 长度 ≠ top_k 时，mixer 不应崩。"""
    # n_t=36, n_s=28 → proportional_mapping 产生 1 或 2 长度 list
    layer_map = proportional_mapping(36, 28)
    seen_lens = {len(x) for x in layer_map}
    assert seen_lens != {2}, "应至少有一个非 2 长度的 entry"

    mixer = SourceLayerMixer(36, 28, 8, top_k=2)
    kv_t = np.random.randn(36, 256, 8, 128).astype(np.float32)
    z = mixer.mix(kv_t, layer_map)
    assert z.shape == (28, 256, 8, 128)


def test_mix_with_all_k1():
    """所有 layer_map 长度都为 1 —— 退化情况不应崩。"""
    layer_map = [[i] for i in range(28)]  # 28 个 1-length list
    mixer = SourceLayerMixer(36, 28, 8, top_k=2)
    kv_t = np.random.randn(36, 256, 8, 128).astype(np.float32)
    z = mixer.mix(kv_t, layer_map)
    assert z.shape == (28, 256, 8, 128)


def test_mix_with_k_greater_than_top_k():
    """layer_map 有 > top_k 个 candidate 时，截断到 top_k。"""
    layer_map = [[0, 1, 2, 3] for _ in range(28)]  # 4 个 candidate
    mixer = SourceLayerMixer(36, 28, 8, top_k=2)
    kv_t = np.random.randn(36, 256, 8, 128).astype(np.float32)
    z = mixer.mix(kv_t, layer_map)
    assert z.shape == (28, 256, 8, 128)


def test_mix_preserves_dtypes():
    """dtype 不变（不应被 tensordot 隐式 cast）。"""
    layer_map = proportional_mapping(36, 28)
    mixer = SourceLayerMixer(36, 28, 8, top_k=2)
    kv_t = np.random.randn(36, 256, 8, 128).astype(np.float32)
    z = mixer.mix(kv_t, layer_map)
    assert z.dtype == np.float32
