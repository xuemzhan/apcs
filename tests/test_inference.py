"""§53 单卡推理管线骨架测试（apcs/inference）。

对应 design.md：
    §52 禁止 1 —— Student 在真实 inference 阶段不重新读 X（zero prefill）
    §53 单卡执行顺序 —— Teacher Load → Forward → Capture → CPU Offload →
        Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode
    §75 诚实性 —— numpy 后端是小型假模型（offline_demo），torch 后端为
        接口骨架，必须显式 raise，不得静默跑 mock。

RED → GREEN 目标：
- test_numpy_backend_capture_and_inject_flow：
    numpy 后端端到端 run()，断言 n_generated == n_gen、
    decode 阶段 zero prefill（统计断言：student_prefill_calls == 0
    且 student_decode_calls == n_gen）、注入 KV 形状与 layer_map 匹配。
- test_torch_backend_raises_not_implemented_without_gpu：
    torch 骨架无 GPU 实现 → 显式 raise NotImplementedError。
"""
from __future__ import annotations

import numpy as np
import pytest

from apcs.alignment.runner import proportional_mapping
from apcs.inference import HandoffPipeline, NumpyBackend, TorchBackend
from apcs.mapper.math import RidgeMapper

# §53 骨架小模型常量：Teacher 4 层 → Student 2 层，seq=16, H=2, head_dim=8, vocab=32
N_T, N_S, SEQ, H, D, VOCAB = 4, 2, 16, 2, 8, 32


def _cfg() -> dict:
    """numpy 后端需要的最小模型参数（缺参时后端必须显式 raise）。"""
    return {
        "teacher": {
            "num_layers": N_T,
            "num_kv_heads": H,
            "head_dim": D,
            "vocab_size": VOCAB,
        },
        "student": {
            "num_layers": N_S,
            "num_kv_heads": H,
            "head_dim": D,
            "vocab_size": VOCAB,
        },
    }


def _fit_mapper() -> RidgeMapper:
    """用与 test_mapper 一致的合成 calibration 数据拟合真实 RidgeMapper。

    证明管线接受"真实 mapper"端到端跑（§20 Ridge baseline 同一接口）。
    """
    rng = np.random.default_rng(0)
    kv_t = rng.standard_normal((N_T, SEQ, H, D)).astype(np.float32)
    w = rng.standard_normal((D, D)).astype(np.float32) / np.sqrt(D)
    kv_s = (
        kv_t[:N_S] @ w + 0.001 * rng.standard_normal((N_S, SEQ, H, D))
    ).astype(np.float32)
    mapper = RidgeMapper(lam=1e-3)
    mapper.fit(kv_t, kv_s, proportional_mapping(N_T, N_S))
    return mapper


def test_numpy_backend_capture_and_inject_flow(tmp_path):
    """numpy 后端端到端：capture → map → inject → decode 链路真实可运行。

    Given: numpy 后端 + 真实 RidgeMapper + proportional layer_map
    When:  run(teacher_tokens, student_prefix, n_gen=5)
    Then:
        - metrics["n_generated"] == n_gen
        - §52 禁止 1：decode 阶段无 prefill（student_prefill_calls == 0，
          student_decode_calls == n_gen）→ gates.zero_prefill == True
        - 注入 KV 形状 == (n_s, S, H, D)，层级树与 layer_map 匹配
        - 产物 metrics.json / summary.md 写出
    """
    backend = NumpyBackend()
    mapper = _fit_mapper()
    layer_map = proportional_mapping(N_T, N_S)
    pipe = HandoffPipeline(backend, mapper, layer_map, _cfg(), out_dir=tmp_path)

    rng = np.random.default_rng(1)
    teacher_tokens = rng.integers(0, VOCAB, size=SEQ)
    student_prefix = np.array([int(teacher_tokens[-1])])
    n_gen = 5

    metrics = pipe.run(teacher_tokens, student_prefix, n_gen=n_gen)

    # §52：n_generated == n_gen
    assert metrics["n_generated"] == n_gen
    assert len(metrics["tokens"]) == n_gen
    assert all(0 <= t < VOCAB for t in metrics["tokens"])

    # §52 禁止 1 统计断言：decode 迭代 n_gen 次、绝无额外 prefill
    assert metrics["student_prefill_calls"] == 0
    assert metrics["student_decode_calls"] == n_gen
    assert metrics["gates"]["zero_prefill"] is True

    # 注入 KV 形状正确、层级树与 layer_map 匹配
    assert metrics["n_student_layers"] == N_S
    assert metrics["injected_kv_shape"] == [N_S, SEQ, H, D]
    assert metrics["kv_s_shape"] == [N_S, SEQ, H, D]
    assert metrics["layer_map"] == layer_map

    # 产物写出（§63 风格）
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "summary.md").exists()


def test_torch_backend_raises_without_model_id(tmp_path):
    """torch 后端为真实 GPU 路径：缺少 model_id 必须显式 raise，不得静默 mock。

    Given: TorchBackend（真实实现，但 run 参数缺 model_id）
    When:  run() 触发 §53 第一步 Teacher Load
    Then:  raise NotImplementedError 且消息含 "torch"/"model_id" 之一
    """
    backend = TorchBackend()
    assert backend.name == "torch"
    pipe = HandoffPipeline(backend, mapper=None, layer_map=[], cfg=_cfg(), out_dir=tmp_path)

    teacher_tokens = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(NotImplementedError) as ei:
        pipe.run(teacher_tokens, student_prefix=np.array([1]), n_gen=2)
    msg = str(ei.value)
    assert "model_id" in msg, f"缺 model_id 时应提示显式传参：{msg}"
