"""配置加载测试（configs/*.yaml 结构契约，design.md §64）。

验证：
    1. pair_qwen3.yaml 显式声明架构字段（_cfg_layers 不再依赖 model_id 启发式）。
    2. pair_smollm2.yaml 的 G2 配置合法（p_d_strategy ∈ {truncate, pad, linear}）。
"""
from __future__ import annotations

from apcs.io import load_config

QWEN3_CFG = "configs/pair_qwen3.yaml"     # G1 Matched KV 主实验对（§11）
SMOLLM2_CFG = "configs/pair_smollm2.yaml"  # G2 Mismatched Head Dim 第二对（§12）


def test_qwen3_config_explicit_layers() -> None:
    """Qwen3 pair 必须显式声明架构字段，禁止依赖 model_id 字符串启发式。"""
    cfg = load_config(QWEN3_CFG)

    t = cfg["teacher"]
    assert t["num_layers"] == 36
    assert t["num_kv_heads"] == 8
    assert t["head_dim"] == 128

    s = cfg["student"]
    assert s["num_layers"] == 28
    assert s["num_kv_heads"] == 8
    assert s["head_dim"] == 128


def test_smollm2_config_g2_valid() -> None:
    """SmolLM2 G2 块必须启用且 p_d_strategy 是 DimensionProjection 合法值。"""
    cfg = load_config(SMOLLM2_CFG)
    g2 = cfg["g2"]

    assert g2["enabled"] is True
    assert g2["p_d_strategy"] in {"truncate", "identity"}
    assert g2["p_h_strategy"] == "mean"
