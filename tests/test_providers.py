"""Provider 抽象层回归测试（架构审查 P1-3）。

══════════════════════════════════════════════════════════════
锁定以下不变量：
    1. Synthetic KVProvider：calib 与 held-out eval 在 latent seed 空间**联合不交**；
    2. cfg 切换 provider.kind 不需改 runner 接口；
    3. HF 路径**显式 raise**而不是静默回退到合成（§75 诚实性）。
    4. write_provider_manifest 落 provider.json 到 run_dir，方便复现审计。
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from apcs.io import load_config
from apcs.providers import (
    open_providers,
    providers_ctx,
    write_provider_manifest,
)
from apcs.providers.synthetic_kv import SyntheticKVProvider
from apcs.providers.synthetic_score import SyntheticScoreProvider
from apcs.providers.synthetic_timing import SyntheticTimingProvider


def _cfg() -> dict:
    """最小可用 cfg（与 qwen3.yaml 一致；测试不直接读 yaml）。"""
    return {
        "teacher": {"model_id": "T", "num_layers": 4, "num_kv_heads": 2,
                    "head_dim": 8, "num_attention_heads": 4},
        "student": {"model_id": "S", "num_layers": 3, "num_kv_heads": 2,
                    "head_dim": 8, "num_attention_heads": 2},
        "seeds": [0],
    }


def test_synthetic_kv_calibration_and_eval_disjoint():
    """calib 与 eval 样本的 latent seed 必须联合不交（防数据泄漏）。"""
    p = SyntheticKVProvider(seed=0)
    p.open(_cfg())
    try:
        calib = list(p.iter_calibration(5, seed=0))
        eval1 = list(p.iter_eval(5, seed=0))
        # 互斥：calib-0..4 与 eval-0..4 的 sample_id 前缀不同
        assert {s.sample_id for s in calib} == {f"calib-{i}" for i in range(5)}
        assert {s.sample_id for s in eval1} == {f"eval-{i}" for i in range(5)}
        # 在 byte 层面 KV 内容也不重叠（同 shape、不同 latent Z）
        for c, e in zip(calib, eval1):
            assert not np.array_equal(c.kv_t, e.kv_t), "calib 与 eval KV 应当完全不同"
    finally:
        p.close()


def test_synthetic_kv_different_model_pair_changes_w():
    """改 seed 视为不同的「模型对」，应得到一组新的 W_t/W_s（与 held-out 隔离）。"""
    p1 = SyntheticKVProvider(seed=0)
    p1.open(_cfg())
    p2 = SyntheticKVProvider(seed=1)
    p2.open(_cfg())
    try:
        # 不同 (seed) → 不同 W → 同样 latent_seed 的当前样本 KV 不相等
        c1 = list(p1.iter_calibration(1, seed=0))[0]
        c2 = list(p2.iter_calibration(1, seed=0))[0]
        assert not np.array_equal(c1.kv_t, c2.kv_t), "不同 seed 对应不同 W（§32 模型对隔离）"
    finally:
        p1.close()
        p2.close()


def test_synthetic_score_returns_clipped_baseline():
    """ScoreProvider.score = base + offset + noise，clip 到 [0,1]。"""
    sp = SyntheticScoreProvider()
    for _ in range(20):
        s = sp.score("teacher", "ta-0", seed=0)
        assert 0.0 <= s <= 1.0
    # Teacher 期望分最高
    teachers = [sp.score("teacher", "ta-0", seed=0) for _ in range(10)]
    students = [sp.score("student", "ta-0", seed=0) for _ in range(10)]
    assert sum(teachers) / 10 > sum(students) / 10, "teacher > student 由 base 表保证"


def test_synthetic_timing_deterministic():
    """SyntheticTimingProvider.measure 在同 ctx 下完全确定。"""
    tp = SyntheticTimingProvider()
    a = tp.measure(ctx=1024, seed=0)
    b = tp.measure(ctx=1024, seed=0)
    assert a == b
    # 各组件随 ctx 线性增长
    a1 = tp.measure(ctx=1024, seed=0)
    a2 = tp.measure(ctx=2048, seed=0)
    assert a2["teacher_prefill"] > a1["teacher_prefill"]


def test_open_providers_synthetic_smoke():
    """端到端：synthetic 路径能开、能 close、能产出样本。"""
    cfg = _cfg()
    ps = open_providers(cfg, need=("kv", "score", "timing"))
    try:
        assert ps["kv"].kind == "synthetic"
        assert ps["score"].kind == "synthetic"
        assert ps["timing"].kind == "synthetic"
        # 抽样一个 verify 可迭代
        s = next(ps["kv"].iter_calibration(1, seed=0))
        assert s.kv_t.shape[0] == 4
        assert s.kv_s.shape[0] == 3
    finally:
        for p in ps.values():
            if hasattr(p, 'close'):
                p.close()


def test_open_providers_hf_raises_not_silent_fallback():
    """★ 核心：hf 路径必须显式 raise，杜绝『偷偷切回合成』。"""
    cfg = _cfg()
    cfg["provider"] = {"kv": "hf"}
    try:
        open_providers(cfg, need=("kv",))
    except (RuntimeError, NotImplementedError) as e:
        msg = str(e)
        assert "hf" in msg.lower() or "TODO" in msg
        # 关键：sink error 必须指向 provider.json 提醒「把它记录下来」
        assert "provider" in msg
    else:
        raise AssertionError("hf 路径不应静默回退到合成，必须显式 raise")


def test_providers_ctx_closes_on_exit():
    """contextmanager 退出时自动 close（防止 leak）。"""
    cfg = _cfg()
    with providers_ctx(cfg, need=("kv",)) as ps:
        kvp = ps["kv"]
        assert kvp._W_t is not None  # 内部状态
    # 退出 with → W 已 None
    assert kvp._W_t is None


def test_write_provider_manifest_round_trip(tmp_path):
    """provider.json 落盘后能 JSON 序列化（§64 复现审计）。"""
    cfg = _cfg()
    with providers_ctx(cfg, need=("kv", "score", "timing")) as ps:
        manifest = write_provider_manifest(tmp_path, **ps)
    # 落盘验证
    p_json = tmp_path / "provider.json"
    assert p_json.exists()
    payload = json.loads(p_json.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"kv", "score", "timing"}
    for name in payload:
        assert payload[name]["kind"] == "synthetic"
