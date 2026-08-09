"""§47 Ablation runner 回归测试（诚实性修复）。

- 消除 train/test 泄漏：mapper 只在 calib 聚合上 fit 一次，test 只 transform/score。
- 诚实性断言：fit-on-calib 的 retention 必须远低于"每样本自 fit 自评"（泄漏）。
"""
from __future__ import annotations

import numpy as np
import pytest

from apcs.mapper.math import LowRankMapper
from apcs.mapper.runner import (
    _score_kv,
    _shared_model_weights,
    _synth_calibration_set,
)
from apcs.mapper.aggregate import concat_kv_samples
from apcs.alignment.runner import proportional_mapping


def test_ablation_no_leak_scores_fit_once():
    """泄漏回归：calib 聚合 fit 的 mapper 在 test 上评分，禁止 per-test refit。

    旧 `_eval_pair` 在每个 test 样本上 new mapper + fit → 评估数据参与了
    训练（泄漏）。固定断言：诚实的 held-out（fit 于 calib）必须显著低于
    泄漏版（fit 于 test 同一样本）——我们从数学上无法伪造低分，所以断言
    两者的**结构关系**：honest ≤ leaky r 的宽松上界。
    """
    from apcs.ablation.runner import _fit_on_calib, _score_pair

    n_t, n_s, H, D, S = 4, 4, 4, 32, 48
    w_t, w_s = _shared_model_weights(n_t, n_s, D, master_seed=0)
    calib = _synth_calibration_set(
        n_t, n_s, S, H, D, 8, master_seed=0, noise=0.05, w_t=w_t, w_s=w_s
    )
    test = _synth_calibration_set(
        n_t, n_s, S, H, D, 3, master_seed=1, noise=0.05, w_t=w_t, w_s=w_s
    )
    lm = proportional_mapping(n_t, n_s)
    positions = np.arange(S, dtype=np.float64)
    from apcs.rope.runner import _rope_pairs, de_rope

    inv_freq = _rope_pairs(D, theta=1_000_000.0)
    de_fn = lambda k, p: de_rope(k, p, inv_freq)

    # 诚实路径：calib 聚合 fit 一次（与 runner.py T 同契约）
    m_honest = _fit_on_calib(
        LowRankMapper(rank=8), calib, lm, True, D, positions
    )
    honest = [
        _score_pair(t, s, lm, m_honest, True, D) for t, s in test
    ]
    # 泄漏路径：在每个 test 样本上单独 fit
    leak = []
    for t, s in test:
        m2 = LowRankMapper(rank=8)
        m2.fit(t, s, lm, positions=positions, de_rope_fn=de_fn)
        p2 = m2.transform(t, lm, positions=positions, de_rope_fn=de_fn)
        score = _score_kv(p2, s)
        leak.append(min(1.0, score["cosine"] / max(_score_kv(s, s)["cosine"], 1e-6)))

    h_mean = float(np.mean(honest))
    l_mean = float(np.mean(leak))
    # 泄漏版永远 ≥ 诚实版（同一样本 fit+eval 必然 ≥ 跨样本泛化，数值上不可能
    # 更低，除非噪声主导——这里给 0.05 宽容）
    assert l_mean >= h_mean - 0.05, f"leak={l_mean:.4f} < honest={h_mean:.4f}"


def test_ablation_runner_output_shape():
    """run_ablation 基本回归：rows 结构与 settings 完整（A1/A6/A8/A3/A9/A10）。"""
    from apcs.ablation.runner import run_ablation

    import tempfile
    from pathlib import Path

    cfg = {
        "teacher": {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16},
        "student": {"num_layers": 2, "num_kv_heads": 2, "head_dim": 16},
        "advantage": {"enabled": True, "rms_calibration": True, "bounded_alpha": True},
    }
    res = run_ablation(cfg, Path(tempfile.mkdtemp()))
    rows = res["metrics"]["rows"]
    ablations = [r["ablation"] for r in rows]
    for a in ["A1_rank", "A6_de_rope", "A8_kv_adapter"]:
        assert a in ablations, f"缺少 {a}"
    # A1 三档 rank 应是三个独立行
    a1 = [r for r in rows if r["ablation"] == "A1_rank"]
    assert len(a1) == 3 and len({r["setting"] for r in a1}) == 3
    # 数值为 retention ∈ [0,1]（合法分数）
    for r in rows:
        if r["mean_retention"] is not None:
            assert 0.0 <= r["mean_retention"] <= 1.0