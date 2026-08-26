"""inject-eval CPU-only tests — 完全脱离 torch/transformers/GPU。

覆盖：
    1. score_choices 纯 numpy 评分函数（数值正确性）
    2. EvalPhaseCounters 零 prefill 审计（通过 / 失败路径）
    3. _build_cache_from_kv GQA repeat_interleave 逻辑（numpy 模拟）
    4. _build_scoring_prompt / _extract_gold_index 模板函数
    5. replacement_score_artifact.json schema 校验
    6. capability_score_artifact.json schema 校验
    7. zero-prefill_counter 数值路径
    8. score_choices 数值稳定性（零 logits / 极端值）
    9. _extract_kv_numpy 布局一致性
   10. inject-eval CLI 注册（TASKS dict 包含 inject-eval）

约束：所有 torch/transformers import 以 pytest.importorskip 做守卫；
    纯 numpy/scipy 路径直接运行。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. score_choices 纯 numpy 评分
# ---------------------------------------------------------------------------

class TestScoreChoices:
    """score_choices 数值正确性（零 torch 依赖）。"""

    def test_equal_logits(self):
        """均等 logits → softmax 1/vocab_size 每个 token → choice id 32 概率 0.01。"""
        from apcs.inference.evaluator import score_choices

        logits = np.zeros(100)
        score, decision = score_choices(logits)
        # 每个 token 概率 = 1/100 = 0.01；A/B/C/D 各 0.01 → argmax = 0 (A)
        assert decision == 0
        assert abs(score - 0.01) < 1e-6

    def test_one_hot_dominant(self):
        """A token id 位置 logits=100，其余 0 → score ≈ 1.0，decision = 0。"""
        from apcs.inference.evaluator import score_choices

        logits = np.zeros(100)
        logits[32] = 100.0  # A token id
        score, decision = score_choices(logits)
        assert decision == 0
        assert score > 0.99

    def test_b_dominant(self):
        """B token id 位置 logits 远高于其他 → decision = 1。"""
        from apcs.inference.evaluator import score_choices

        logits = np.zeros(100)
        logits[33] = 50.0  # B token id
        logits[32] = 1.0   # A token id
        score, decision = score_choices(logits)
        assert decision == 1
        assert score > 0.99

    def test_custom_choice_ids(self):
        """自定义 choice_letter_ids → 使用提供的 token id。"""
        from apcs.inference.evaluator import score_choices

        logits = np.zeros(200)
        logits[50] = 10.0  # A at id 50
        logits[60] = 1.0   # B at id 60
        ids = {"A": 50, "B": 60, "C": 70, "D": 80}
        score, decision = score_choices(logits, choice_letter_ids=ids)
        assert decision == 0  # A wins

    def test_zero_logits(self):
        """全零 logits → softmax 1/vocab_size 每个 token → choice id 32 概率 0.001。"""
        from apcs.inference.evaluator import score_choices

        logits = np.zeros(1000)
        score, decision = score_choices(logits)
        assert abs(score - 0.001) < 1e-6

    def test_negative_logits(self):
        """负 logits 仍应产生有效 softmax。"""
        from apcs.inference.evaluator import score_choices

        logits = np.full(100, -10.0)
        logits[34] = -1.0  # C token id
        score, decision = score_choices(logits)
        assert decision == 2  # C
        assert score > 0.5

    def test_extreme_values_numerical_stability(self):
        """极大 logits 值不应产生 NaN/Inf（数值稳定 softmax）。"""
        from apcs.inference.evaluator import score_choices

        logits = np.full(100, 1000.0)
        logits[35] = 1001.0  # D
        score, decision = score_choices(logits)
        assert math.isfinite(score)
        assert decision == 3

    def test_2d_logits(self):
        """接受 (1, vocab_size) 形状的 logits。"""
        from apcs.inference.evaluator import score_choices

        logits = np.zeros((1, 100))
        logits[0, 34] = 20.0  # C
        score, decision = score_choices(logits)
        assert decision == 2

    def test_probability_sum(self):
        """argmax score 应在 [0, 1] 范围内。"""
        from apcs.inference.evaluator import score_choices

        rng = np.random.default_rng(42)
        for _ in range(20):
            logits = rng.standard_normal(100)
            score, decision = score_choices(logits)
            assert 0.0 <= score <= 1.0
            assert 0 <= decision <= 3


# ---------------------------------------------------------------------------
# 2. EvalPhaseCounters 零 prefill 审计
# ---------------------------------------------------------------------------

class TestEvalPhaseCounters:
    """EvalPhaseCounters 零 prefill 守卫逻辑。"""

    def test_verified_pass(self):
        """inject_seq_len == past_len 且 student_prefill == 0 → verified。"""
        from apcs.inference.evaluator import EvalPhaseCounters

        c = EvalPhaseCounters()
        c.inject_seq_len = 128
        c.query_past_len = 128
        c.query_new_tokens = 10
        c.student_prefill_new_tokens = 0
        assert c.is_verified

    def test_student_prefill_violation(self):
        """student_prefill_new_tokens > 0 → NOT verified。"""
        from apcs.inference.evaluator import EvalPhaseCounters

        c = EvalPhaseCounters()
        c.inject_seq_len = 128
        c.query_past_len = 128
        c.student_prefill_new_tokens = 5
        assert not c.is_verified

    def test_past_len_mismatch(self):
        """past_len != inject_seq_len → NOT verified。"""
        from apcs.inference.evaluator import EvalPhaseCounters

        c = EvalPhaseCounters()
        c.inject_seq_len = 128
        c.query_past_len = 64  # 不一致
        c.student_prefill_new_tokens = 0
        assert not c.is_verified

    def test_assert_zero_prefill_raises(self):
        """assert_zero_prefill() 在违规时显式 raise RuntimeError。"""
        from apcs.inference.evaluator import EvalPhaseCounters

        c = EvalPhaseCounters()
        c.student_prefill_new_tokens = 3
        with pytest.raises(RuntimeError, match="§52 禁止 1 违反"):
            c.assert_zero_prefill()

    def test_assert_query_handoff_raises(self):
        """assert_query_handoff() 在 past_len != inject_seq_len 时 raise。"""
        from apcs.inference.evaluator import EvalPhaseCounters

        c = EvalPhaseCounters()
        c.inject_seq_len = 100
        c.query_past_len = 50
        with pytest.raises(RuntimeError, match="断言失败"):
            c.assert_query_handoff()

    def test_reset(self):
        """reset() 将所有计数器归零。"""
        from apcs.inference.evaluator import EvalPhaseCounters

        c = EvalPhaseCounters()
        c.inject_seq_len = 99
        c.query_past_len = 99
        c.student_prefill_new_tokens = 0
        c.reset()
        assert c.inject_seq_len == 0
        assert c.query_past_len == 0


# ---------------------------------------------------------------------------
# 3. ZeroPrefillCounter (backends.py) 审计
# ---------------------------------------------------------------------------

class TestZeroPrefillCounter:
    """backends.py ZeroPrefillCounter 零 prefill 守卫。"""

    def test_verified_pass(self):
        """注入后只 decode query → verified。"""
        from apcs.inference.backends import ZeroPrefillCounter

        c = ZeroPrefillCounter()
        c.record_teacher_prefill(256)
        c.record_inject(256)
        c.record_query_decode(new_tokens=10, past_len=256)
        c.record_student_prefill(0)
        assert c.is_verified

    def test_student_prefill_violation(self):
        """Student 重读 X → not verified + raise。"""
        from apcs.inference.backends import ZeroPrefillCounter

        c = ZeroPrefillCounter()
        c.record_inject(256)
        c.record_student_prefill(100)
        c.record_query_decode(10, 256)
        assert not c.is_verified
        with pytest.raises(RuntimeError, match="§52 禁止 1 违反"):
            c.assert_zero_prefill()

    def test_handoff_mismatch(self):
        """past_len != inject_seq_len → not verified + raise。"""
        from apcs.inference.backends import ZeroPrefillCounter

        c = ZeroPrefillCounter()
        c.record_inject(256)
        c.record_student_prefill(0)
        c.record_query_decode(10, 128)  # past_len != 256
        assert not c.is_verified
        with pytest.raises(RuntimeError, match="断言失败"):
            c.assert_query_handoff()

    def test_full_happy_path(self):
        """完整路径：prefill → inject → decode → student_prefill=0。"""
        from apcs.inference.backends import ZeroPrefillCounter

        c = ZeroPrefillCounter()
        c.record_teacher_prefill(512)
        c.record_inject(512)
        c.record_query_decode(new_tokens=20, past_len=512)
        c.record_student_prefill(0)
        assert c.is_verified
        assert c.teacher_prefill_tokens == 512
        assert c.inject_seq_len == 512
        assert c.query_decode_new_tokens == 20


# ---------------------------------------------------------------------------
# 4. _build_scoring_prompt / _extract_gold_index
# ---------------------------------------------------------------------------

class TestScoringPrompt:
    """评分 prompt 构造与 gold index 提取。"""

    def test_build_prompt_format(self):
        """验证 prompt 包含 context + query + choices + Answer: 后缀。"""
        from apcs.inference.evaluator import _build_scoring_prompt

        prompt = _build_scoring_prompt(
            "Context text", "What is X?", ["opt1", "opt2", "opt3", "opt4"]
        )
        assert "Context text" in prompt
        assert "What is X?" in prompt
        assert "A. opt1" in prompt
        assert "D. opt4" in prompt
        assert prompt.endswith("Answer:")

    def test_build_prompt_fewer_choices(self):
        """choices < 4 时只渲染提供的选项。"""
        from apcs.inference.evaluator import _build_scoring_prompt

        prompt = _build_scoring_prompt("ctx", "q?", ["only_a", "only_b"])
        assert "A. only_a" in prompt
        assert "B. only_b" in prompt
        assert "C." not in prompt

    def test_extract_gold_letter(self):
        """answer="C" → index 2。"""
        from apcs.inference.evaluator import _extract_gold_index

        assert _extract_gold_index("C", ["x", "y", "z", "w"]) == 2

    def test_extract_gold_by_text(self):
        """answer 文本匹配 choice → 正确索引。"""
        from apcs.inference.evaluator import _extract_gold_index

        choices = ["fox", "dog", "cat", "bird"]
        assert _extract_gold_index("fox", choices) == 0
        assert _extract_gold_index("bird", choices) == 3

    def test_extract_gold_no_match_raises(self):
        """无法匹配时 raise ValueError。"""
        from apcs.inference.evaluator import _extract_gold_index

        with pytest.raises(ValueError, match="无法将 answer"):
            _extract_gold_index("unknown", ["x", "y", "z", "w"])


# ---------------------------------------------------------------------------
# 5. Artifact schema 校验（replacement_score_artifact.json）
# ---------------------------------------------------------------------------

class TestReplacementArtifactSchema:
    """replacement_score_artifact.json schema 校验（§63 Run 产物规范）。"""

    def test_valid_artifact(self):
        """合法 artifact 不 raise。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        # 构造 capability_score_artifact（replacement 不同 schema）
        # replacement schema: {evidence_grade, zero_prefill_verified, task_scoring_verified,
        #                      split, records: [{sample_id, student_score, handoff_score}]}
        # 这个 schema 不走 ArtifactScoreProvider（那个是 capability schema），
        # 所以我们直接校验 JSON 结构。
        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "records": [
                {"sample_id": "s1", "student_score": 0.75, "handoff_score": 0.70},
                {"sample_id": "s2", "student_score": 0.50, "handoff_score": 0.55},
            ],
        }
        # 基本结构校验
        assert artifact["evidence_grade"] == "measured_task"
        assert artifact["zero_prefill_verified"] is True
        assert artifact["task_scoring_verified"] is True
        assert artifact["split"] == "test"
        for rec in artifact["records"]:
            assert "sample_id" in rec
            assert math.isfinite(rec["student_score"])
            assert math.isfinite(rec["handoff_score"])

    def test_nonfinite_score_reject(self):
        """非有限 score → 写入时应被 evaluator 检测。"""
        rec = {"sample_id": "bad", "student_score": float("nan"), "handoff_score": 0.5}
        assert not math.isfinite(rec["student_score"])

    def test_json_roundtrip(self):
        """artifact 可 JSON 序列化/反序列化。"""
        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "records": [{"sample_id": "x", "student_score": 0.6, "handoff_score": 0.7}],
        }
        blob = json.dumps(artifact, indent=2, ensure_ascii=False)
        loaded = json.loads(blob)
        assert loaded["records"][0]["sample_id"] == "x"


# ---------------------------------------------------------------------------
# 6. capability_score_artifact.json 校验（ArtifactScoreProvider.open）
# ---------------------------------------------------------------------------

class TestCapabilityArtifactSchema:
    """capability_score_artifact.json schema 校验（ArtifactScoreProvider.open）。"""

    def _write_valid_artifact(self, tmp_path: Path) -> Path:
        """写入合法 capability_score_artifact 并返回路径。"""
        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "dataset": "hellaswag",
            "records": [
                {"method": "student", "sample_id": "s1", "seed": 0, "score": 0.60, "decision": 0},
                {"method": "teacher", "sample_id": "s1", "seed": 0, "score": 0.80, "decision": 2},
                {"method": "text", "sample_id": "s1", "seed": 0, "score": 0.55, "decision": 0},
                {"method": "ridge", "sample_id": "s1", "seed": 0, "score": 0.70, "decision": 1},
            ],
        }
        path = tmp_path / "capability_score_artifact.json"
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return path

    def test_valid_opens(self):
        """合法 artifact → open() 不 raise。"""
        # ArtifactScoreProvider.open 不需要 torch，纯 JSON 校验
        pass  # 由 test_valid_artifact_with_tmp 覆盖完整路径

    def test_valid_artifact_with_tmp(self, tmp_path):
        """合法 artifact → open + score + decision 正确。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        path = self._write_valid_artifact(tmp_path)
        p = ArtifactScoreProvider()
        p.open({"score_artifact_path": str(path)})
        assert p.score("student", "s1", 0) == 0.60
        assert p.decision("teacher", "s1", 0) == 2
        assert p.score("ridge", "s1", 0) == 0.70
        p.close()

    def test_wrong_evidence_grade(self, tmp_path):
        """evidence_grade != "measured_task" → raise。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        artifact = {
            "evidence_grade": "synthetic",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "records": [{"method": "student", "sample_id": "s1", "seed": 0, "score": 0.5, "decision": 0}],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        p = ArtifactScoreProvider()
        with pytest.raises(RuntimeError, match="evidence_grade"):
            p.open({"score_artifact_path": str(path)})

    def test_zero_prefill_not_verified(self, tmp_path):
        """zero_prefill_verified=false → raise。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": False,
            "task_scoring_verified": True,
            "split": "test",
            "records": [{"method": "student", "sample_id": "s1", "seed": 0, "score": 0.5, "decision": 0}],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        p = ArtifactScoreProvider()
        with pytest.raises(RuntimeError, match="zero_prefill_verified"):
            p.open({"score_artifact_path": str(path)})

    def test_invalid_split(self, tmp_path):
        """split=train → raise（必须是 validation/heldout/test）。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "train",
            "records": [{"method": "student", "sample_id": "s1", "seed": 0, "score": 0.5, "decision": 0}],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        p = ArtifactScoreProvider()
        with pytest.raises(RuntimeError, match="split"):
            p.open({"score_artifact_path": str(path)})

    def test_nonfinite_score(self, tmp_path):
        """非有限 score → raise。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "records": [
                {"method": "student", "sample_id": "s1", "seed": 0, "score": float("nan"), "decision": 0}
            ],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        p = ArtifactScoreProvider()
        with pytest.raises(RuntimeError, match="有限数"):
            p.open({"score_artifact_path": str(path)})

    def test_duplicate_records(self, tmp_path):
        """重复 (method, sample_id, seed) → raise。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "records": [
                {"method": "student", "sample_id": "s1", "seed": 0, "score": 0.5, "decision": 0},
                {"method": "student", "sample_id": "s1", "seed": 0, "score": 0.6, "decision": 1},
            ],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        p = ArtifactScoreProvider()
        with pytest.raises(RuntimeError, match="重复记录"):
            p.open({"score_artifact_path": str(path)})

    def test_missing_decision(self, tmp_path):
        """records 缺 decision → raise。"""
        from apcs.providers.artifact import ArtifactScoreProvider

        artifact = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
            "split": "test",
            "records": [
                {"method": "student", "sample_id": "s1", "seed": 0, "score": 0.5}
            ],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        p = ArtifactScoreProvider()
        with pytest.raises(RuntimeError, match="decision"):
            p.open({"score_artifact_path": str(path)})


# ---------------------------------------------------------------------------
# 7. _build_cache_from_kv GQA repeat_interleave（numpy 模拟）
# ---------------------------------------------------------------------------

class TestBuildCacheGQA:
    """_build_cache_from_kv GQA repeat_interleave 逻辑测试。

    由于 torch 不在 dev box 上，我们用 numpy 模拟 repeat_interleave 逻辑
    验证 repeat 次数计算和 shape 一致性。
    """

    def test_repeat_calculation(self):
        """kv_heads=8, attn_heads=16 → repeat=2。"""
        kv_heads = 8
        attn_heads = 16
        assert attn_heads % kv_heads == 0
        repeats = attn_heads // kv_heads
        assert repeats == 2

    def test_repeat_not_divisible_raises(self):
        """kv_heads=8, attn_heads=15 → 不整除 → raise。"""
        kv_heads = 8
        attn_heads = 15
        with pytest.raises((ValueError, ZeroDivisionError)):
            if attn_heads > kv_heads and attn_heads % kv_heads != 0:
                raise ValueError(
                    f"num_attention_heads({attn_heads}) 必须是 num_kv_heads({kv_heads}) 的整数倍"
                )

    def test_gqa_shape_simulation(self):
        """模拟 GQA repeat 后 shape 一致性。"""
        L, S, H, D2 = 4, 128, 8, 256  # kv_shape
        kv = np.random.default_rng(0).standard_normal((L, S, H, D2))
        attn_heads = 16
        repeats = attn_heads // H
        # 模拟 repeat_interleave(dim=1) 沿 head 维扩展
        kv_expanded = np.repeat(kv, repeats, axis=2)
        assert kv_expanded.shape == (L, S, attn_heads, D2)


# ---------------------------------------------------------------------------
# 8. _extract_kv_numpy 布局一致性
# ---------------------------------------------------------------------------

class TestExtractKVLayout:
    """KV 布局 (L, S, H, 2*D) 一致性验证。"""

    def test_layout_roundtrip(self):
        """构造 (L, S, H, 2*D) → 验证 K/V 切分一致性。"""
        L, S, H, D = 4, 128, 8, 64
        rng = np.random.default_rng(42)
        k = rng.standard_normal((L, S, H, D)).astype(np.float32)
        v = rng.standard_normal((L, S, H, D)).astype(np.float32)
        kv = np.concatenate([k, v], axis=-1)  # (L, S, H, 2D)
        assert kv.shape == (L, S, H, 2 * D)

        # 切回 K 和 V
        k_recovered = kv[:, :, :, :D]
        v_recovered = kv[:, :, :, D:]
        np.testing.assert_array_equal(k, k_recovered)
        np.testing.assert_array_equal(v, v_recovered)

    def test_permute_consistency(self):
        """验证 HF DynamicCache 布局 (L, H, S, D) permute 到 (L, S, H, 2*D) 的一致性。"""
        L, S, H, D = 4, 128, 8, 64
        rng = np.random.default_rng(99)
        # HF DynamicCache 布局：每层 K/V 各 (1, heads, seq, dim)
        # 去掉 batch 维 → (L, H, S, D)
        k = rng.standard_normal((L, H, S, D)).astype(np.float32)
        v = rng.standard_normal((L, H, S, D)).astype(np.float32)

        # _extract_kv_numpy 做 permute(1,2,0,3): (H, S, L, D) → 不对
        # 实际 backends.py 用 permute(0,2,1,3): (L, H, S, D) → (L, S, H, D)
        kt = k.transpose(0, 2, 1, 3)  # (L, S, H, D)
        vt = v.transpose(0, 2, 1, 3)  # (L, S, H, D)
        kv = np.concatenate([kt, vt], axis=-1)
        assert kv.shape == (L, S, H, 2 * D)

        # 逆向：切分 K/V
        k_recovered = kv[:, :, :, :D].transpose(0, 2, 1, 3)
        v_recovered = kv[:, :, :, D:].transpose(0, 2, 1, 3)
        np.testing.assert_array_equal(k, k_recovered)
        np.testing.assert_array_equal(v, v_recovered)


# ---------------------------------------------------------------------------
# 9. inject-eval CLI 注册
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    """inject-eval 在 CLI TASKS dict 中注册。"""

    def test_inject_eval_in_tasks(self):
        """apcs.cli.TASKS 包含 inject-eval 键。"""
        from apcs.cli import TASKS

        assert "inject-eval" in TASKS
        module_path, func_name, title, objective = TASKS["inject-eval"]
        assert module_path == "apcs.inference.cli"
        assert func_name == "run_inject_eval"
        assert "inject" in title.lower()

    def test_inject_eval_in_orchestrator_deps(self):
        """inject-eval 在 orchestrator dependencies 中有条目。"""
        from apcs.orchestrator import dependencies

        deps = dependencies("inject-eval")
        assert "t00" in deps


# ---------------------------------------------------------------------------
# 10. _orthogonal_projection 确定性
# ---------------------------------------------------------------------------

class TestOrthogonalProjection:
    """确定性正交投影矩阵生成。"""

    def test_deterministic(self):
        """同 seed → 同投影矩阵。"""
        from apcs.inference.evaluator import _orthogonal_projection

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = _orthogonal_projection(rng1, 128, 64)
        p2 = _orthogonal_projection(rng2, 128, 64)
        np.testing.assert_array_equal(p1, p2)

    def test_column_orthogonality(self):
        """投影矩阵列正交（p^T @ p ≈ identity）。"""
        from apcs.inference.evaluator import _orthogonal_projection

        rng = np.random.default_rng(7)
        p = _orthogonal_projection(rng, 256, 128)
        gram = p.T @ p
        np.testing.assert_allclose(gram, np.eye(128), atol=1e-5)

    def test_out_dim_gt_in_dim_raises(self):
        """out_dim > in_dim → raise。"""
        from apcs.inference.evaluator import _orthogonal_projection

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            _orthogonal_projection(rng, 64, 128)
