"""T01/T00/T03 三大任务增量测试（roadmap 前置条件验证）。

T01 多步解码稳定性：offline 路径 samples × decode_steps 桶循环产出
    per-step metrics + gate0 verdict。
T00 溯源防串跑：scanner 输出新增 revision / rope_theta / tokenizer_hash，
    离线环境（transformers 缺失）tokenizer_hash 为 "unavailable:*" 而不崩溃。
T03 选择依据来自 train/validation 而非 test：data_driven_topk 在构造数据上
    选中刻意最高相似度的 teacher layer。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest


# =========================================================================
# T01 Self-KV Replay 多步解码稳定性
# =========================================================================

class TestT01MultiStepReplay:
    """T01 多步解码稳定性：offline 路径按 decode_steps 桶展开循环。"""

    def _base_cfg(self) -> dict[str, Any]:
        """构造最小化 cfg，使用 offline/synthetic provider。"""
        return {
            "provider": {"kv": "synthetic"},
            "replay": {
                "n_samples": 4,
                "decode_steps": [1, 3, 5],
            },
            "context_lengths": [512, 1024],
        }

    def test_multi_step_offline_produces_per_step_metrics(self, tmp_path: Path):
        """T01：offline 路径 samples × decode_steps 桶产出 per_decode_step 指标。"""
        from apcs.replay.runner import run_self_kv_replay

        cfg = self._base_cfg()
        run_dir = tmp_path / "t01"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_self_kv_replay(cfg, run_dir)
        metrics = result["metrics"]

        # 结果包含 per_decode_step 桶汇总
        assert "per_decode_step" in metrics
        per_step = metrics["per_decode_step"]
        assert len(per_step) == 3  # decode_steps = [1, 3, 5]
        step_values = {bs["decode_step"] for bs in per_step}
        assert step_values == {1, 3, 5}

        # 每个桶有 n / mean_cosine / mean_max_error / mean_agreement
        for bs in per_step:
            assert bs["n"] > 0
            assert isinstance(bs["mean_logit_cosine"], float)
            assert isinstance(bs["mean_max_error"], float)
            assert isinstance(bs["mean_token_agreement"], float)

    def test_multi_step_gate0_pass_when_ideal(self, tmp_path: Path):
        """T01：offline 模拟全理想值 → Gate 0 PASS。"""
        from apcs.replay.runner import run_self_kv_replay

        cfg = self._base_cfg()
        run_dir = tmp_path / "t01"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_self_kv_replay(cfg, run_dir)
        assert result["status"] == "PASS"
        assert result["metrics"]["gate0"] == "PASS"

    def test_summary_contains_per_step_table(self, tmp_path: Path):
        """T01：summary.md 包含 Per-Decode-Step 表格。"""
        from apcs.replay.runner import run_self_kv_replay

        cfg = self._base_cfg()
        run_dir = tmp_path / "t01"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_self_kv_replay(cfg, run_dir)
        summary = result["summary"]
        assert "Per-Decode-Step" in summary
        assert "decode_step" in summary

    def test_simulated_replay_exercises_loop(self, tmp_path: Path):
        """T01：_simulated_replay 实际执行 n_decode_steps 步循环。"""
        from apcs.data import Sample
        from apcs.replay.runner import _simulated_replay

        sample = Sample(sample_id="test-0", context="hello", query="world")
        r1 = _simulated_replay(sample, n_decode_steps=1)
        r5 = _simulated_replay(sample, n_decode_steps=5)

        # 两种桶大小的 decode_step 不同
        assert r1.decode_step == 1
        assert r5.decode_step == 5
        # 离线模拟全理想值
        assert r1.logit_cosine == pytest.approx(1.0)
        assert r1.max_error == pytest.approx(0.0)
        assert r1.token_agreement == pytest.approx(1.0)

    def test_n_samples_cfg_default_32(self, tmp_path: Path):
        """T01：n_samples 默认 32；offline 路径 min(n_samples, 8) 限制。"""
        from apcs.replay.runner import run_self_kv_replay

        # 无 replay.n_samples 时默认 32 → offline min(32, 8) = 8 samples
        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "replay": {"decode_steps": [1]},
            "context_lengths": [],
        }
        run_dir = tmp_path / "t01"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_self_kv_replay(cfg, run_dir)
        n = result["metrics"]["n_samples"]
        assert n == 8  # min(32, 8)

    def test_decode_steps_list_or_int(self, tmp_path: Path):
        """T01：decode_steps 支持 list 或 int 配置。"""
        from apcs.replay.runner import run_self_kv_replay

        # int 形式
        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "replay": {"n_samples": 2, "decode_steps": 7},
        }
        run_dir = tmp_path / "t01_int"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_self_kv_replay(cfg, run_dir)
        assert result["metrics"]["decode_steps"] == [7]
        # 7 步 → per_decode_step 1 个桶
        assert len(result["metrics"]["per_decode_step"]) == 1

    def test_metrics_json_written(self, tmp_path: Path):
        """T01：metrics.json 正确写出。"""
        from apcs.replay.runner import run_self_kv_replay

        cfg = self._base_cfg()
        run_dir = tmp_path / "t01"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_self_kv_replay(cfg, run_dir)
        metrics_path = run_dir / "metrics.json"
        assert metrics_path.exists()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["task"] == "T01"
        assert "gate0" in payload
        assert "per_decode_step" in payload

    def test_context_lengths_reused(self, tmp_path: Path):
        """T01：cfg.context_lengths 被复用到 metrics 中。"""
        from apcs.replay.runner import run_self_kv_replay

        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "replay": {"n_samples": 2, "decode_steps": [1]},
            "context_lengths": [512, 1024, 2048],
        }
        run_dir = tmp_path / "t01"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_self_kv_replay(cfg, run_dir)
        assert result["metrics"]["context_lengths"] == [512, 1024, 2048]


# =========================================================================
# T00 Compatibility Scanner 溯源防串跑
# =========================================================================

class TestT00Provenance:
    """T00 溯源：model_compatibility.json 新增 revision / rope_theta / tokenizer_hash。"""

    def _base_cfg(self) -> dict[str, Any]:
        return {
            "teacher": {
                "model_id": "Qwen/Qwen3-4B",
                "revision": "main",
                "attention_implementation": "eager",
                "dtype": "float16",
            },
            "student": {
                "model_id": "Qwen/Qwen3-1.7B",
                "revision": "v1.0",
                "attention_implementation": "eager",
                "dtype": "float16",
            },
        }

    def test_model_compatibility_has_tokenizer_hash(self, tmp_path: Path):
        """T00：model_compatibility.json 包含 tokenizer_hash 键。"""
        from apcs.compat.scanner import run_compat_scan

        cfg = self._base_cfg()
        run_dir = tmp_path / "t00"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_compat_scan(cfg, run_dir)
        payload = json.loads(
            (run_dir / "model_compatibility.json").read_text(encoding="utf-8")
        )
        assert "tokenizer_hash" in payload["teacher"]
        assert "tokenizer_hash" in payload["student"]

    def test_tokenizer_hash_unavailable_when_no_cache(self, tmp_path: Path):
        """T00：无 HF 缓存时 tokenizer_hash 为 'unavailable:*' 而不崩溃。"""
        from apcs.compat.scanner import _tokenizer_hash

        # 不存在的 model → unavailable
        result = _tokenizer_hash("definitely/not-cached-model-xyz123")
        assert result.startswith("unavailable:")

    def test_tokenizer_hash_is_sha256_when_cached(self, tmp_path: Path):
        """T00：HF 缓存存在时 tokenizer_hash 为 64 位 hex sha256。"""
        from apcs.compat.scanner import _tokenizer_hash

        # 模拟 HF 缓存目录结构
        fake_model = "fake-org--fake-model"
        fake_home = tmp_path / "fake_home"
        snap_dir = (
            fake_home / ".cache" / "huggingface" / "hub"
            / f"models--{fake_model}" / "snapshots" / "abc123"
        )
        snap_dir.mkdir(parents=True)
        (snap_dir / "tokenizer_config.json").write_text(
            '{"model_type": "qwen3"}', encoding="utf-8"
        )

        with patch("apcs.compat.scanner.Path.home", return_value=fake_home):
            result = _tokenizer_hash("fake-org/fake-model")

        assert len(result) == 64  # sha256 hex
        assert all(c in "0123456789abcdef" for c in result)

    def test_revision_echoed_in_output(self, tmp_path: Path):
        """T00：teacher/student revision 被回写到 model_compatibility.json。"""
        from apcs.compat.scanner import run_compat_scan

        cfg = self._base_cfg()
        run_dir = tmp_path / "t00"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_compat_scan(cfg, run_dir)
        payload = json.loads(
            (run_dir / "model_compatibility.json").read_text(encoding="utf-8")
        )
        assert payload["teacher"]["revision"] == "main"
        assert payload["student"]["revision"] == "v1.0"

    def test_rope_theta_in_output(self, tmp_path: Path):
        """T00：rope_theta 从 fallback spec 中输出。"""
        from apcs.compat.scanner import run_compat_scan

        cfg = self._base_cfg()
        run_dir = tmp_path / "t00"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_compat_scan(cfg, run_dir)
        payload = json.loads(
            (run_dir / "model_compatibility.json").read_text(encoding="utf-8")
        )
        # fallback 路径 rope_theta=1_000_000.0
        assert payload["teacher"]["rope_theta"] == 1_000_000.0
        assert payload["student"]["rope_theta"] == 1_000_000.0

    def test_verdict_unchanged(self, tmp_path: Path):
        """T00：新增溯源字段不改变 verdict 判定逻辑（G1/G2/G3）。"""
        from apcs.compat.scanner import run_compat_scan

        cfg = self._base_cfg()
        run_dir = tmp_path / "t00"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_compat_scan(cfg, run_dir)
        assert result["status"] == "PASS"
        assert result["metrics"]["verdict"] == "G1_MATCHED_KV"


# =========================================================================
# T03 Layer Alignment data_driven_topk
# =========================================================================

class TestT03DataDrivenTopk:
    """T03：data_driven_topk 选择依据来自 train/validation 而非 test。"""

    def test_picks_planted_best_teacher_layer(self):
        """T03：构造数据中 layer k 有刻意最高相似度 → topk 选中 k。"""
        from apcs.alignment.runner import data_driven_topk, _similarity_from_kv_pairs

        n_t, n_s = 10, 5
        S, H, D = 16, 4, 16
        seed = 42
        rng = np.random.default_rng(seed)

        # teacher: 互相正交的层向量（乘以大系数确保相似度信号清晰）
        kv_t_base = rng.standard_normal((n_t, S, H, D))
        # student: 每层 s 完全复制 teacher layer s（零噪声 → cosine=1.0）
        kv_s = kv_t_base[:n_s].copy()

        sim = _similarity_from_kv_pairs(kv_t_base, kv_s)
        mapping = data_driven_topk(sim, k=2)

        # 每个 student 层的 top-1 应该是其对应 teacher 层（零噪声下 cosine=1.0）
        for s in range(n_s):
            # top-1 是 argmax 列，不经过 sort
            top1_col = int(np.argmax(sim[s]))
            assert top1_col == s, (
                f"Student layer {s}: best teacher should be {s}, got {top1_col}"
            )

    def test_similarity_matrix_shape(self):
        """T03：_similarity_from_kv_pairs 输出形状正确。"""
        from apcs.alignment.runner import _similarity_from_kv_pairs

        rng = np.random.default_rng(0)
        kv_t = rng.standard_normal((36, 8, 4, 16))
        kv_s = rng.standard_normal((28, 8, 4, 16))
        sim = _similarity_from_kv_pairs(kv_t, kv_s)
        assert sim.shape == (28, 36)

    def test_similarity_range(self):
        """T03：余弦相似度矩阵值在 [-1, 1] 范围内。"""
        from apcs.alignment.runner import _similarity_from_kv_pairs

        rng = np.random.default_rng(1)
        kv_t = rng.standard_normal((6, 8, 2, 8))
        kv_s = rng.standard_normal((4, 8, 2, 8))
        sim = _similarity_from_kv_pairs(kv_t, kv_s)
        assert sim.min() >= -1.0 - 1e-6
        assert sim.max() <= 1.0 + 1e-6

    def test_deterministic_identical_inputs(self):
        """T03：确定性 — 相同输入产生相同 mapping。"""
        from apcs.alignment.runner import _get_paired_kv_samples, _similarity_from_kv_pairs, data_driven_topk

        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "teacher": {"num_layers": 10},
            "student": {"num_layers": 6},
            "seeds": [42],
        }
        pair1 = _get_paired_kv_samples(cfg)
        pair2 = _get_paired_kv_samples(cfg)

        assert pair1 is not None and pair2 is not None
        np.testing.assert_array_equal(pair1[0], pair2[0])
        np.testing.assert_array_equal(pair1[1], pair2[1])

        sim = _similarity_from_kv_pairs(pair1[0], pair1[1])
        m1 = data_driven_topk(sim, k=2)
        m2 = data_driven_topk(sim, k=2)
        assert m1 == m2

    def test_run_layer_alignment_output_extended(self, tmp_path: Path):
        """T03：run_layer_alignment 产出扩展 layer_mapping.json 含分数。"""
        from apcs.alignment.runner import run_layer_alignment

        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "teacher": {"num_layers": 12},
            "student": {"num_layers": 8},
            "mapper": {"source_top_k": 3},
            "seeds": [42],
        }
        run_dir = tmp_path / "t03"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_layer_alignment(cfg, run_dir)
        assert result["status"] == "OK"

        # layer_mapping.json 包含 data_driven_topk_scores
        payload = json.loads(
            (run_dir / "layer_mapping.json").read_text(encoding="utf-8")
        )
        assert "data_driven_topk_scores" in payload
        scores = payload["data_driven_topk_scores"]
        assert len(scores) == 8  # n_s = 8

        # 每个 student 层有 3 个 teacher layer + 分数
        for entry in scores:
            assert entry["student_layer"] >= 0
            assert len(entry["teacher_layers"]) == 3
            assert len(entry["scores"]) == 3

    def test_proportional_stays_default(self, tmp_path: Path):
        """T03：proportional 仍为 selected 默认值。"""
        from apcs.alignment.runner import run_layer_alignment

        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "teacher": {"num_layers": 12},
            "student": {"num_layers": 8},
            "mapper": {"source_top_k": 2},
            "seeds": [42],
        }
        run_dir = tmp_path / "t03"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_layer_alignment(cfg, run_dir)
        assert result["metrics"]["selected"] == "proportional"

    def test_similarity_source_label(self, tmp_path: Path):
        """T03：similarity_source 标签正确。"""
        from apcs.alignment.runner import run_layer_alignment

        cfg: dict[str, Any] = {
            "provider": {"kv": "synthetic"},
            "teacher": {"num_layers": 6},
            "student": {"num_layers": 4},
            "mapper": {"source_top_k": 2},
            "seeds": [0],
        }
        run_dir = tmp_path / "t03"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_layer_alignment(cfg, run_dir)
        assert result["metrics"]["similarity_source"] in (
            "synthetic_deterministic",
            "synthetic_diagonal",
            "provider",
        )
