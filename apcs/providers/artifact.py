"""经审计的真实实验 artifact Provider。

这些 Provider 不伪造模型执行；它们只接收由真实 HandoffPipeline
导出的逐样本评分或逐次端到端计时，并对证据条件做严格校验。
缺少 zero-prefill、held-out split 或端到端来源时显式拒绝打开。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _path_from_cfg(cfg: dict[str, Any], key: str) -> Path:
    value = cfg.get("provider", {}).get(key, cfg.get(key))
    if not value:
        raise RuntimeError(f"artifact provider 需要配置 {key}")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"artifact 不存在: {path}")
    return path


@dataclass
class ArtifactScoreProvider:
    """读取逐 ``(method, sample_id, seed)`` 的真实任务评分。"""

    kind: str = "artifact"
    _path: Path | None = None
    _payload: dict[str, Any] = field(default_factory=dict)
    _records: dict[tuple[str, str, int], dict[str, Any]] = field(default_factory=dict)

    def open(self, cfg: dict[str, Any]) -> None:
        self._path = _path_from_cfg(cfg, "score_artifact_path")
        self._payload = json.loads(self._path.read_text(encoding="utf-8"))
        required_truth = {
            "evidence_grade": "measured_task",
            "zero_prefill_verified": True,
            "task_scoring_verified": True,
        }
        for key, expected in required_truth.items():
            if self._payload.get(key) != expected:
                raise RuntimeError(f"score artifact 需要 {key}={expected!r}")
        if self._payload.get("split") not in {"validation", "heldout", "test"}:
            raise RuntimeError("score artifact split 必须是 validation/heldout/test")
        records = self._payload.get("records")
        if not isinstance(records, list) or not records:
            raise RuntimeError("score artifact.records 必须是非空列表")
        self._records = {}
        for row in records:
            key = (str(row["method"]), str(row["sample_id"]), int(row["seed"]))
            if key in self._records:
                raise RuntimeError(f"score artifact 存在重复记录: {key}")
            score = float(row["score"])
            if not math.isfinite(score):
                raise RuntimeError(f"score 必须是有限数: {key}")
            if "decision" not in row:
                raise RuntimeError(f"score artifact 缺 decision: {key}")
            self._records[key] = {**row, "score": score, "decision": int(row["decision"])}

    def sample_ids(self) -> list[str]:
        return sorted({key[1] for key in self._records})

    def score(self, method: str, sample_id: str, seed: int) -> float:
        key = (str(method), str(sample_id), int(seed))
        if key not in self._records:
            raise KeyError(f"score artifact 缺少记录: {key}")
        return float(self._records[key]["score"])

    def decision(self, method: str, sample_id: str, seed: int) -> int:
        key = (str(method), str(sample_id), int(seed))
        if key not in self._records:
            raise KeyError(f"score artifact 缺少记录: {key}")
        return int(self._records[key]["decision"])

    def close(self) -> None:
        self._records = {}

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "audited_score_artifact",
            "path": str(self._path) if self._path else None,
            "evidence_grade": self._payload.get("evidence_grade"),
            "dataset": self._payload.get("dataset"),
            "split": self._payload.get("split"),
            "zero_prefill_verified": self._payload.get("zero_prefill_verified"),
            "task_scoring_verified": self._payload.get("task_scoring_verified"),
            "n_records": len(self._records),
        }


@dataclass
class ArtifactTimingProvider:
    """读取逐 ``(context, seed)`` 的 HandoffPipeline 端到端计时。"""

    kind: str = "artifact"
    _path: Path | None = None
    _payload: dict[str, Any] = field(default_factory=dict)
    _records: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)

    def open(self, cfg: dict[str, Any]) -> None:
        self._path = _path_from_cfg(cfg, "timing_artifact_path")
        self._payload = json.loads(self._path.read_text(encoding="utf-8"))
        if self._payload.get("timing_evidence") != "end_to_end_handoff":
            raise RuntimeError("timing artifact 必须标注 timing_evidence=end_to_end_handoff")
        if self._payload.get("cuda_synchronized") is not True:
            raise RuntimeError("timing artifact 必须证明 cuda_synchronized=true")
        if int(self._payload.get("warmup", 0)) < 1 or int(self._payload.get("repeats", 0)) < 10:
            raise RuntimeError("timing artifact 要求 warmup>=1 且 repeats>=10")
        required = {"teacher_prefill", "student_prefill", "map", "load", "query"}
        records = self._payload.get("records")
        if not isinstance(records, list) or not records:
            raise RuntimeError("timing artifact.records 必须是非空列表")
        self._records = {}
        for row in records:
            key = (int(row["context"]), int(row["seed"]))
            if key in self._records:
                raise RuntimeError(f"timing artifact 存在重复记录: {key}")
            if not required.issubset(row):
                raise RuntimeError(f"timing artifact 缺少分项 {sorted(required - set(row))}: {key}")
            values = {name: float(row[name]) for name in required}
            if any((not math.isfinite(v) or v < 0) for v in values.values()):
                raise RuntimeError(f"timing 必须是非负有限数: {key}")
            self._records[key] = values

    def measure(self, ctx: int, seed: int) -> dict[str, float]:
        key = (int(ctx), int(seed))
        if key not in self._records:
            raise KeyError(f"timing artifact 缺少记录: {key}")
        return dict(self._records[key])

    def measure_vram(self) -> int:
        return int(self._payload.get("vram_mb", 0))

    def close(self) -> None:
        self._records = {}

    def describe(self) -> dict[str, Any]:
        return {
            "implementation": "audited_timing_artifact",
            "path": str(self._path) if self._path else None,
            "evidence_grade": "measured_system",
            "timing_evidence": self._payload.get("timing_evidence"),
            "hardware": self._payload.get("hardware"),
            "warmup": self._payload.get("warmup"),
            "repeats": self._payload.get("repeats"),
            "cuda_synchronized": self._payload.get("cuda_synchronized"),
            "n_records": len(self._records),
        }


__all__ = ["ArtifactScoreProvider", "ArtifactTimingProvider"]
