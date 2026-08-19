"""真实数据集准备与不可变 split manifest 生成（无需 GPU）。"""
from __future__ import annotations

from typing import Any

from ..io.runs import write_json
from .hf_dataset import load, split_manifest


def _dataset_names(cfg: dict[str, Any]) -> list[str]:
    datasets = cfg.get("datasets", {})
    names = list(datasets.get("fidelity", []) or [])
    primary = (datasets.get("teacher_advantage", {}) or {}).get("primary")
    if primary:
        names.append(primary)
    return list(dict.fromkeys(str(x) for x in names))


def run_prepare_data(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """下载/规范化数据，并生成互斥 train/validation/test 清单。"""
    opts = cfg.get("dataset_prepare", {})
    n = int(opts.get("samples_per_split", 128))
    seed = int(opts.get("seed", cfg.get("seeds", [0])[0]))
    names = _dataset_names(cfg)
    if not names:
        raise ValueError("cfg.datasets 未配置 fidelity 或 teacher_advantage.primary")

    manifests: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    for name in names:
        manifests[name] = {}
        snapshots[name] = {}
        id_sets: dict[str, set[str]] = {}
        for split in ("train", "validation", "test"):
            rows = load(name, n=n, split=split, seed=seed)
            manifests[name][split] = split_manifest(name, rows, split, seed)
            id_sets[split] = {row.sample_id for row in rows}
            snapshots[name][split] = [
                {
                    "sample_id": row.sample_id,
                    "context": row.context,
                    "query": row.query,
                    "answer": row.answer,
                    "split": row.split,
                }
                for row in rows
            ]
        overlaps = {
            "train_validation": sorted(id_sets["train"] & id_sets["validation"]),
            "train_test": sorted(id_sets["train"] & id_sets["test"]),
            "validation_test": sorted(id_sets["validation"] & id_sets["test"]),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"dataset {name!r} split 泄漏：{overlaps}")
        manifests[name]["overlap_check"] = {"passed": True, **overlaps}

    payload = {
        "schema_version": 1,
        "seed": seed,
        "samples_per_split_requested": n,
        "datasets": manifests,
    }
    write_json(run_dir / "dataset_manifest.json", payload)
    write_json(run_dir / "dataset_samples.json", snapshots)
    metrics = {
        "task": "prepare-data",
        "datasets": names,
        "seed": seed,
        "split_overlap_check": "PASS",
        "manifest": "dataset_manifest.json",
        "snapshot": "dataset_samples.json",
    }
    summary = (
        "# Dataset preparation\n\n"
        f"- datasets: {names}\n"
        f"- seed: {seed}\n"
        f"- requested samples/split: {n}\n"
        "- train/validation/test overlap: PASS\n"
    )
    return {"status": "OK", "metrics": metrics, "summary": summary}


__all__ = ["run_prepare_data"]
