"""Add newly recorded inject-eval runs to the paper figure datasets.

Reads ``reports/runs/*/inject-eval/metrics.json`` for the given run prefixes and
writes the entry format consumed by ``gen_fig2/3/4`` into each paper's
``figures/data/all_results.json`` (preserving existing entries).
"""
from __future__ import annotations

import json
from pathlib import Path

NEW_RUN_PREFIXES = [
    "v15-4b-to-1.7b-affine-c30-probes-s42-20260829-234406",
    "v15-4b-to-1.7b-jointmlp-c30-s42",
    "v14-4b-to-1.7b-rat-c30-noanchor",
    "v14-4b-to-1.7b-rat-c30-probes",
    "v14-4b-to-0.6b-rat-c200",
    "v15-4b-to-1.7b-affine-c30-longctx-s42",
    "v15-4b-to-1.7b-affine-c30-probes-octant-s42",
    "v15-4b-to-1.7b-affine-c30-replay-s42",
    "v15-4b-to-1.7b-affine-c30-s42-20260829-231341",
    "v13-4b-to-0.6b-affine-c200-20260829-154145",
    "v15-8b-to-1.7b-ridge-c500",
    "v15-8b-to-0.6b-affine-c30",
    "v15-4b-to-1.7b-mlp-c",
    "v15-4b-to-1.7b-jointmlp-c30-seed",
    "v15-4b-x-llama1b-rect-c30-s42",
    "v15-4b-x-gemma2-2b-rect-c30-s42",
    "v15-4b-x-llama32-3b-rect-c30-s42",
    "v15-4b-x-qwen25-1.5b-rect-c30-s42",
    "v15-4b-x-gemma3-1b-rect-c30-s42",
    "v15-4b-x-llama32-3b-rect-align-c30-s42",
    "v15-4b-x-gemma3-1b-rect-align-c30-s42",
    "v15-4b-x-qwen25-1.5b-rect-align-c30-s42",
]

TARGETS = [
    Path("paper/cache_audit/figures/data/all_results.json"),
    Path("paper/cache_audit_iclr2026/figures/data/all_results.json"),
]


def make_entry(m: dict) -> dict:
    cg = m.get("chg_gold") or {}
    ca = m.get("chg_accuracy") or {}
    return {
        "chg": (ca.get("ridge_kv_both") or {}).get("mean"),
        "chg_gold": cg.get("ridge_kv_both"),
        "chg_gold_native": cg.get("ridge_native"),
        "chg_gold_selfkv": cg.get("ridge_self_kv"),
        "chg_gold_all": cg,
        "method_stats": m.get("method_stats"),
        "capability_gate": m.get("capability_gate"),
        "rope": None,
    }


def main() -> int:
    runs_root = Path("reports/runs")
    added: dict[str, dict] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or d.name.endswith(".current"):
            continue
        if not any(d.name.startswith(p) for p in NEW_RUN_PREFIXES):
            continue
        mpath = d / "inject-eval" / "metrics.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        added[d.name] = make_entry(m)
        print("add", d.name, "n=", m.get("n_samples"))

    for tgt in TARGETS:
        data = json.loads(tgt.read_text(encoding="utf-8")) if tgt.exists() else {}
        data.update(added)
        tgt.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print("wrote", tgt, "entries", len(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
