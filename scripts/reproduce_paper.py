"""Reproduce the cache-translation audit paper's core configurations.

For each config, rewrites model ids to the locally cached snapshots, runs the
inject-eval task, and prints the reproduced gold CHG next to the value reported
in the paper (recorded in the original run artifacts).

Usage:  python scripts/reproduce_paper.py [--only SUBSTR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

MODEL_MAP = {
    "qwen3-4b": "/workspace/models/Qwen3-4B",
    "qwen3-8b": "/workspace/models/Qwen3-8B",
    "qwen3-1.7b": "/workspace/models/Qwen3-1.7B",
    "qwen3-0.6b": "/workspace/models/Qwen3-0.6B",
}

# (config, label, reported metric)
CASES = [
    ("configs/v12_4b_1.7b_ridge.yaml", "Ridge per-head c30", "chg", -0.296),
    ("configs/v12_4b_1.7b_affine.yaml", "Affine per-head c30", "chg", -0.138),
    ("configs/v13_4b_1.7b_affine_c200.yaml", "Affine c200", "chg", -0.239),
    ("configs/v13_4b_1.7b_affinelayer_c200.yaml", "Affine per-layer c200", "chg", -0.275),
    ("configs/v13_4b_1.7b_taskaware_c200.yaml", "Task-aware c200", "chg", -0.216),
    ("configs/v14_4b_1.7b_rat_c30.yaml", "RAT c30", "chg", -0.268),
    ("configs/v14_4b_1.7b_rat_c200.yaml", "RAT c200", "chg", -0.221),
    ("configs/v14_4b_1.7b_rat_c30_noanchor.yaml", "RAT no-anchor c30", "chg", -0.223),
    ("configs/v15_4b_1.7b_mlp_c30.yaml", "MLP c30", "chg", -0.303),
    ("configs/v15_4b_1.7b_mlp_c200.yaml", "MLP c200", "chg", -0.236),
    ("configs/v15_4b_1.7b_jointmlp_c30.yaml", "Joint MLP c30", "chg", -0.265),
    ("configs/v15_8b_1.7b_ridge_c500.yaml", "8B->1.7B Ridge c500", "chg", -0.224),
    ("configs/v13_4b_0.6b_affine_c200.yaml", "4B->0.6B Affine c200", "chg", +0.010),
    ("configs/v15_8b_0.6b_affine_c30_tail.yaml", "8B->0.6B Affine c30", "chg", -0.023),
    ("configs/v15_4b_1.7b_affine_c10_tail.yaml", "Affine c10 (fixed eval)", "chg", -0.251),
    ("configs/v15_4b_1.7b_affine_c60_tail.yaml", "Affine c60 (fixed eval)", "chg", -0.238),
    ("configs/v15_4b_1.7b_affine_c100_tail.yaml", "Affine c100 (fixed eval)", "chg", -0.257),
    ("configs/v15_4b_1.7b_affine_c500_tail.yaml", "Affine c500 (fixed eval)", "chg", -0.244),
    ("configs/v15_4b_1.7b_probes_affine.yaml", "Oracle probes (affine)", "probe", None),
    ("configs/v15_4b_1.7b_probes_octant.yaml", "Octant probes", "probe", None),
    ("configs/v15_4b_1.7b_affine_c30_longctx.yaml", "Long-context needle", "chg", -0.285),
    ("configs/v15_4b_x_llama1b_rect_c30.yaml", "Cross-arch Llama-3.2-1B", "chg", +0.000),
    ("configs/v15_4b_x_gemma2-2b_rect_c30.yaml", "Cross-arch Gemma-2-2B", "chg", -0.009),
]


def localize(cfg: dict) -> dict:
    for role in ("teacher", "student"):
        mid = str(cfg[role].get("model_id", ""))
        low = mid.lower()
        for key, path in MODEL_MAP.items():
            if key in low and Path(path).exists():
                cfg[role]["model_id"] = path
                break
    return cfg


def run_one(config: str, label: str, kind: str, reported) -> dict:
    src = yaml.safe_load(Path(config).read_text())
    base_name = src["experiment"]["name"]
    repro_name = f"{base_name}-repro"
    src = localize(src)
    src["experiment"]["name"] = repro_name
    src["experiment"]["run_id"] = repro_name + "-${run.timestamp}"
    out_cfg = Path("configs/repro") / Path(config).name
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(src, open(out_cfg, "w"), sort_keys=False)

    env = dict(os.environ, HF_HOME="/workspace/.cache")
    cmd = [sys.executable, "-m", "apcs.cli", "inject-eval",
           "--config", str(out_cfg), "--new-run", "--force"]
    print(f"\n>>> REPRO {label} ({config})", flush=True)
    proc = subprocess.run(cmd, cwd="/workspace/KVCache", env=env,
                          capture_output=True, text=True, timeout=2400)
    log = Path(f"/tmp/opencode/repro_{base_name}.log")
    log.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"    FAILED rc={proc.returncode}; see {log}", flush=True)
        return {"label": label, "config": config, "status": "FAIL"}

    # newest run dir with this experiment name
    runs = sorted([p for p in Path("reports/runs").glob(repro_name + "*")
                   if not p.name.endswith(".current")])
    if not runs:
        return {"label": label, "config": config, "status": "NO_RUN"}
    m = json.loads((runs[-1] / "inject-eval" / "metrics.json").read_text())
    if kind == "chg":
        c = (m.get("chg_gold") or {}).get("ridge_kv_both", {})
        got = c.get("mean")
        print(f"    reproduced CHG={got:+.4f} [{c.get('ci_low'):+.4f},{c.get('ci_high'):+.4f}]"
              f"  reported={reported:+.3f}  delta={got-reported:+.4f}", flush=True)
        return {"label": label, "config": config, "status": "OK", "n": m.get("n_samples"),
                "repro_chg": got, "ci": [c.get("ci_low"), c.get("ci_high")],
                "reported": reported, "delta": got - reported}
    # probe kind: dump all probe methods
    probes = {k: v.get("gold_prob_mean") for k, v in (m.get("method_stats") or {}).items()
              if "win_" in k or "mix_a" in k or "self_kv" in k}
    print(f"    probes: {json.dumps({k: round(v, 4) for k, v in probes.items() if v})}", flush=True)
    return {"label": label, "config": config, "status": "OK", "n": m.get("n_samples"),
            "probes": probes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    results = []
    for config, label, kind, reported in CASES:
        if args.only and args.only not in config:
            continue
        try:
            results.append(run_one(config, label, kind, reported))
        except Exception as e:  # noqa: BLE001
            print(f"    EXCEPTION {label}: {e}", flush=True)
            results.append({"label": label, "config": config, "status": "EXC", "err": str(e)})
    Path("/tmp/opencode/repro_summary.json").write_text(json.dumps(results, indent=2))
    print("\n===== REPRODUCTION SUMMARY =====", flush=True)
    for r in results:
        if r.get("status") == "OK" and "repro_chg" in r:
            print(f"{r['label']:28s} repro={r['repro_chg']:+.4f} reported={r['reported']:+.3f} "
                  f"delta={r['delta']:+.4f}", flush=True)
        else:
            print(f"{r['label']:28s} {r.get('status')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
