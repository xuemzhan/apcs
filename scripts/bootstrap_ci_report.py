"""Recompute every interval the paper reports, from the recorded per-sample scores.

Source of truth
---------------
``reports/runs/<run>/inject-eval/inject_eval/capability_score_artifact.json``
stores one record per (sample, method) with the gold-letter probability. A row's
interval is the paired percentile bootstrap of ``gold(method) - gold(student)``
over the samples present in both, computed with ``apcs.metrics.bootstrap_ci``
(the same function the injection pipeline uses, default seed 0).

The audit-8 review asked for the ordinary intervals to use 10^4 resamples rather
than 10^3, because the recorded 10^3 intervals carry Monte-Carlo noise of the
same order as the reporting margin epsilon=0.02 for the rows that sit near it.

Usage
-----
    python3 scripts/bootstrap_ci_report.py                    # print the table
    python3 scripts/bootstrap_ci_report.py --n-boot 100000    # stability check
    python3 scripts/bootstrap_ci_report.py --json reports/path.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apcs.metrics import bootstrap_ci  # noqa: E402

RUNS_ROOT = Path(__file__).resolve().parents[1] / "reports" / "runs"

# --- rows exactly as they appear in the manuscript -------------------------

TABLE1 = [
    ("identity-control", "v12-4b-1.7b-ridge-20260829-144617", "ridge_self_kv"),
    ("native-cache", "v12-4b-1.7b-ridge-20260829-144617", "ridge_native"),
    ("ridge-per-head-c30", "v12-4b-1.7b-ridge-20260829-140435", "ridge_kv_both"),
    ("affine-per-head-c30", "v12-4b-1.7b-affine-20260829-140858", "ridge_kv_both"),
    ("affine-per-head-c200", "v13-4b-to-1.7b-affine-c200-20260829-152704", "ridge_kv_both"),
    ("affine-per-layer-c200", "v13-4b-to-1.7b-affine_layer-c200-20260829-153202", "ridge_kv_both"),
    ("task-aware-c200", "v13-4b-to-1.7b-taskaware-c200-20260829-155727", "ridge_kv_both"),
    ("rat-c30", "v14-4b-to-1.7b-rat-c30-20260829-174525", "ridge_kv_both"),
    ("rat-c200", "v14-4b-to-1.7b-rat-c200-20260829-174926", "ridge_kv_both"),
    ("rat-noanchor-c30", "v14-4b-to-1.7b-rat-c30-noanchor-20260829-180839", "ridge_kv_both"),
    ("heo-avg-k1", "v15-4b-to-1.7b-heo-topk1-ridge-c200-20260912-190517", "ridge_kv_both"),
    ("heo-avg-k3", "v15-4b-to-1.7b-heo-topk3-ridge-c200-20260912-191428", "ridge_kv_both"),
    ("heo-avg-k5", "v15-4b-to-1.7b-heo-topk5-ridge-c200-20260912-192407", "ridge_kv_both"),
    ("heo-concat-k1", "v15-4b-to-1.7b-heo-concat-fwe-k1-c100-20260913-123506", "ridge_kv_both"),
    ("heo-concat-k3", "v15-4b-to-1.7b-heo-concat-fwe-k3-c100-20260913-124740", "ridge_kv_both"),
    ("heo-concat-k5", "v15-4b-to-1.7b-heo-concat-fwe-k5-c100-20260913-130754", "ridge_kv_both"),
    ("ridge-c500-8b-to-1.7b", "v15-8b-to-1.7b-ridge-c500-20260830-131432", "ridge_kv_both"),
    ("affine-c200-4b-to-0.6b", "v13-4b-to-0.6b-affine-c200-20260829-154145", "ridge_kv_both"),
    ("affine-c30-8b-to-0.6b", "v15-8b-to-0.6b-affine-c30-s42-20260830-232500", "ridge_kv_both"),
    ("rat-c200-4b-to-0.6b", "v14-4b-to-0.6b-rat-c200-20260829-175436", "ridge_kv_both"),
    ("affine-c30-1k-retrieval", "v15-4b-to-1.7b-affine-c30-longctx-s42-20260912-072916", "ridge_kv_both"),
    ("affine-c10-4k-retrieval", "v15-4b-to-1.7b-affine-c30-longctx4k-20260913-144905", "ridge_kv_both"),
    ("affine-c10-8k-retrieval", "v15-4b-to-1.7b-affine-c10-longctx8k-20260913-150122", "ridge_kv_both"),
]

PROBES = [
    ("probe-rat-c30", "v14-4b-to-1.7b-rat-c30-probes-20260829-175933"),
    ("probe-affine-c30", "v15-4b-to-1.7b-affine-c30-probes-s42-20260829-194413"),
]

PROBE_METHODS = [("self-kv", "ridge_self_kv"), ("frac-0.25", "ridge_mix_a25"),
                 ("frac-0.50", "ridge_mix_a50"), ("frac-0.75", "ridge_mix_a75"),
                 ("window-top", "ridge_win_high"), ("window-middle", "ridge_win_mid"),
                 ("window-bottom", "ridge_win_low")]

APPENDIX = [
    ("llama1b-rect", "v15-4b-x-llama1b-rect-c30-s42-20260912-092258", "ridge_kv_both"),
    ("gemma2-2b-rect", "v15-4b-x-gemma2-2b-rect-c30-s42-20260912-094149", "ridge_kv_both"),
    ("llama32-3b-rect", "v15-4b-x-llama32-3b-rect-c30-s42-20260912-095811", "ridge_kv_both"),
    ("gemma3-1b-rect", "v15-4b-x-gemma3-1b-rect-c30-s42-20260912-100213", "ridge_kv_both"),
    ("qwen25-1.5b-rect", "v15-4b-x-qwen25-1.5b-rect-c30-s42-20260912-100734", "ridge_kv_both"),
    ("llama32-3b-align", "v15-4b-x-llama32-3b-rect-align-c30-s42-20260912-165746", "ridge_kv_both"),
    ("gemma3-1b-align", "v15-4b-x-gemma3-1b-rect-align-c30-s42-20260912-170211", "ridge_kv_both"),
    ("llama1b-align", "v15-4b-x-llama1b-rect-align-c30-s42-20260912-190021", "ridge_kv_both"),
    ("gemma2-2b-align", "v15-4b-x-gemma2-2b-rect-align-c30-s42-20260912-190144", "ridge_kv_both"),
    ("replay-1.7b", "v15-4b-to-1.7b-affine-c30-replay-s42-20260912-110006", "ridge_replay"),
    ("replay-off-1.7b", "v15-4b-to-1.7b-affine-c30-replay-s42-20260912-110006", "ridge_kv_both"),
    ("text-channel", "v12-4b-1.7b-taskmix-summary-tail100-20260912-193441", "summary"),
]

# gradient-trained families: every recorded training of the family
FAMILIES = [
    ("per-head MLP, c=30", [
        "v15-4b-to-1.7b-mlp-c30-s42-20260829-195607",
        "v15-4b-to-1.7b-mlp-c30-s42-repro-20260912-115929",
        "v15-4b-to-1.7b-mlp-c30-seed0-20260912-145147",
        "v15-4b-to-1.7b-mlp-c30-seed1-20260912-150059",
        "v15-4b-to-1.7b-mlp-c30-seed2-20260912-151111"]),
    ("per-head MLP, c=200", [
        "v15-4b-to-1.7b-mlp-c200-s42-20260829-200557",
        "v15-4b-to-1.7b-mlp-c200-s42-repro-20260912-120751",
        "v15-4b-to-1.7b-mlp-c200-seed0-20260912-152031",
        "v15-4b-to-1.7b-mlp-c200-seed1-20260912-153003",
        "v15-4b-to-1.7b-mlp-c200-seed2-20260912-154233"]),
    ("joint MLP, c=30", [
        "v15-4b-to-1.7b-jointmlp-c30-s42-20260912-072431",
        "v15-4b-to-1.7b-jointmlp-c30-s42-repro-20260912-121754",
        "v15-4b-to-1.7b-jointmlp-c30-seed0-20260912-164750",
        "v15-4b-to-1.7b-jointmlp-c30-seed1-20260912-164949",
        "v15-4b-to-1.7b-jointmlp-c30-seed2-20260912-165141"]),
]

OCTANT_RUN = "v15-4b-to-1.7b-affine-c30-probes-octant-s42-20260912-070436"


def load_gold(run: str, method: str) -> dict[str, float]:
    path = RUNS_ROOT / run / "inject-eval" / "inject_eval" / "capability_score_artifact.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return {r["sample_id"]: r["gold_prob"] for r in records
            if r["method"] == method and r.get("gold_prob") is not None}


def paired_diffs(run: str, method: str, baseline: str = "student") -> list[float]:
    a = load_gold(run, method)
    b = load_gold(run, baseline)
    return [a[s] - b[s] for s in sorted(a.keys() & b.keys())]


def ci(run: str, method: str, n_boot: int, baseline: str = "student") -> dict:
    diffs = paired_diffs(run, method, baseline)
    point, lo, hi = bootstrap_ci(diffs, n_boot=n_boot)
    return dict(run=run, method=method, n=len(diffs), point=point, lo=lo, hi=hi,
                diffs=diffs)


def fmt(x: float) -> str:
    return f"+{x:.3f}" if x >= 0 else f"-{abs(x):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out: dict[str, dict] = {}
    print(f"paired percentile bootstrap, {args.n_boot} resamples "
          f"(apcs.metrics.bootstrap_ci, seed 0)\n")
    print(f"{'row':30s} {'n':>4s}  {'point':>8s}  95% CI")
    for group, rows in (("table-1", TABLE1), ("appendix", APPENDIX)):
        for label, run, method in rows:
            r = ci(run, method, args.n_boot)
            out[label] = {k: v for k, v in r.items() if k != "diffs"}
            print(f"{label:30s} {r['n']:4d}  {fmt(r['point']):>8s}  "
                  f"[{fmt(r['lo'])}, {fmt(r['hi'])}]")

    for label, run in PROBES:
        for tag, method in PROBE_METHODS:
            r = ci(run, method, args.n_boot)
            out[f"{label}:{tag}"] = {k: v for k, v in r.items() if k != "diffs"}
            print(f"{label}:{tag:30s} {r['n']:4d}  {fmt(r['point']):>8s}  "
                  f"[{fmt(r['lo'])}, {fmt(r['hi'])}]")

    oct_best = None
    for i in range(8):
        r = ci(OCTANT_RUN, f"ridge_win_oct{i}", args.n_boot)
        if oct_best is None or r["point"] > oct_best["point"]:
            oct_best = r
    out["octant-best"] = {k: v for k, v in oct_best.items() if k != "diffs"}
    print(f"{'octant-best':30s} {oct_best['n']:4d}  {fmt(oct_best['point']):>8s}  "
          f"[{fmt(oct_best['lo'])}, {fmt(oct_best['hi'])}]")

    print("\ngradient-trained families (every recorded training):")
    print(f"{'family':22s} {'runs':>4s}  {'point range':>18s}  "
          f"{'largest CI upper':>17s}  per-run upper bounds")
    for family, runs in FAMILIES:
        rs = [ci(r, "ridge_kv_both", args.n_boot) for r in runs]
        largest = max(rs, key=lambda r: r["hi"])
        out[f"family:{family}"] = {
            "point_range": [min(r["point"] for r in rs), max(r["point"] for r in rs)],
            "largest_ci_upper": largest["hi"],
            "upper_bounds": [r["hi"] for r in rs],
            "runs": [r["run"] for r in rs],
        }
        print(f"{family:22s} {len(rs):4d}  "
              f"{fmt(min(r['point'] for r in rs)):>8s}..{fmt(max(r['point'] for r in rs)):>8s}  "
              f"{fmt(largest['hi']):>17s}  "
              f"{[fmt(r['hi']) for r in rs]}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
