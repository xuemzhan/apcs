"""Aggregate claim statistics over the recorded audit-protocol runs.

Every number this script prints is used verbatim in the paper's aggregate
statements (Section 5.2 and Section 3.3). It reads only
``reports/runs/*/inject-eval/metrics.json``.

Definitions used here:
  * recorded evaluation run   -- a run directory that contains a metrics file.
  * audit-protocol run        -- a run whose directory name starts with v12-,
                                 v13-, v14-, v15-, or the tail-protocol sanity
                                 prefix; v1.1-era runs used a different metric
                                 and an in-sample calibration split and are
                                 excluded from every number in the paper.
  * translated configuration  -- a (run, method) row that is not the identity
                                 control, not the native-content control, not
                                 the text channel, and not a probe window
                                 (method names containing ``mix_`` or ``win_``).
  * replacement margin        -- epsilon = 0.02 gold probability; a row clears
                                 it when its 95% CI lower bound is above
                                 -epsilon.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS_ROOT = Path("reports/runs")
EPS = 0.02
EXCLUDED = {"ridge_self_kv", "ridge_native", "text", "student", "teacher"}
AUDIT_PREFIXES = ("v12-", "v13-", "v14-", "v15-", "qwen3-affine-c30-tail-sanity")


def is_audit_protocol(run_id: str) -> bool:
    return run_id.startswith(AUDIT_PREFIXES)


def is_probe(method: str) -> bool:
    return "mix_" in method or "win_" in method


def iter_rows():
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        if not run_dir.is_dir() or run_dir.name.endswith(".current"):
            continue
        path = run_dir / "inject-eval" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        yield run_dir.name, metrics


def main() -> int:
    runs = list(iter_rows())
    print(f"recorded evaluation runs (metrics file present): {len(runs)}")

    rows = []
    for run_id, metrics in runs:
        if not is_audit_protocol(run_id):
            continue
        chg = metrics.get("chg_gold") or {}
        for method, stats in chg.items():
            if method in EXCLUDED or not isinstance(stats, dict):
                continue
            if stats.get("mean") is None:
                continue
            rows.append(
                {
                    "run": run_id,
                    "method": method,
                    "mean": stats["mean"],
                    "lo": stats.get("ci_low"),
                    "hi": stats.get("ci_high"),
                    "n": metrics.get("n_samples"),
                    "probe": is_probe(method),
                }
            )

    translations = [r for r in rows if not r["probe"]]
    probes = [r for r in rows if r["probe"]]
    print(f"audit-protocol configuration rows: {len(rows)}"
          f" (translations {len(translations)}, probes {len(probes)})")

    best = max(translations, key=lambda r: r["mean"])
    print(f"max translation point estimate: {best['mean']:+.4f}"
          f"  [{best['lo']:+.4f}, {best['hi']:+.4f}]  n={best['n']}"
          f"  {best['run']} / {best['method']}")

    clears = [r for r in translations if r["lo"] is not None and r["lo"] > -EPS]
    print(f"translations clearing the epsilon={EPS} margin (ci_low > -eps): {len(clears)}")
    seen = set()
    for r in clears:
        key = (r["method"], round(r["mean"], 4), round(r["lo"], 4), round(r["hi"], 4))
        if key in seen:
            continue
        seen.add(key)
        print(f"    {r['run']} / {r['method']}"
              f"  {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] n={r['n']}")
    print(f"distinct translated configurations clearing the margin: {len(seen)}")

    if probes:
        bp = max(probes, key=lambda r: r["mean"])
        print(f"max probe point estimate: {bp['mean']:+.4f}"
              f"  [{bp['lo']:+.4f}, {bp['hi']:+.4f}]  {bp['run']} / {bp['method']}")
        print("probe rows whose interval excludes zero on the positive side: "
              f"{sum(1 for r in probes if r['lo'] is not None and r['lo'] > 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
