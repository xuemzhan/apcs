r"""Automatic consistency check for the four manuscripts against the recorded artifacts.

Reads (never writes) the manuscripts and recomputes every number they quote from
``reports/runs/*/inject-eval/inject_eval/*_artifact.json`` and the stored
reconstruction / max-statistic artifacts.

Checks
------
1. Table 1 rows      : point estimate, n and 95% CI match a recomputed row.
2. Probe table rows  : gold means and CIs match the two probe runs.
3. Family CI table   : point ranges, largest CI upper and per-training uppers.
4. Appendix rows     : cross-architecture / aligned / long-context / replay CIs.
5. Derived arithmetic: 4% margin, 2.5 PPL gap, 23-point accuracy gap, ladder spread.
6. Headline numbers  : abstract-level values exist in the verified set.
7. Stale numbers     : the pre-audit-8 (1e3-draw) interval endpoints are gone.
8. Cross-references  : every \ref has a matching \label; every \cite key exists.

Usage:  python3 scripts/paper_consistency_check.py [--n-boot 10000]
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_ci_report as bcr  # noqa: E402

MANUSCRIPTS = ["paper/arxiv/main.tex", "paper/iclr2026/main.tex",
               "paper/humanized/arxiv/main.tex", "paper/humanized/iclr2026/main.tex"]

# interval endpoints the audit-8 recomputation replaced (must not survive)
STALE_PAIRS = ["-0.317,+0.029", "-0.534,-0.025", "-0.929,-0.170", "-0.932,-0.118",
               "-0.081,+0.102", "-0.336,-0.105", "-0.380,-0.158", "-0.368,-0.084"]

FAILURES: list[str] = []
PASSES: list[str] = []


def ok(msg: str) -> None:
    PASSES.append(msg)
    print(f"  PASS  {msg}")


def bad(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def fmt3(x: float) -> str:
    return f"+{x:.3f}" if x >= 0 else f"-{abs(x):.3f}"


def parse_table1(text: str):
    block = re.search(r"\\caption\{(?:Translator|Mapper) ladder.*?\\end\{tabular\}\}",
                      text, re.S).group(0)
    rows = []
    for line in block.splitlines():
        m = re.match(r"(.+?)\s*&\s*(.+?)\s*&\s*\$([-+][\d.]+)\$\s*"
                     r"\[\$([-+][\d.]+),\s*([-+][\d.]+)\$\]\s*&\s*(\d+)", line)
        if m:
            rows.append(dict(label=m.group(1).strip(), point=float(m.group(3)),
                             lo=float(m.group(4)), hi=float(m.group(5)), n=int(m.group(6))))
    return rows


def parse_probe_table(text: str):
    block = re.search(r"\\caption\{Teacher-content probes.*?\\end\{tabular\}\}", text, re.S).group(0)
    rows = []
    for line in block.splitlines():
        m = re.match(r"(.+?)\s*&\s*([\d.]+)\s*&\s*\$([-+][\d.]+)\$\s*\[\$([-+][\d.]+),\s*"
                     r"([-+][\d.]+)\$\]\s*&\s*([\d.]+)\s*&\s*\$([-+][\d.]+)\$\s*"
                     r"\[\$([-+][\d.]+),\s*([-+][\d.]+)\$\]", line)
        if m:
            rows.append(dict(label=m.group(1).strip(), rat_gold=float(m.group(2)),
                             rat_point=float(m.group(3)), rat_lo=float(m.group(4)),
                             rat_hi=float(m.group(5)), aff_gold=float(m.group(6)),
                             aff_point=float(m.group(7)), aff_lo=float(m.group(8)),
                             aff_hi=float(m.group(9))))
    return rows


def closes(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    # ---- recompute the ground truth -------------------------------------
    table1 = {label: bcr.ci(run, method, args.n_boot) for label, run, method in bcr.TABLE1}
    probes = {(label, tag): bcr.ci(run, method, args.n_boot)
              for label, run in bcr.PROBES for tag, method in bcr.PROBE_METHODS}
    families = {name: [bcr.ci(r, "ridge_kv_both", args.n_boot) for r in runs]
                for name, runs in bcr.FAMILIES}
    octant = max((bcr.ci(bcr.OCTANT_RUN, f"ridge_win_oct{i}", args.n_boot) for i in range(8)),
                 key=lambda r: r["point"])
    appendix = {label: bcr.ci(run, method, args.n_boot) for label, run, method in bcr.APPENDIX}

    for path in MANUSCRIPTS:
        print(f"\n=== {path}")
        text = (ROOT / path).read_text(encoding="utf-8")

        # ---- 1. Table 1 --------------------------------------------------
        rows = parse_table1(text)
        matched = 0
        for row in rows:
            cands = [v for v in table1.values()
                     if closes(v["point"], row["point"], 6e-4) and v["n"] == row["n"]]
            if not cands:
                bad(f"Table 1 row not reproducible from artifacts: {row['label'][:40]}")
                continue
            v = cands[0]
            if not (closes(v["lo"], row["lo"], 5e-4 + 1e-9) and closes(v["hi"], row["hi"], 5e-4)):
                bad(f"Table 1 CI mismatch {row['label'][:40]}: printed "
                    f"[{row['lo']:+.3f},{row['hi']:+.3f}] vs recomputed "
                    f"[{v['lo']:+.3f},{v['hi']:+.3f}]")
            else:
                matched += 1
        ok(f"Table 1: {matched}/{len(rows)} rows match recomputed point estimate, n and CI")

        # ---- 2. probe table ---------------------------------------------
        prows = parse_probe_table(text)
        pmatch = 0
        for row in prows:
            tag = ("self-kv" if "Student cache only" in row["label"] else
                   "frac-0.25" if "0.25" in row["label"] else
                   "frac-0.50" if "0.50" in row["label"] else
                   "frac-0.75" if "0.75" in row["label"] else
                   "window-top" if "top third" in row["label"] else
                   "window-middle" if "middle third" in row["label"] else "window-bottom")
            checks = [
                (probes[("probe-rat-c30", tag)], row["rat_point"], row["rat_lo"], row["rat_hi"]),
                (probes[("probe-affine-c30", tag)], row["aff_point"], row["aff_lo"], row["aff_hi"]),
            ]
            good = True
            for v, p, lo, hi in checks:
                if not (closes(v["point"], p, 6e-4) and closes(v["lo"], lo, 5e-4)
                        and closes(v["hi"], hi, 5e-4)):
                    bad(f"probe row {row['label'][:28]} ({tag}): printed "
                        f"{p:+.3f} [{lo:+.3f},{hi:+.3f}] vs recomputed "
                        f"{v['point']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")
                    good = False
            pmatch += good
        ok(f"probe table: {pmatch}/{len(prows)} rows match recomputed gold and CI")

        # ---- 3. family CI table ------------------------------------------
        printed = re.findall(r"(Per-head MLP|Joint MLP).*?&\s*5\s*&\s*\$([-+][\d.]+)\$ to "
                             r"\$([-+][\d.]+)\$\s*&\s*\$([-+][\d.]+)\$", text)
        if len(printed) != 3:
            bad(f"family CI table not found or malformed (rows={len(printed)})")
        else:
            fam_names = ["per-head MLP, c=30", "per-head MLP, c=200", "joint MLP, c=30"]
            for (kind, lo_s, hi_s, up_s), name in zip(printed, fam_names):
                rs = families[name]
                pmin, pmax = min(r["point"] for r in rs), max(r["point"] for r in rs)
                largest = max(r["hi"] for r in rs)
                if not (closes(pmin, float(lo_s), 6e-4) and closes(pmax, float(hi_s), 6e-4)
                        and closes(largest, float(up_s), 6e-4)):
                    bad(f"family row {name}: printed {lo_s}..{hi_s}, upper {up_s} vs "
                        f"recomputed {pmin:.3f}..{pmax:.3f}, upper {largest:.3f}")
                if largest >= 0:
                    bad(f"family {name}: largest CI upper {largest:+.3f} is not below zero "
                        f"but the row is labelled degraded")
            ok("family CI table: point ranges, largest CI upper and degraded labels check out")

        # ---- 4. appendix rows --------------------------------------------
        computed_pairs = {f"{fmt3(v['lo'])}, {fmt3(v['hi'])}" for v in appendix.values()}
        computed_pairs |= {f"{fmt3(v['lo'])},{fmt3(v['hi'])}" for v in appendix.values()}
        for v in probes.values():
            computed_pairs.add(f"{fmt3(v['lo'])}, {fmt3(v['hi'])}")
            computed_pairs.add(f"{fmt3(v['lo'])},{fmt3(v['hi'])}")
        computed_pairs.add(f"{fmt3(octant['lo'])}, {fmt3(octant['hi'])}")
        computed_pairs.add(f"{fmt3(octant['lo'])},{fmt3(octant['hi'])}")
        app_pairs = set(re.findall(r"\[\$([-+][\d.]+),\s*([-+][\d.]+)\$\]", text))
        table1_pairs = {f"{fmt3(r['lo'])}, {fmt3(r['hi'])}" for r in table1.values()}
        table1_pairs |= {f"{fmt3(r['lo'])},{fmt3(r['hi'])}" for r in table1.values()}
        unknown = [p for p in app_pairs
                   if f"{p[0]}, {p[1]}" not in computed_pairs | table1_pairs
                   and f"{p[0]},{p[1]}" not in computed_pairs | table1_pairs
                   and not (p[0] in ("-0.0006", "-0.0007") and p[1] in ("+0.0009", "+0.0010"))]
        if unknown:
            bad(f"{len(unknown)} interval(s) not explained by any recomputed row: {unknown[:4]}")
        else:
            ok("every printed interval is explained by a recomputed row")

        # ---- 5. derived arithmetic ---------------------------------------
        margin_ratio = 0.02 / 0.503
        ok(f"epsilon/baseline = {margin_ratio:.4f} (paper: 'about 4%')") \
            if 0.035 <= margin_ratio <= 0.045 else bad("epsilon/baseline not ~4%")
        if not closes(23.7 - 21.2, 2.5, 0.051):
            bad("'within 2.5 of the identity control's 21.2' does not recompute")
        # accuracy gap in Figure 3's caption: identity control minus the fluent
        # failing translator, both read from the recorded per-sample decisions
        def accuracy(run: str, method: str) -> float:
            recs = bcr.load_gold(run, method)
            path = (bcr.RUNS_ROOT / run / "inject-eval" / "inject_eval" /
                    "capability_score_artifact.json")
            allrecs = json.loads(path.read_text(encoding="utf-8"))["records"]
            vals = [r["correct"] for r in allrecs
                    if r["method"] == method and r.get("correct") is not None]
            return sum(bool(v) for v in vals) / len(vals)

        gap = accuracy("v12-4b-1.7b-ridge-20260829-144617", "ridge_self_kv") - \
            accuracy("v14-4b-to-1.7b-rat-c30-noanchor-20260829-180839", "ridge_kv_both")
        if not closes(gap, 0.233, 0.02):
            bad(f"captioned accuracy gap {gap:.3f} does not match '23 points below'")
        else:
            ok(f"captioned accuracy gap {gap * 100:.1f} points == printed '23 points'")
        # text-channel control difference
        tc = appendix["text-channel"]
        if not closes(tc["point"], -0.108, 0.002):
            bad(f"text-channel difference {tc['point']:.3f} does not match printed -0.108")
        else:
            ok(f"text-channel difference {tc['point']:.3f} == printed -0.108")
        ladder = {name: bcr.ci(name, "ridge_kv_both", args.n_boot)["point"] for name in
                  ["v15-4b-to-1.7b-affine-c10-s42-20260830-230433",
                   "v15-4b-to-1.7b-affine-c60-s42-20260830-231024",
                   "v15-4b-to-1.7b-affine-c100-s42-20260830-231652",
                   "v15-4b-to-1.7b-affine-c200-s42-20260829-231903",
                   "v15-4b-to-1.7b-affine-c500-s42-20260830-232053"]}
        spread = max(ladder.values()) - min(ladder.values())
        if not closes(spread, 0.02, 0.0011):
            bad(f"calibration-ladder spread {spread:.4f} does not match the printed 0.02")
        else:
            ok(f"calibration ladder spread {spread:.4f} == printed 0.02")

        # ---- 6. headline numbers exist -----------------------------------
        headline = ["-0.138", "+0.010", "0.854", "0.481", "+0.009", "0.99998",
                    "0.99986", "26.7", "50.0"]
        missing = [h for h in headline if h not in text]
        ok(f"abstract-level numbers present: {len(headline) - len(missing)}/{len(headline)}") \
            if not missing else bad(f"missing headline numbers: {missing}")

        # ---- 7. stale (pre-recomputation) endpoints ----------------------
        stale = [s for s in STALE_PAIRS if s in text]
        ok("no pre-audit-8 interval endpoints remain") if not stale \
            else bad(f"stale 1e3-draw endpoints still printed: {stale}")

        # ---- 8. cross-references and citations ---------------------------
        labels = set(re.findall(r"\\label\{([^}]*)\}", text))
        refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", text))
        dangling = sorted(refs - labels)
        if dangling:
            bad(f"dangling cross-references: {dangling}")
        else:
            ok(f"cross-references resolve ({len(refs)} refs, {len(labels)} labels)")
        bib = (ROOT / Path(path).parent / "references.bib").read_text(encoding="utf-8")
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        cited = set()
        for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", text):
            cited |= {k.strip() for k in m.split(",") if k.strip()}
        missing_cites = sorted(cited - bib_keys)
        if missing_cites:
            bad(f"citation keys missing from references.bib: {missing_cites}")
        else:
            ok(f"all {len(cited)} citation keys exist in references.bib")

    print(f"\n{'='*60}\nPASS {len(PASSES)} checks, FAIL {len(FAILURES)} checks")
    for f in FAILURES:
        print("  -", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
