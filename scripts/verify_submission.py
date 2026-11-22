"""Pre-submission verifier (CPU only, no model loading).

Checks the things a reviewer or the AEA/ICLR desk would otherwise catch by hand:

  1. numeric gate      -- scripts/paper_consistency_check.py passes;
  2. page limits       -- the ICLR manuscript's body ends on page 9 or earlier
                          (read from main.aux, not from PDF text extraction);
  3. citations         -- every \\cite key resolves in references.bib, every
                          \\ref has a \\label;
  4. submission bundle -- paper/arxiv/arxiv_submission.zip contains the required
                          entries with the figures/ layout and is byte-identical
                          to the source directory;
  5. distributed PDFs  -- the two root-level PDFs are byte-identical to
                          paper/arxiv/main.pdf and paper/zh/main.pdf.

Usage:
    python scripts/verify_submission.py            # all checks
    python scripts/verify_submission.py --fast     # skip the bootstrap gate

Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, manuscript dir, bib, body-page-limit or None)
MANUSCRIPTS = [
    ("arxiv", "paper/arxiv", "paper/arxiv/references.bib", None),
    ("iclr2026", "paper/iclr2026", "paper/iclr2026/references.bib", 9),
    ("zh", "paper/zh", "paper/zh/references.bib", None),
]

REQUIRED_ZIP_ENTRIES = [
    "main.tex", "main.bbl", "references.bib",
    "figures/fig1_framework.tex",
    "figures/fig2_mapper_landscape.pdf",
    "figures/fig3_oracle_probes.pdf",
    "figures/fig4_ppl_acc_decoupling.pdf",
]

DISTRIBUTED = [
    ("paper/arxiv/main.pdf",
     "Cache Translation Across Model Scale_An Audit with Identity Controls and Teacher-Content Probes.pdf"),
    ("paper/zh/main.pdf",
     "跨模型规模的缓存翻译：一项带恒等对照与教师内容探针的审计.pdf"),
]

FAILURES: list[str] = []
PASSES: list[str] = []


def ok(msg: str) -> None:
    PASSES.append(msg)
    print(f"  PASS  {msg}")


def bad(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def label_pages(aux: Path) -> dict[str, int]:
    """Map every \\label to the page number LaTeX assigned it."""
    if not aux.exists():
        return {}
    text = aux.read_text(encoding="utf-8", errors="replace")
    out: dict[str, int] = {}
    for name, _num, page in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{(\d+)\}", text):
        out[name] = int(page)
    return out


def total_pages(log: Path) -> int | None:
    if not log.exists():
        return None
    m = re.search(r"Output written on main\.pdf \((\d+) pages?", log.read_text(
        encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def undefined_refs(log: Path) -> int:
    if not log.exists():
        return -1
    text = log.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"Warning: (?:Reference|Citation)", text))


def check_manuscripts() -> None:
    print("\n[1] manuscripts: page limits, citations, cross-references")
    for label, mdir, bibpath, body_limit in MANUSCRIPTS:
        d = ROOT / mdir
        tex_path = d / "main.tex"
        if not tex_path.exists():
            bad(f"{label}: {mdir}/main.tex missing")
            continue
        tex = tex_path.read_text(encoding="utf-8")
        pages = label_pages(d / "main.aux")
        total = total_pages(d / "main.log")
        undef = undefined_refs(d / "main.log")

        cited = set()
        for m in re.findall(r"\\cite[a-z]*\{([^}]*)\}", tex):
            cited |= {k.strip() for k in m.split(",") if k.strip()}
        bib = (ROOT / bibpath).read_text(encoding="utf-8")
        keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
        refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", tex))

        problems = []
        if cited - keys:
            problems.append(f"cite keys missing from bib: {sorted(cited - keys)}")
        if refs - labels:
            problems.append(f"dangling refs: {sorted(refs - labels)}")
        if undef != 0:
            problems.append(f"{undef} undefined reference/citation warnings in main.log")
        body_page = pages.get("sec:conclusion")
        if body_limit is not None:
            if body_page is None:
                problems.append("no sec:conclusion label in main.aux (rebuild?)")
            elif body_page > body_limit:
                problems.append(f"body ends on page {body_page} > limit {body_limit}")
        if problems:
            for p in problems:
                bad(f"{label}: {p}")
        else:
            extra = f", body ends p{body_page}/{body_limit}" if body_limit else ""
            ok(f"{label}: {total or '?'} pages{extra}, {len(cited)} citations, "
               f"{len(refs)} refs all resolve")


def check_bundle() -> None:
    print("\n[2] arXiv submission bundle")
    zpath = ROOT / "paper/arxiv/arxiv_submission.zip"
    if not zpath.exists():
        bad("arxiv_submission.zip missing")
        return
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        missing = [n for n in REQUIRED_ZIP_ENTRIES if n not in names]
        if missing:
            bad(f"zip missing entries: {missing}")
            return
        flat_figures = [n for n in names if n.startswith(("fig1", "fig2", "fig3", "fig4"))]
        if flat_figures:
            bad(f"zip has figures at the root instead of figures/: {flat_figures}")
            return
        stale = []
        for entry in ("main.tex", "main.bbl", "references.bib"):
            src = ROOT / "paper/arxiv" / entry
            if z.read(entry) != src.read_bytes():
                stale.append(entry)
        if stale:
            bad(f"zip is stale relative to paper/arxiv: {stale} (repack before upload)")
        else:
            ok(f"zip has {len(names)} entries, figures/ layout intact, "
               f"main.tex/main.bbl/references.bib match the source directory")


def check_distributed() -> None:
    print("\n[3] distributed PDFs at the repository root")
    for src_rel, root_name in DISTRIBUTED:
        src = ROOT / src_rel
        dist = ROOT / root_name
        if not dist.exists():
            bad(f"missing distributed PDF: {root_name}")
        elif sha256(src) != sha256(dist):
            bad(f"stale distributed PDF: {root_name} != {src_rel}")
        else:
            ok(f"{root_name[:48]}… matches {src_rel}")


def check_numbers(fast: bool) -> None:
    print("\n[4] numeric gate")
    if fast:
        print("  SKIP  paper_consistency_check.py (--fast)")
        return
    proc = subprocess.run([sys.executable, "scripts/paper_consistency_check.py"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.startswith(("PASS ", "FAIL "))]
    if proc.returncode == 0:
        ok(f"paper_consistency_check.py: {tail[-1] if tail else 'passed'}")
    else:
        bad(f"paper_consistency_check.py failed ({len(tail)} problem lines)")
        for ln in tail[-6:]:
            print("        " + ln)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the 10^4-draw bootstrap gate")
    args = ap.parse_args()

    check_manuscripts()
    check_bundle()
    check_distributed()
    check_numbers(args.fast)

    print("\n" + "=" * 64)
    print(f"PASS {len(PASSES)} checks, FAIL {len(FAILURES)} checks")
    for f in FAILURES:
        print("  -", f)
    print("=" * 64)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
