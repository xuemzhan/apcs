# Anonymous review version

This directory is the double-blind review copy of the paper. It is generated
from `paper/cache_audit/main.tex` and is content-identical to the signed
versions except for the front matter and one wording change (both listed below).

## Contents

| File | Notes |
|---|---|
| `main.tex` | article-style source, anonymous front matter |
| `references.bib` | 21 entries, all cited keys resolve |
| `figures/` | `fig1_framework.tex` (Figure 1 body) and the three generated figures |
| `main.pdf` | compiled review copy, 16 pages |

## Build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

Three passes after `bibtex` are needed for the cross-references and float
placement to settle; check for `Warning: Reference` in the log.

## What was changed for anonymity

1. The author block and the `authblk` package were removed (`\author{}`), and
   the PDF metadata author is `Anonymous`.
2. "the `apcs` repository's recorded runs" became "the accompanying code
   repository's recorded runs".
3. No URLs, acknowledgements, funding statements, or email addresses appear
   anywhere in the source or in the compiled PDF (verified by scanning the PDF
   text, the PDF metadata, and the raw PDF bytes).

## What was deliberately kept

The reproducibility appendix keeps the module paths that describe the
implementation (`apcs.inference.evaluator`, `apcs.mapper`, `apcs.data.hf_dataset`,
`apcs.inference.kv_store`) because they are what a reader needs to map each claim
to a component. They carry no author identity.

A venue-format anonymous copy (ICLR 2026 style, same content) is available at
`paper/cache_audit_iclr2026/main.pdf` if the submission requires that template.

## Code

The code repository is **not** included in this package. If reviewers are given
code, prepare a snapshot without `.git` (the commit history contains author
identifying information) and without any local absolute paths.
