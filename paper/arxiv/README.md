# arXiv submission package

This directory is a self-contained, standard-LaTeX manuscript prepared for arXiv upload.

## Build

```text
tectonic main.tex --outdir ../../output/pdf
```

The submission source should include `main.tex` and `references.bib`. The generated PDF is for verification and does not need to be included in the source archive.

## Before submission

1. Replace `Anonymous Authors` with the real author/affiliation metadata. This is the only submission-blocking metadata intentionally left unresolved.
2. Rerun the corrected real-GPU T04/T05 experiments; the archived numbers are explicitly labeled as a 70-token preliminary pilot.
3. Do not promote synthetic T09, proxy T10, placeholder T12, or selected-repeat T13 outputs into measured result tables. The revised artifact providers and all-run T13 aggregation are the required paths.
4. Add license/ethics statements if required by the target category or institution.
5. Compare the corrected real experiment directly with Cross-Model KV Cache Transfer, C2C, and Mixture-of-Translators under the same zero-prefill receiver constraint.
6. Compile from a clean directory and check the final PDF page by page.

The source uses only common TeX Live packages accepted by arXiv and sets `\pdfoutput=1` for PDFLaTeX-compatible processing.
