# arXiv submission package (v4)

This directory is a self-contained, standard-LaTeX manuscript prepared for arXiv upload. It
matches the fourth revision of the Word draft: the paper is reorganized around the scientific
question, the reference implementation and engineering details live in Appendix A, the planned
figures are embedded, and the prose has been rewritten to remove formulaic drafting patterns.

## Build

```text
tectonic main.tex --outdir ../../output/pdf
```

The submission source includes `main.tex` (self-contained bibliography) and the six figure PNGs.
The generated PDF is for verification and does not need to be included in the source archive.

## Status and before submission

1. Replace `Anonymous Authors` with the real author/affiliation metadata. This is the only
   submission-blocking metadata intentionally left unresolved.
2. All quantitative claims in Section 6 (Expected Outcomes and Testable Predictions) and both
   tables are PRIOR PROJECTIONS, not measurements. Every number must be replaced by frozen-test,
   multi-seed, measured P50/P95 results before any submission claim.
3. The measured real-GPU run (Qwen3-4B to Qwen3-1.7B, 70-token pilot with key cosine 0.9661 and
   value cosine 0.5772) lives in the project reports, not in this draft. Decide whether to
   reinstate it as a preliminary-measurements subsection before submission.
4. Add license/ethics statements if required by the target category or institution.
5. Compile from a clean directory and check the final PDF page by page.

The source uses only common TeX Live packages accepted by arXiv and sets `\pdfoutput=1` for
PDFLaTeX-compatible processing.
