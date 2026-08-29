## Methodology Transparency Review (SRQR-aware)

### MUST-FIX (submission blockers)
- No methodology blocker was surfaced by the fallback pass.

### SHOULD-FIX (quality improvements)
- (abstract) "Across six mapper families (per-head ridge, affine, per-layer affine, task-aware, a residual-anchored translator, and a per-head MLP), a controlled calibration ladder on a fixed evaluation set, and 28 recorded GPU runs, no configuration makes the strong student exceed its own prefill: gold-probability change spans to against a self-kv control at ." — Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.
- (method) "Student self-prefill is the baseline the handoff must not degrade." — Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### SRQR Checklist Deltas
- Sampling rationale: clarify how the evidence base supports the paper's strongest claims.
- Data collection details (time/place/duration): add context when results depend on specific settings.
- Coding process (stages, coders, disagreement resolution): specify if qualitative or hybrid analysis is used.
- Saturation: state whether the evidence scope is exhaustive or bounded.
- Triangulation: explain whether multiple evidence sources were reconciled.
- Reflexivity: acknowledge researcher choices that shape interpretation.
