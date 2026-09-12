# Evidence Map — `main.tex` (= `main_v2.tex`, `arxiv/main.tex`, and the ICLR body)

As of 2026-09-12 the three paper copies carry identical content and numbers:
`paper/cache_audit/main.tex` (article style), `paper/cache_audit/arxiv/main.tex`
(arXiv packaging) and `paper/cache_audit_iclr2026/main.tex` (ICLR 2026 template,
same body with the template's preamble and table-width settings). Figures 1--4 and
Tables 1--3 are generated/synced from the same sources; the figure scripts are
`figures/fig1_framework.tex` (Figure 1 body), `figures/gen_fig2_mapper_landscape.py`,
`figures/gen_fig3_oracle_probes.py`, `figures/gen_fig4_ppl_acc_decoupling.py`, and
the aggregate numbers come from `scripts/aggregate_claims.py`.

Pairs every empirical sentence in the new draft with the recorded artifact it
comes from. Numbers are copied from the artifacts, not re-derived by hand;
derived quantities are marked as such.

## Conventions

- `RUN <run-id>` = `reports/runs/<run-id>/inject-eval/metrics.json`
- `PS <run-id>`  = `reports/runs/<run-id>/inject-eval/inject_eval/capability_score_artifact.json`
- `R2`          = `reports/reconstruction_r2/reconstruction_r2.json`
- `AGG`         = output of `python scripts/aggregate_claims.py`

Evidence tiers (Section 3.3 of the draft):

| Tier | Meaning |
|---|---|
| **Confirmatory** | gate and decision rule fixed in the protocol before the runs (identity gate; replacement/gain gates at ε=0.02 on the primary pair; fraction and window probes) |
| **Frozen-gate extension** | family or axis added while revising, evaluated with the same frozen gates (RAT, MLP families, 8B teacher, calibration ladder, cross-architecture, token-aligned, long-context, octant sweep, refit variance) |
| **Exploratory / derived** | analysis choices and derivations made after the runs (sample-size calculation, R² decomposition, mechanical reading) |

## Decisions recorded for this version

1. **Seed convention.** Ranges are reported over *all independent trainings* of
   a configuration, with `n_train` stated: per-head MLP `n_train=5` at c=30 and
   c=200, joint MLP `n_train=4`. Seeded refits are reproducible given the seed;
   the one-off earlier trainings differ from them because they predate
   `joint_seed`/explicit seeding. The reported range is wider, not narrower,
   than a three-seed range, and the verdict is unchanged either way.
2. **Sample-size calculation** is reported as a derived, order-of-magnitude
   quantity with its method stated, not as a measured result.
3. **Run counts** use the reproducible definition "run directory containing an
   `inject-eval/metrics.json`" (162), replacing the earlier "96 audit-protocol
   runs", which no rule in the repository reproduces.

---

## Claim map

### A. Mechanics (H1) — confirmatory

| # | Claim | Evidence | Caveats |
|---|---|---|---|
| A1 | Identity injection reproduces the student's own prefill: per-sample logit cosine 0.99998; gold CHG +0.0001, CI [−0.0006, +0.0009], n=30 | `RUN v15-4b-to-1.7b-jointmlp-c30-s42-20260912-072431` (`ridge_self_kv`) | Single evaluation set; "exact" is a logit-cosine criterion, not token-level equality |
| A2 | Identity holds across all reported configurations: cross-architecture rows within ±0.001; token-aligned rows within ±0.0002; long-context +0.0000 [−0.0010, +0.0012] | `RUN v15-4b-x-*-rect-c30-s42-*`, `RUN v15-4b-x-*-rect-align-c30-s42-*`, `RUN v15-4b-to-1.7b-affine-c30-longctx-s42-20260912-072916` | Mechanism check only; does not generalize to other injection implementations |
| A3 | Zero re-prefill verified by cache-length counters | `zero_prefill_verified` field in `RUN *` (153/162 set) | Counting rule belongs in the appendix |
| A4 | Cold-start / disk-persistence equivalence | **NOT CLAIMED.** The only `online: true` run (`RUN v11-4b-to-1.7b-rotated-20260829-123002`) predates the audit protocol and has no paired offline counterpart | Stated as an explicit gap in Results and Limitations |

### B. Mapping (H2)

| # | Claim | Tier | Evidence | Caveats |
|---|---|---|---|---|
| B1 | Seven mapper families stay below the student's own prefill; point estimates from −0.138 to −0.392 | Confirmatory (core ladder) | `RUN v12-4b-1.7b-affine-20260829-142010`, `RUN v12-4b-1.7b-ridge-20260829-144617`, `RUN v13-*`, `RUN v14-*`, `RUN v15-4b-to-1.7b-mlp-c30-seed1-20260912-150059` | The span mixes evaluation sets; the fixed-set ladder (B3) is the controlled version |
| B2 | Best configuration (affine c=30, n=30) is −0.138 [−0.317, +0.029]; its interval includes zero | Confirmatory | `RUN v12-4b-1.7b-affine-20260829-142010` | Must be reported separately from the n=100 ladder row |
| B3 | Calibration budget is inert on a fixed evaluation set: c=10 → −0.251, c=30 → −0.249, c=60 → −0.238, c=100 → −0.257, full pool → −0.244 (spread 0.019) | Frozen-gate extension | `RUN v15-4b-to-1.7b-affine-c{10,30,60,100,500}-s42-*` | c=500 requests 500 but is pool-capped at 200; label requested vs effective |
| B4 | Mapper-training variance is small: five calibration draws span [−0.292, −0.249], SD ≈ 0.018 | Frozen-gate extension | `RUN v15-4b-to-1.7b-affine-c30-calibseed{43,44,45,46}-*` + canonical `...-20260829-231341` | Draw variance only; evaluation-set variance is carried by the bootstrap intervals |
| B5 | Nonlinearity does not help: per-head MLP c30 ∈ [−0.392, −0.257] (5 trainings), c200 ∈ [−0.256, −0.226] (5); joint MLP c30 ∈ [−0.302, −0.265] (4) | Frozen-gate extension | `RUN v15-4b-to-1.7b-mlp-c30-*`, `...-mlp-c200-*`, `...-jointmlp-c30-*` | Seed-dependent; report ranges + `n_train` |
| B6 | 8B teacher changes nothing: ridge c=500, 8B→1.7B = −0.224 [−0.320, −0.123], n=100 | Frozen-gate extension | `RUN v15-8b-to-1.7b-ridge-c500-20260830-131432` | Different family (ridge) from the main affine ladder |
| B7 | Weak student is not established as worse or non-inferior: 4B→0.6B +0.010 [−0.081, +0.102] (n=63); 8B→0.6B −0.023 [−0.104, +0.064] (n=100) | Confirmatory gate / derived power | `RUN v13-4b-to-0.6b-affine-c200-20260829-154145`, `RUN v15-8b-to-0.6b-affine-c30-s42-20260830-232500` | Near-chance absolute accuracy: student 0.286 and 0.280 |
| B8 | At the observed paired SD (≈0.40), ε=0.02 non-inferiority would need ≈500 evaluation samples; the 8B→0.6B point estimate (−0.023) sits at the margin and cannot clear it at any n | Exploratory (derived) | `PS v13-4b-to-0.6b-affine-c200-20260829-154145`, `PS v15-8b-to-0.6b-affine-c30-s42-20260830-232500` | Normal approximation to a one-sided 95% test; order of magnitude only |
| B9 | Long-context needle task fails the same way: teacher 0.733 / student 0.533 / translated 0.233 accuracy; gold CHG −0.285 [−0.534, −0.025], n=30 | Frozen-gate extension | `RUN v15-4b-to-1.7b-affine-c30-longctx-s42-20260912-072916` | No reproduction artifact for this row |
| B10 | Cross-architecture: one of five clears the replacement gate (Llama-3.2-1B, +0.000 [−0.012, +0.012], accuracy 0.300); Gemma-2-2B sits at the margin (−0.009 [−0.020, +0.002]); three fall short; none recovers teacher gold 0.691 | Frozen-gate extension | `RUN v15-4b-x-{llama1b,llama32-3b,gemma2-2b,gemma3-1b,qwen25-1.5b}-rect-c30-s42-*` | Position-matched after independent tokenization; mechanism/gate check |
| B11 | Token alignment preserves the negative direction on two genuinely cross-tokenizer pairs: Llama-3.2-3B −0.024 [−0.047, −0.004], Gemma-3-1B −0.073 [−0.134, −0.021]; Qwen2.5 alignment is a no-op | Frozen-gate extension | `RUN v15-4b-x-{llama32-3b,gemma3-1b,qwen25-1.5b}-rect-align-c30-s42-*` | **No aligned run for the two margin rows** (Llama-3.2-1B, Gemma-2-2B) |
| B12 | Aggregate: largest translation point estimate +0.010; exactly one distinct translated configuration clears the margin | Derived | `AGG` | Replaces "no configuration reaches replacement", which is false |

### C. Exploitability (H3)

| # | Claim | Tier | Evidence | Caveats |
|---|---|---|---|---|
| C1 | Fraction probe (RAT source): 0.503 → 0.380 at α=0.25, 0.249 at α=0.75; mononotone point estimate; intervals exclude zero from α=0.5 (−0.224 [−0.378, −0.082]; −0.254 [−0.421, −0.079]) but not at α=0.25 (−0.123 [−0.261, +0.009]) | Confirmatory | `RUN v14-4b-to-1.7b-rat-c30-probes-20260829-175933` | Mildest mixture is not significant |
| C2 | Fraction probe (affine source): −0.086 [−0.182, +0.002], −0.147 [−0.336, +0.019], −0.211 [−0.408, −0.022] | Frozen-gate extension | `RUN v15-4b-to-1.7b-affine-c30-probes-s42-20260829-234406` | Only α=0.75 excludes zero |
| C3 | Window probe: top third indistinguishable from the student (0.502 vs 0.503); middle and bottom thirds harmful (0.265, 0.224) | Confirmatory | same runs as C1/C2 | Sensitivity profile, not answer-routing localization |
| C4 | Octant sweep: only layers 10–13 exclude zero (−0.186 [−0.322, −0.060]); top three eighths span −0.011 to +0.020, including two small positive point estimates (+0.020, +0.006) whose intervals span zero | Frozen-gate extension | `RUN v15-4b-to-1.7b-affine-c30-probes-octant-s42-20260912-070436` | The positives are reported explicitly rather than summarized away |
| C5 | Native (norm-rescaled) teacher content degrades the student: −0.334 [−0.508, −0.154] (n=30); −0.266 [−0.363, −0.168] and −0.251 [−0.351, −0.152] on the fixed evaluation set | Confirmatory | `RUN v15-4b-to-1.7b-jointmlp-c30-s42-20260912-072431`; `RUN v15-4b-to-1.7b-affine-c{30,200}-*` | Content control, not a translation |
| C6 | Perplexity–accuracy decoupling: RAT no-anchor rank-32 restores PPL 23.7 (identity 21.2) but accuracy stays 0.267 (student 0.500); CHG −0.223 [−0.367, −0.088] | Frozen-gate extension | `RUN v14-4b-to-1.7b-rat-c30-noanchor-20260830-000054` | Single point, n=30; three PPL conventions must not be conflated |
| C7 | Joint MLP reconstructs a fluent cache (mean PPL 29.6 over 4 trainings vs student 25.4) yet fails (mean accuracy 0.200) | Frozen-gate extension | `RUN v15-4b-to-1.7b-jointmlp-c30-*` (4) | Means and ranges over the same training set |

### D. Mechanism

| # | Claim | Tier | Evidence | Caveats |
|---|---|---|---|---|
| D1 | Held-out reconstruction R²: keys pooled 0.923 / per-head 0.810; values pooled 0.297 / per-head 0.323 (20 fit, 10 held-out) | Confirmatory (measurement) | `R2` | Single split; the per-(layer,head) mean is the one quoted in the text |
| D2 | Repository ridge baseline on the same split: pooled R² 0.919 (K) / 0.291 (V) | Confirmatory | `reports/runs/t04-recon-r2-4b-1.7b/t04/metrics.json` | Different aggregation from D1; not an independent validation |
| D3 | Channel ablation: v-only accuracy 0.367 (PPL 1.85e7) > k-only 0.300 (PPL 253) > kv-both 0.200 (PPL 88.5); CHG −0.167, −0.276, −0.296 | Confirmatory (measurement) / exploratory (reading) | `RUN v12-4b-1.7b-ridge-20260829-144617` | v-only CHG interval includes zero |
| D4 | Teacher/student KV norm ratios 1.58× (K) and 7.44× (V) | Confirmatory | `RUN v15-4b-to-1.7b-jointmlp-c30-s42-20260912-072431` (`kv_norm_diagnostics`) | Scale is a first-order candidate, not the proven cause |
| D5 | "Consistent with a weight-mediated advantage" | Exploratory (interpretation) | A1–C6, D1 | Worded as interpretation, never as "capability is not in the cache" |

### E. Protocol and reproducibility

| # | Claim | Evidence | Caveats |
|---|---|---|---|
| E1 | Recorded evaluation runs: 162 (directories with a metrics file) | `AGG` | Replaces the unreproducible "96 audit-protocol runs" |
| E2 | Deterministic families reproduce within 0.002; gradient-trained families are seed-dependent and reported as ranges | `*-repro-20260912-*` runs; largest observed deviation 0.0014 (task-aware) | Long-context rows and three cross-architecture rows have no reproduction artifact |
| E3 | Run artifacts record git commit, mapper configuration, protocol flags, per-sample scores | `RUN *` fields | The `protocol_version` field is a stale constant and is therefore not cited as a protocol identifier |

---

## Per-section evidence check

### Abstract
logit cosine 0.99998 / +0.0001 [−0.0006, +0.0009] → A1. Seven families → B1.
Four scale pairs → setup. Five alternative students → B10. 162 runs → E1.
Point estimates −0.392…−0.138 → B1. Ladder [−0.257, −0.238] → B3. Only
margin-clearing translation is near chance → B12/B10. 0.6B intervals → B7.
Probe statement → C1/C2/C4. Native −0.334 → C5. PPL/accuracy → C6. R² +0.81 /
+0.32 → D1. No claim is made about cold-start equivalence (A4) or serving cost.

### 1. Introduction
Third paragraph (verdicts) → A1, B1, B2, B3, B7, C1, C4, C6. Contributions
paragraph → framework (A1, B3, C1), bounded design space (B1–B6), verdict
(B12, C1–C5). The introduction names Section 3.3 for the tier labels.

### 2. Related Work
Prior-work descriptions are qualitative. The only quantitative comparison
(73–98% retention) is attributed to `\citep{heo2026}` and immediately qualified
as not like-for-like (retention vs gain; shared vs disjoint splits). No number
from that line is compared against ours as if measured on the same protocol.
The replay attribution to MoT is flagged as inference in Limitations.

### 3. Framework and 3.3 tiers
ε = 0.02, the replacement/gain gate split, and the identity criterion (logit
cosine) are protocol decisions, not measurements. The multiplicity paragraph
rests on B12 (largest translation estimate +0.010; largest probe estimate
+0.020; no probe interval excludes zero on the positive side) and states the
exclusion of the superseded v1.1 runs.

### 4. Audit methodology
Mapper ladder hyperparameters, RAT components, probe definitions, split and
prompt discipline, and the run-accounting sentence (162 runs, E1) are
reproducibility statements; none carries a performance claim.

### 5.1 H1
A1, A2. The cold-start sentence now points to Limitations instead of asserting
equivalence (A4).

### 5.2 H2
B1, B2, B3, B4, B5, B6, B7, B8, B12; Table 2 rows map one-to-one to the runs
listed in the claim map; the gate column is the ε=0.02 rule applied to each
row's recorded interval.

### 5.3 (within 5.2) Weak student
B7, B8. The power sentence is the only derived quantity in Results and is
labelled as such.

### 5.4 Robustness
B9 (long-context), B10 (cross-architecture), B11 (token alignment, with the
unrun margin rows named).

### 5.5 H3
C1, C2, C3, C4, C5. Significance is stated per mixture fraction per source;
the two positive octant point estimates are reported explicitly.

### 5.6–5.7 Decoupling and channel asymmetry
C6, C7, D3 (measurement), with the mechanism reading marked exploratory.

### 6. Architectural analysis
D1, D2, D4; D5 is introduced here as interpretation and is labelled as such.

### 7. Implications
Restates B12 and C1–C5 at the level of "what a future claim must check". The
text-channel sentence carries the no-interval caveat from Limitations.

### 8. Limitations
Nothing new is claimed; each limitation points at a claim map entry: A4
(cold start), B11 (alignment coverage), C1–C5 (probe family), B5 (translator
capacity), B4/B5 (variance), C6/F3 (text channel), D1 (R² split), E2
(reproduction coverage), and prior work (MoT).

### 9. Conclusion
B12, C1–C5, D5.

---

## Open gaps (write `[NEEDS DATA: …]`, do not guess)

| Gap | What is missing | Unlocks |
|---|---|---|
| G1 | Paired offline/online runs under the audit protocol (store manifest + per-sample scores) | A4 cold-start equivalence |
| G2 | Token-aligned runs for Llama-3.2-1B and Gemma-2-2B | B11 coverage of the margin rows |
| G3 | Reproduction artifacts for the long-context row and for Llama-3.2-3B / Gemma-3-1B / Qwen2.5-1.5B | E2 coverage |
| G4 | Per-sample scores for the text-channel baseline, or a re-run under the current protocol | the text-channel sentence's evidence level |
| G5 | Second held-out split for the reconstruction R² | D1 range instead of a point |
| G6 | End-to-end serving measurements (prefill time, map/load time, VRAM) | any deployment-cost claim (currently absent by design) |
