# Claim Map (Step 1) — Cache Translation Across Model Scale: An Audit

**Purpose**: before any prose is written, this file lists every claim the paper will make, the
exact evidence that supports it, its evidence tier, and its caveats. Nothing here is invented:
each row points at a run directory (with the metric key) or a stored analysis artifact.

**Ground truth used** (the user supplied no separate result dump, so the repository's recorded
artifacts are treated as the experimental results):

| Source | What it provides |
|---|---|
| `reports/runs/<run>/inject-eval/metrics.json` | per-run `chg_gold.<method>` = mean, 95% bootstrap CI, n, p; `method_stats` = accuracy / gold / PPL; `kv_norm_diagnostics`; `psr`; `protocol_version`; `zero_prefill_verified` |
| `reports/runs/<run>/inject-eval/config.json` | exact model pair, mapper, calibration size, seeds, sequence length |
| `reports/reconstruction_r2/*.json` | held-out reconstruction R² (affine, joint-MLP, Heo top-k avg/concat) |
| `tmp/confirmatory_maxstat.json` | family-wise max-statistic bound over the six confirmatory probe configurations |
| `scripts/aggregate_claims.py` (stdout) | run/row counts and extremal rows |
| `docs/protocol/EXPERIMENT_LOG.md`, `docs/audits/audit*.md` | protocol history, errata, tier definitions |

**Evidence tiers** (used in the Confidence column, following the draft's own §3.3 labels):

- **C = confirmatory** — gate form fixed before the audit runs. Covers: the H1 identity gate;
  the non-inferiority *form* of the replacement gate plus the ε-free gain gate on the primary
  pair; the fraction and window probes under the rule "exploitability is established only if
  some configuration's interval excludes zero on the positive side".
- **F = frozen-gate extension** — added while revising, scored under the same gates and metrics
  (residual-anchored and MLP translators, 8B teacher, calibration ladder, cross-architecture and
  token-aligned students, 1K retrieval, octant sweep, refit variance).
- **E = exploratory** — derived after the runs from recorded metrics (sample-size calculation,
  R² decomposition, mechanical reading of the architectural section). Hypothesis-generating only.

**Numbering note**: rows are grouped by the section they will appear in. `[NEEDS DATA: …]`
marks a claim I cannot currently support at the precision the draft uses.

---

## 1. Abstract / Introduction-level claims

| # | Claim | Evidence source (table/row) | Tier | Caveats |
|---|---|---|---|---|
| 1.1 | Identity injection reproduces the student's own prefill bit-for-bit | `v12-4b-1.7b-ridge-20260829-144617` → `chg_gold.ridge_self_kv` = +0.0001 [-0.0006, +0.0009], n=30; `v15-4b-to-1.7b-affine-c30-s42-20260829-194933` → -4.7e-5 [-0.00052, +0.00039], n=232, `method_stats.ridge_self_kv.logit_cos_mean` = 0.999983 | C | Two run vintages with different n; the paper must state which n it reports per quoted number |
| 1.2 | No tested mapper lifts the strong student above its own prefill | Table 3 rows, all strong-student rows: affine c30 -0.138, affine c200 -0.239, per-layer -0.275, task-aware -0.216, ridge c30 -0.296, RAT -0.268/-0.221, Heo avg -0.273/-0.308/-0.291, Heo concat -0.279/-0.223/-0.254 (run IDs in §3 below) | C (primary pair) / F (added families) | n = 30–100 depending on row; two rows are ranges over trainings, not single intervals |
| 1.3 | Point estimates span −0.392 to −0.138 across the strong-student ladder | −0.138 = `v12-4b-1.7b-affine-20260829-140858`; −0.392 = `v15-4b-to-1.7b-mlp-c30-seed1-20260912-150059` | C / F | The −0.392 endpoint comes from one MLP training; the MLP family is seed-dependent (see 5.2) |
| 1.4 | Calibration budget is inert on the primary pair | `v15-…-affine-c10/c30/c60/c100/c200/c500` (tail-100 eval): −0.2511 / −0.2493 / −0.2378 / −0.2569 / −0.2438 / −0.2438 | F | All six are on the fixed tail-100 set (n=100); the LITERATURE-style c200 row quoted elsewhere in the draft uses a different eval set (n=63, −0.2389) |
| 1.5 | Three configurations clear the ε=0.02 reporting margin, all near-chance students | `scripts/aggregate_claims.py`: 4 rows / 3 distinct configs — `gemma2-2b-rect-align` −0.0066 [-0.0181, +0.0049]; `llama1b-rect-align` −0.0029 [-0.0112, +0.0053]; `llama1b-rect` +0.0001 [-0.0116, +0.0119] | F | "Clears the margin" ≠ non-inferior established; ε was selected after the gate form was frozen |
| 1.6 | 4B→0.6B is not established as worse or non-inferior | `v13-4b-to-0.6b-affine-c200-20260829-154145`: +0.0101 [-0.0806, +0.1018], n=63 | C | Largest point estimate in the audit; CI is wide relative to ε |
| 1.7 | Teacher full prefill is far above the student on the primary pair | Table 3 reference rows: teacher 0.767 acc / 0.693 gold; student 0.500 / 0.503; `method_stats` in the same runs | C | Reference row, not a translation result |
| 1.8 | Literature now contains near-native retention and above-receiver-baseline gains (motivation) | External citations only: CacheBridge 99.83% mean target retention on Qwen3; context-reuse layer LongBench2 27.59%→34.48% for Qwen2.5-7B→1.5B | — (external) | Not our measurement; must be cited, never merged with our numbers |

## 2. Framework / protocol claims (Method and Setup)

| # | Claim | Evidence source | Tier | Caveats |
|---|---|---|---|---|
| 2.1 | Primary pair: Qwen3-4B teacher → Qwen3-1.7B student, HellaSwag+ARC four-choice, gold-letter probability at the final prompt position | `configs/pair_qwen3.yaml`; every `inject-eval/config.json` in the listed runs | C | Metric definition must match the code (`gold` = softmax over four letter logits) |
| 2.2 | Strict zero re-prefill holds in every reported run | `metrics.json.zero_prefill_verified` = true in all listed runs | C | The flag is asserted by the code path; the audit never runs a target-side prefill for translated rows |
| 2.3 | Calibration and evaluation splits are disjoint | `metrics.json.calib_eval_disjoint` = true | C | — |
| 2.4 | Gate rule: replacement requires CI lower bound > −ε with ε = 0.02 selected as a reporting margin after the gate form was frozen; a separate gain gate requires CI lower bound > 0 | `metrics.json.capability_gate` = {metric: gold_prob, method: ridge_kv_both, chg, ci_low, gate}; draft §3.2 | C | ε sensitivity lives in the appendix; ε is **not** pre-registered |
| 2.5 | The audit covers 4 scale pairs, ≥5 alternative students, and the mapper families listed in §3 | run inventory (221 directories; 177 with metrics); `scripts/aggregate_claims.py` → 181 audit-protocol rows = 133 translations + 48 probes | C (pair count) / F (extensions) | **The draft's current counts (128 translation rows, 171 runs) are stale** — regenerate from the script |
| 2.6 | "Seven mapper families" | [NEEDS DATA: canonical family list] — repository mapper types include ridge, affine (per-head/per-layer), low-rank, RAT, MLP, joint MLP, task-aware, concat-ridge | — | The number 7 must be reconciled with the code's mapper registry before it is printed |

## 3. Results — H2 (replacement / mapper ladder), primary pair

All rows: `chg_gold.ridge_kv_both` unless noted; gold-probability change over the student's own prefill.

| # | Claim | Evidence source (run, n, value) | Tier | Caveats |
|---|---|---|---|---|
| 3.1 | Best linear mapper (affine per-head, c=30) does not clear the margin | `v12-4b-1.7b-affine-20260829-140858` n=30: −0.1376 [-0.3175, +0.0295] → n.e. | C | Single calibration draw; λ=1e-3 |
| 3.2 | With more calibration the same family degrades | `v13-4b-to-1.7b-affine-c200-repro-20260912-112952` n=63: −0.2389 [-0.3535, −0.1342] → deg.; `v15-4b-to-1.7b-affine-c200-s42-20260829-231903` n=100: −0.2438 [-0.3371, −0.1424] | C (c200) / F (tail-100) | Two eval sets; the draft's "c=200" row and its "calibration ladder" row are different runs |
| 3.3 | Per-head structure matters: per-layer affine is worse | `v13-4b-to-1.7b-affine_layer-c200-20260829-153202` n=63: −0.275 [-0.407, −0.140] | F | — |
| 3.4 | Ridge per-head is worse than affine per-head | `v12-4b-1.7b-ridge-20260829-140435` n=30: −0.2960 [-0.4792, −0.1211] | C | — |
| 3.5 | A task-aware objective improves the point estimate but still degrades | `v13-4b-to-1.7b-taskaware-c200-20260829-155727` n=63: −0.2164 [-0.3359, −0.1049] | F | Best strong-student row; PPL 142.1 |
| 3.6 | The architecture-anchored translator (RAT) fails too | `v14-4b-to-1.7b-rat-c30-20260829-174525` n=30: −0.2682 [-0.4398, −0.0918]; `v14-4b-to-1.7b-rat-c200-20260829-174926` n=63: −0.2207 [-0.3401, −0.1038] | F | — |
| 3.7 | Removing RAT's analytic anchor *improves* perplexity but not the answer deficit | `v14-4b-to-1.7b-rat-c30-noanchor-20260829-180839` n=30: −0.2231 [-0.3672, −0.0883], PPL 23.7 vs RAT c30 PPL 501.6 | F | Component ablation; PPL comparison is within the same run family |
| 3.8 | Non-linearity does not help (per-head MLP) | five runs `v15-…-mlp-c30-*`: −0.2573 … −0.3923 (n=30); five runs `v15-…-mlp-c200-*`: −0.2257 … −0.2560 (n=63) | F | Same-seed re-run disagrees: `mlp-c30-s42` −0.3034 vs `mlp-c30-s42-repro` −0.2573 (Δ=0.046) → **training is not reproducible given the seed** |
| 3.9 | Over-parameterized joint MLP does not help | five runs `v15-…-jointmlp-c30-*`: −0.2641 … −0.3015 (n=30) | F | Same seed reproduces (Δ=0.0007) unlike 3.8 |
| 3.10 | Heo-style top-k (averaged source layers, c=200) fails at every k | `v15-…-heo-topk{1,3,5}-ridge-c200-20260912-*` n=100: −0.2732 [-0.3800, −0.1577] / −0.3083 [-0.4169, −0.2000] / −0.2911 [-0.3992, −0.1923] | F | Averaged variant; c=200 audit contexts of 512 tokens |
| 3.11 | Aligning to the reference's concatenation + 1,024-token FineWeb-Edu-style calibration does not rescue it | `v15-…-heo-concat-fwe-k{1,3,5}-c100-20260913-*` n=100: −0.2787 [-0.4103, −0.1448] / −0.2227 [-0.3676, −0.0838] / −0.2544 [-0.3821, −0.1217] | F | 100 sequences, not the reference's 500 (63 GB host limit); calibration corpus is deterministically generated, not the reference's corpus |
| 3.12 | Identity control passes inside every one of the above runs | `chg_gold.ridge_self_kv` ≈ 0 (max |mean| ≤ 0.0002 across the listed runs) | C | — |
| 3.13 | Weak student (4B→0.6B, affine c200) reaches parity but not non-inferiority | `v13-4b-to-0.6b-affine-c200-20260829-154145` n=63: +0.0101 [-0.0806, +0.1018] → n.e. | C | Do not phrase as "established non-inferior" |
| 3.14 | Weak student with an 8B teacher also shows no gain | `v15-8b-to-0.6b-affine-c30-s42-20260830-232500` n=100: −0.0228 [-0.1041, +0.0636] → n.e. | F | — |
| 3.15 | 8B teacher → 1.7B student still degrades | `v15-8b-to-1.7b-ridge-c500-20260830-131432` n=100: −0.2241 [-0.3204, −0.1229] → deg. | F | — |

## 4. Results — H3 (teacher-content probes)

| # | Claim | Evidence source | Tier | Caveats |
|---|---|---|---|---|
| 4.1 | Confirmatory probe family: six configurations, none positive; family-wise bootstrap 95% upper estimate of the maximum gain = +0.009 | `v14-4b-to-1.7b-rat-c30-probes-20260829-175933` (n=30): mix_a25 −0.1233, mix_a50 −0.2239, mix_a75 −0.2544, win_low −0.2789, win_mid −0.2378, win_high −0.0009; `tmp/confirmatory_maxstat.json`: T=−0.0009, U_0.95=+0.0092, B=10000 | C | **Non-centered** percentile max, not a centered max-t: word it "bootstrap 95% upper estimate", not "simultaneous 95% exclusion". The draft's full-probe-set value (+0.075) has [NEEDS DATA: no artifact found] |
| 4.2 | Blended probes degrade monotonically with the teacher fraction | RAT c30 probes `method_stats`: self 0.5031 → a25 0.3798 → a50 0.2792 → a75 0.2486 (gold) | C | Translator-dependent in magnitude: the affine-c30 replication gives 0.5031 → 0.4169 → 0.3563 → 0.2922 |
| 4.3 | Replacing the top third of layers with translated content is harmless; middle and bottom thirds are harmful (RAT c30) | RAT c30 probes: win_high −0.0009 [-0.0096, +0.0095]; win_mid −0.2378; win_low −0.2789 | C | The mid/low ordering flips under the affine-c30 replication (mid −0.2588, low −0.1821) → claim only "top window ≈ neutral", not a stable layer ordering |
| 4.4 | Eight-way octant sweep: best row +0.0196, no interval excludes zero on the positive side | `v15-4b-to-1.7b-affine-c30-probes-octant-s42-20260912-070436` (affine c30, n=30): oct5 +0.0196 [-0.0113, +0.0529]; `scripts/aggregate_claims.py`: 0 probe rows with positive-excluding interval | E | n=30; best row selected over 8 windows |
| 4.5 | Direct native-cache injection also degrades (content control) | `v12-4b-1.7b-ridge-20260829-144617` / `v12-4b-1.7b-affine-repro-20260912-112407` → `chg_gold.ridge_native` = −0.3341 [-0.5078, −0.1539], n=30 | F | Control, not a translation result; native path is not in the online run |
| 4.6 | A text-channel summary baseline is significantly worse than the student | `v12-4b-1.7b-taskmix-summary-tail100-20260912-193441` → `method_stats.summary` acc 0.440 / gold 0.4036 vs student 0.52 / 0.5116 (Δgold = −0.108, n=100) | F | The CI [-0.181, −0.036] used in the draft is [NEEDS DATA: not present in metrics.json; only the point difference is computable] |

## 5. Analysis — reconstruction, channels, mechanism

| # | Claim | Evidence source | Tier | Caveats |
|---|---|---|---|---|
| 5.1 | Affine per-head mapper reconstructs keys but only ~1/3 of value variance | `reports/reconstruction_r2/reconstruction_r2.json` (20 fit / 10 held-out contexts): K per-(layer,head) R² = 0.810 (raw 0.821), V = 0.323 | E | Single fit; unit is the held-out context |
| 5.2 | A much better reconstructor does not improve task outcome | `reports/reconstruction_r2/joint_mlp_reconstruction.json`: K 0.854 (train 0.949), V 0.481 (train 0.647) vs affine K 0.810 / V 0.323; downstream joint-MLP CHG −0.2641 … −0.3015 (runs in 3.9) | E | Different fitting procedure (20/10 contexts) than the downstream runs |
| 5.3 | Source-layer combination changes reconstruction monotonically, not the verdict | average: K 0.8225/0.7871/0.7621, V 0.3173/0.2479/0.1816 (k=1/3/5); concat: K 0.8255/0.8583/0.8639, V 0.3259/0.4186/0.4283 — both from `reports/reconstruction_r2/heo_{,concat_}topk{1,3,5}_reconstruction.json`, while CHG stays negative in 3.10/3.11 | E (R²) + F (CHG) | Concat supports are per-target-layer; keep the two evidence types distinct in wording |
| 5.4 | Teacher/student KV norm ratio: K 1.58, V 7.44 | `v12-4b-1.7b-affine-20260829-142010` → `kv_norm_diagnostics.K.ratio_mean` = 1.5809, `V.ratio_mean` = 7.4359 | E | Layer-wise spread is large (V ratio ranges ~0.2 to ~30) |
| 5.5 | Channel ablation: value-only ≳ key-only ≳ both on accuracy, with perplexity ordered the other way | `v12-4b-1.7b-ridge-20260829-140435` (n=30): v_only −0.1666 [-0.3739, +0.0466] acc 0.367 PPL 1.85e7; k_only −0.2764 [-0.4437, −0.1098] acc 0.300 PPL 252.8; kv_both −0.2960 acc 0.200 PPL 88.5 | E | Single run, n=30; report as an observation, not a mechanism proof |
| 5.6 | Fluent cache ≠ usable cache | `v14-…-rat-c30-noanchor` PPL 23.7 with gold 0.280 (student gold 0.503); `v12-…-ridge` v_only PPL 1.85e7; `jointmlp-c30` PPL 29.6 with CHG ≈ −0.27 | E | PPL and accuracy come from the same runs; the decoupling figure must plot exactly these points |
| 5.7 | Naive target-side replay does not recover the student | `v15-4b-to-1.7b-affine-c30-replay-s42-20260912-110006` → `chg_gold.ridge_replay` = −0.2321 [-0.3329, −0.1363] vs `ridge_kv_both` (no replay) −0.2493 [-0.3500, −0.1487], n=100 | F | Replay violates zero re-prefill; not a re-implementation of any published correction loss |

## 6. Robustness, validity and system-level claims

| # | Claim | Evidence source | Tier | Caveats |
|---|---|---|---|---|
| 6.1 | Long-context failure is not a 1K artifact | 1K: `v15-4b-to-1.7b-affine-c30-longctx-s42-20260912-072916` n=30 −0.2849 [-0.5343, −0.0251]; 4K: `v15-…-c30-longctx4k-20260913-144905` n=10 −0.5964 [-0.9289, −0.1703]; 8K: `v15-…-c10-longctx8k-20260913-150122` n=10 −0.5771 [-0.9317, −0.1184] | F | 4K/8K use n=10 and `eval_from_tail_n=0` with float16 caches — **not** the fixed tail-100 protocol; state this explicitly |
| 6.2 | An earlier "4K" row was in fact 1K (plumbing bug), now fixed | `docs/protocol/EXPERIMENT_LOG.md` errata block; code change in `apcs/inference/cli.py` (passes `target_tokens`) | — (erratum) | The buggy row must not be cited; the erratum should stay in the appendix |
| 6.3 | Cross-architecture students do not gain | `v15-4b-x-*`: Llama-3.2-1B +0.0001 [-0.0116, +0.0119]; Gemma-2-2B −0.0089 [-0.0202, +0.0021]; Llama-3.2-3B −0.0437 [-0.0729, −0.0132]; Gemma-3-1B −0.0612 [-0.1349, +0.0077]; Qwen2.5-1.5B −0.1571 [-0.2085, −0.1059] (all n=100) | F | Three receivers are near chance; the check establishes mechanics and gate-level replacement, not capability |
| 6.4 | Token alignment does not change the verdict | aligned runs: Llama-3.2-3B −0.0237 [-0.047, −0.004]; Gemma-3-1B −0.0732 [-0.134, −0.021]; Llama-3.2-1B −0.0029 [-0.0112, +0.0053]; Gemma-2-2B −0.0066 [-0.0181, +0.0049] (all n=100) | F | Alignment is a no-op for the shared-tokenizer student |
| 6.5 | Deterministic families reproduce; gradient-trained families are seed-dependent | `docs/protocol/EXPERIMENT_LOG.md` reproduction audit row (23 configs; |Δ| ≤ 0.002 for the deterministic families) | F | **Correction needed**: per-head MLP re-run with identical config/seed differs by 0.046 (see 3.8). The draft's "the seeded refits reproduce given the seed" cannot stand as written for that family |
| 6.6 | Cold start is exact on the translated and identity paths | `v15-…-affine-c30-persist-20260912-200748` vs `v15-…-affine-c30-online-20260912-201353`: `ridge_kv_both` −0.13756 in both; `ridge_self_kv` +0.00013 in both; identical acc/gold | F | **Scope it**: `ridge_native` differs between the two (−0.3341 offline vs −0.2389 online) because the online path has no native cache. One run each, not repeated pairs |
| 6.7 | Aggregate audit statistics | `scripts/aggregate_claims.py`: 177 runs with metrics; 181 audit-protocol rows (133 translations, 48 probes); largest translation point estimate +0.0101 [-0.0806, +0.1018] | C / F | Draft says "128 translated configuration rows" and README says 171 runs → **regenerate and reconcile before printing** |
| 6.8 | No deployment cost claim is made | `metrics.json.psr.mean` = −9.67 (n=232) for `v15-…-affine-c30-s42-20260829-194933`; −3.4 quoted in the log for the v1.2 measurement path | E | Both include a measurement-only forward pass; the draft must keep stating that no serving-cost claim is made |

## 7. Claims the draft currently makes that are NOT in the evidence

| # | Draft wording | Status |
|---|---|---|
| 7.1 | "Any positive average gain larger than about one gold-probability point is therefore excluded at 95% confidence" | Overstated for a non-centered percentile max; use 4.1's wording |
| 7.2 | Family-wise bound over the full probe set = +0.075 | [NEEDS DATA: no stored artifact; either recompute and save it or drop the sentence] |
| 7.3 | Text-channel CI [−0.181, −0.036] | [NEEDS DATA: artifact not found; point difference −0.108 is computable from `method_stats`] |
| 7.4 | "Seven mapper families" | [NEEDS DATA: canonical family list] |
| 7.5 | Power statement: "clearing the reporting margin would take roughly an order of magnitude more evaluation samples" | [NEEDS DATA: no power-analysis artifact; the draft cites a paired-variance calculation that must be shown] |
| 7.6 | "the seeded refits reproduce given the seed" | Contradicted for the per-head MLP family (3.8, 6.5); needs qualification or removal |

## 8. Known blockers before Results prose can be drafted

1. **Row/run counts** must be regenerated (`scripts/aggregate_claims.py`) and used consistently in the abstract, Table 3 caption and §5.2.
2. **Which n** each quoted number uses must be attached per row (30 / 63 / 100 / 10 are all in play).
3. **Two probe translators** (RAT c30 confirmatory; affine c30 replication) must be labelled wherever probe numbers appear.
4. **The MLP reproducibility statement** must be corrected.
5. **4K/8K rows** must be labelled as a different evaluation regime (float16 caches, `eval_from_tail_n=0`, n=10).
6. **Cold-start claim** must be scoped to the translated/identity paths.
