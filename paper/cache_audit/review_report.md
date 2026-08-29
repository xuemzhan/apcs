# Deep Review — Cache Translation Across Heterogeneous LLMs: An Audit with Identity Controls and Oracle Bounds

Reviewer mode: paper-audit deep-review + independent reviewer pass, evidence-backed,
anchored to the paper text and the underlying run artifacts (21+ recorded GPU runs).

Legend: [S]=script-backed, [L]=reviewer judgment. Severity: MAJOR / MODERATE / MINOR.
All numeric claims were cross-checked against `figures/data/all_results.json` and
`reports/runs/*/inject-eval/metrics.json`.

---

## Verdict

**Direction: sound and publishable as a negative-result / methodology audit.
Submission gate: FAIL until MAJOR-A1 (H3 probe content source) and MAJOR-B1
(multi-seed contradiction) are resolved.**

Every headline number in the paper (H2 ladder, self-kv control, 0.6B parity,
probe monotonicity, PPL/accuracy decoupling, norm ratios) reconciles exactly
with the recorded data; the arXiv packaging is compliant. The residual risk is
not in the arithmetic; it is in two claim-strength decisions and one
experimental confound, all fixable.

---

## A. Scientific methodology

### A1 [MAJOR] Oracle probes mix content from the weakest translator, so H3's "bounds every translator" overclaims
- Quote (Sec. 4.4): "they hand the student its own cache and mix in teacher content at a controlled budget ... any improvement over α=0 is a lower bound on what a perfect translator could deliver; monotone degradation bounds exploitability at zero."
- Evidence: the probe run (`v14_4b_1.7b_rat_c30_probes`) sets `mapper.type=rat`; the mixed "teacher content" is the RAT output, whose own translation is poor (PPL 501 at c=30, gold 0.235). A monotone decline of *this* content in the student cache does not bound a *perfect* translator; it bounds "noisy teacher content does not help."
- Fix options (pick one, ideally two):
  1. Re-run probes with the best translator available (affine per-head, c=30: gold 0.365) as the mix source. If the monotone decline persists, the verdict is robust to translator quality.
  2. Add a "content upper bound" probe: native teacher KV, per-(layer,head) norm-aligned to student statistics, mixed in. This is the best geometric proxy for "ideal content" independent of any mapper.
  3. At minimum, weaken Sec. 4.4/Abstract to "no benefit from translated content under the tested translators," removing "bounds every translator / exploitability at zero" until (1) or (2) runs.
- Confidence: high that the probe is entangled with the mapper; medium on whether the corrected probe would change the verdict.

### A2 [MAJOR] The "scale ladder is non-monotone" claim confounds calibration budget with evaluation set
- Quote (Sec. 5.2): "the ladder is non-monotone in data: at 200 calibration examples the affine map degrades to −0.239 from −0.138 at 30."
- Evidence: c=30 numbers come from the v12 run (eval n=30, seed-42 sample set), c=200 from the v13 run (eval n=63, different shuffle and sample set). Calibration budget and evaluation set vary jointly, so the −0.138→−0.239 difference cannot be attributed to data volume.
- Fix: fix one evaluation set; fit affine at c=30 and c=200 on disjoint calibration subsets of the same pool; report both CHG. If the sign persists, the bias–variance narrative is supported; otherwise demote to "not observed."
- Note: the abstract already carries the causal reading ("no configuration makes the strong student exceed its own prefill"), which is safe; only the *direction of the ladder* is currently under-evidenced.

### A3 [MODERATE] H2 gate is internally inconsistent (CHG > 0 vs "no worse than self")
- Quote (Sec. 3.2): "some g_θ reaches replacement level (gate: CHG > 0 with CI excluding zero)" versus Fig 1 "replacement level (no worse than self)" and Sec. 5.2 "statistically indistinguishable from its own prefill, which is replacement-level service" (0.6B, +0.010, CI contains 0).
- A CHG whose CI contains zero cannot satisfy "CI excluding zero," yet it is called H2-success for the weak student. Use two gates explicitly: *replacement* (CI lower bound ≥ −ε) vs *gain* (CHG > 0, CI excluding zero). State which applies to each pair.

### A4 [MODERATE] λ-inertness is demonstrated only where λ is least relevant
- Quote (Sec. 5.2): "λ over [10⁻³, 10⁻¹] changes nothing."
- The sweep is at c=200, where data dominate the ridge solution. At c=30 (underdetermined, where λ matters) no sweep exists. Restrict the claim to c=200 or sweep λ at c=30.

### A5 [MODERATE] Weak-student "replacement-level service" is at near-chance absolute quality
- 0.6B student accuracy 0.286–0.302 vs random 0.25; handoff parity is parity near chance. The sentence "replacement-level service for a weak student" should add the absolute level, or a reviewer will read it as a deployment endorsement.

## B. Logic and internal consistency

### B1 [MAJOR] Conclusion-adjacent multi-seed claim points at content that does not exist
- Quote (Sec. 7): "the multi-seed sweep in the appendix-scale runs showed the same signs."
- The appendix ("Additional Measurements") contains no multi-seed data, and every audit-protocol result in the paper uses seed 42. Either add a 3-seed run for the headline configurations or delete the sentence. Do not cite the v1.1 three-seed sweeps: they run the excluded protocol (confidence metric, in-sample calibration, polluted cache).

### B2 [MODERATE] Table 1 mixes evaluation sizes without labeling them
- Rows c=30 (n=30) and c=200 (n=63) sit in one table with one caption; CI widths differ for that reason, not only because of the mapper. Add an evaluation-n column or footnote.

### B3 [MODERATE] Accuracy and gold probability are interleaved without unit labels
- e.g. "the teacher's summary of the context scores 0.413 against the student's 0.508" (both accuracy) sits next to gold-probability numbers elsewhere. It is internally consistent (verified against data) but a reader can conflate the two. Add "(accuracy)" / "(gold prob.)" at first use per figure/sentence.

## C. Novelty and positioning

### C1 [MODERATE] "None of the cited evaluations uses an identity control or an exploitability probe" is too absolute
- MoT performs injection-window root-cause analysis (U-shaped loss vs translation position) and correction-deficit ablations, which overlap the win_* probe idea. The defensible difference is: MoT includes a target-side replay/correction pass; this audit holds strict zero re-prefill and additionally separates mechanics via identity injection. Say that, not "none."

### C2 [MODERATE] The deliverable "audit checklist" is only in the conclusion
- The abstract/conclusion promise a checklist; promote it to a concrete boxed list (the four required instruments and their gates) so the methodological contribution is self-contained.

## D. Experimental closure

### D1 [MODERATE] No multi-seed evidence under the audit protocol
- See B1. Minimum: 3 seeds × {affine c30, affine c200} on the primary pair; report sign stability, not just mean.

### D2 [MODERATE] Probe space is one content source × linear blending × three windows
- Beyond A1, finer windows (per-octant) and a "best-translator" source are the two cheap extensions that would move H3 from suggestive to closed.

### D3 [MINOR] The architectural explanation (Sec. 6) is argued, not measured
- The claim "V_S is not a function of V_T" (kernel invisibility) is theoretically sound but unquantified. A two-line measurement would close it: reconstruction R² plateau of the best mapper on held-out pairs, reported alongside the PPL/acc decoupling.

## E. Writing style

- [S][MINOR] 137 sentence-complexity flags; the worst offenders are the RAT definition (86 words), the identity-control paragraph (45), and the zero-prefill counters sentence (45). Split them; the SNL style target is ~21 words.
- [S][MINOR] Four bibliography entries carry placeholder authors ("and others" / "Others"): KVComm, Interlat, HCache, activation-steering. arXiv requires real authors; complete or drop before upload.
- [S][MINOR] chktex unavailable in this environment (format layer skipped); paper-audit's 1.9/6 score is heuristic and dominated by Minor sentence flags; do not treat it as a rejection signal.
- [L][MINOR] Sec. 6 discussion depth: the mechanism section is substantive; the script's low "explanatory ratio" is a false positive on this section.
- [L][MINOR] Conclusion lacks a limitations pointer; the separate §8 exists, but add an explicit forward reference in the Conclusion.

---

## Revision roadmap (priority order)

1. **P0 — H3 honesty (A1):** probe with the best translator (affine c30) as content source; if feasible add the norm-aligned native-content probe; adjust Abstract/Sec. 4.4 wording either way.
2. **P0 — Internal contradiction (B1):** add a 3-seed × {affine c30, affine c200} run under the audit protocol, or delete the multi-seed sentence and state single-seed honestly.
3. **P1 — Controlled ladder (A2):** fix one eval set, vary calibration budget; relabel the ladder claim per outcome.
4. **P1 — Gate consistency (A3):** split replacement vs gain gates; state which gate each pair meets.
5. **P1 — Bibliography hygiene (E):** complete or remove placeholder-author entries.
6. **P2 — Numeric labeling (B2, B3, A5):** eval-n column in Table 1; unit labels on acc vs gold; near-chance caveat for 0.6B.
7. **P2 — Positioning (C1, C2):** sharpen the MoT difference; box the audit checklist.
8. **P2 — Closure (D2, D3, A4):** λ sweep at c=30; reconstruction R² plateau measurement; finer probe windows.

## Files reviewed
- paper/cache_audit/arxiv/main.tex (flattened arXiv copy, 11 pages, 0 compile errors)
- paper/cache_audit/main.tex (working copy)
- figures/data/all_results.json (83 recorded run metrics)
- reports/runs/*/inject-eval/metrics.json (spot-checked per claim)
