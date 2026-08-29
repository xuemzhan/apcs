# Deep Review Report

**Paper**: `/workspace/apcs/paper/cache_audit/main.tex` | **Language**: EN | **Mode**: deep-review
**Generated**: 2026-08-29 21:06
**Artifacts**: `/workspace/apcs/review_results/cache-translation-across-heterogeneous-llms-an-audit-with-identity-controls-and-oracle-bounds`

## Overall Assessment

Deep review found 1 major, 3 moderate, 0 minor issues. The highest-priority concerns are: Abstract and conclusion claims need explicit evidence traceability; Cross-section numeric consistency should be reconciled.

- **Major**: 1
- **Moderate**: 3
- **Minor**: 0

## Academic Pre-Review Committee

### Editor (Desk Reject Screen)

## Editor Pre-Screen (1-10)

Score: 6.1/10
Verdict: Pass to Review

### Desk-Reject Triggers (if any)
- Abstract and conclusion claims need explicit evidence traceability

### Top 3 Reasons (no hedging)
1. Abstract and conclusion claims need explicit evidence traceability

### Fast Fixes (within 1-2 days)
- Clarify abstract to address abstract and conclusion claims need explicit evidence traceability.
- Clarify abstract to address cross-section numeric consistency should be reconciled.
- Clarify related_work to address novelty claim should be grounded against the closest prior work.

### Reviewer 1 (Theory Contribution)

## Theory Contribution Review

### 3 Fatal Theory Holes
1. (abstract) Abstract and conclusion claims need explicit evidence traceability — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.
2. (related_work) Novelty claim should be grounded against the closest prior work — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Concrete Moves
- Tighten the paper's theoretical positioning in abstract to resolve abstract and conclusion claims need explicit evidence traceability.
- Tighten the paper's theoretical positioning in related_work to resolve novelty claim should be grounded against the closest prior work.

### Reviewer 3 (Literature Dialogue)

## Literature Dialogue Review

### Closest Prior Work Risks
- (related_work) Novelty claim should be grounded against the closest prior work — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Gap Claim Risks
- The claimed gap should be defended more explicitly: Novelty claim should be grounded against the closest prior work.

### Fast Fixes
- Name the closest prior comparator in related_work and explain the real novelty delta.

### Reviewer 2 (Methodology & Transparency)

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

### Reviewer 4 (Logic Chain)

## Logic Chain Review

### Breakpoints
- (abstract) Abstract and conclusion claims need explicit evidence traceability — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

### Structural Fix Moves
- Add one explicit bridge sentence in abstract so the argument chain closes cleanly.

### Committee Consensus

## Committee Consensus

Overall Score: 5.4/10
Editor Verdict: Pass to Review

### Score Formula
- base 9.0
- minus 1.5 * major (1)
- minus 0.7 * moderate (3)
- minus 0.2 * minor (0)
- floor 1.0
- desk reject cap 4.0

### Top 3 Issues To Fix First
1. Abstract and conclusion claims need explicit evidence traceability
2. Cross-section numeric consistency should be reconciled
3. Comparison protocol should make fairness assumptions explicit

## Paper Summary

# Paper Summary: Cache Translation Across Heterogeneous LLMs:\\ An Audit with Identity Controls and Oracle Bounds

## Research Question
- Cache translation maps the key-value (KV) cache that a large teacher model forms over a context into the cache space of a smaller student model, so that the student answers without re-prefilling the context

## Core Thesis
- We conclude that the teacher's answer-relevant advantage does not survive KV-space translation under zero re-prefill: it lives in the teacher's weights, not in its cache.

## Headline Claims
- We conclude that the teacher's answer-relevant advantage does not survive KV-space translation under zero re-prefill: it lives in the teacher's weights, not in its cache.

## Section Map
- abstract (36-63): 278 words
- introduction (65-195): 925 words
- method (312-347): 300 words
- method_2 (348-418): 546 words
- result (419-569): 1169 words
- discussion (570-595): 243 words
- conclusion (640-656): 116 words

## Closure Targets
- No closure target was extracted automatically.

## Major Issues

### M1: Abstract and conclusion claims need explicit evidence traceability
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: low
- **Section**: abstract
- **Related Sections**: abstract, results, conclusion
- **Root Cause Key**: `abstract-and-conclusion-claims-need-explicit-evidence-traceability`
- **Quote Verified**: no
- **Quote**: —
- **Explanation**: At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

## Moderate Issues

### M1: Cross-section numeric consistency should be reconciled
- **Type**: presentation
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: medium
- **Section**: abstract
- **Related Sections**: abstract, introduction, method
- **Root Cause Key**: `cross-section-numeric-consistency-should-be-reconciled`
- **Quote Verified**: no
- **Quote**: `Across six mapper families (per-head ridge, affine, per-layer affine, task-aware, a residual-anchored translator, and a per-head MLP), a controlled calibration ladder on a fixed evaluation set, and 28 recorded GPU runs, no configuration makes the strong student exceed its own prefill: gold-probability change spans to against a self-kv control at .`
- **Explanation**: Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.

### M2: Comparison protocol should make fairness assumptions explicit
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: medium
- **Section**: method
- **Related Sections**: method
- **Root Cause Key**: `comparison-protocol-should-make-fairness-assumptions-explicit`
- **Quote Verified**: no
- **Quote**: `Student self-prefill is the baseline the handoff must not degrade.`
- **Explanation**: Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### M3: Novelty claim should be grounded against the closest prior work
- **Type**: claim_accuracy
- **Source**: [LLM] via `prior_art_and_novelty_grounding`
- **Confidence**: low
- **Section**: related_work
- **Related Sections**: related_work, results
- **Root Cause Key**: `novelty-claim-should-be-grounded-against-the-closest-prior-work`
- **Quote Verified**: no
- **Quote**: —
- **Explanation**: The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

## Phase 0 Automated Findings

### [Script] BIB

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Check: /workspace/apcs/paper/cache_audit/main.tex |
| --- | Minor | PASS |
| --- | Minor | entries: 0 |
| --- | Minor | entries: 0 |

### [Script] CITATIONS

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | No citation stacking issues found. |

### [Script] DEAI

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Use --analyze for full analysis |

### [Script] EXPERIMENT

| Line | Severity | Issue |
|------|----------|-------|
| 478 | Major | Performance claim lacks an explicit baseline or comparator. |
| 419 | Minor | No ablation or component-level evidence is mentioned; verify that contribution attribution is covered. |
| 570 | Major | Discussion may lack depth: low ratio of explanatory/attribution language (0/23 lines). Add causal analysis explaining why results occur. |
| 570 | Major | Discussion may lack layered structure: it should cover at least two categories (mechanism, comparison with prior work, limitations/boundaries, implications/outlook) instead of only restating results. |
| 640 | Major | Conclusion lacks limitations or future work discussion. |
| 640 | Minor | Conclusion lacks implications or broader impact statement. |
| 640 | Minor | Conclusion lacks explicit summary of core findings. |

### [Script] FIGURES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | figures in /workspace/apcs/paper/cache_audit/main.tex... |
| --- | Minor | 3 figures. |
| --- | Minor | Line 434: figures/fig2_mapper_landscape.pdf |
| --- | Minor | Line 513: figures/fig3_oracle_probes.pdf |
| --- | Minor | Line 543: figures/fig4_ppl_acc_decoupling.pdf |
| --- | Minor | All figures passed check. |

### [Script] FORMAT

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | ============================================================ |
| --- | Minor | Format Check Report |
| --- | Minor | ============================================================ |
| --- | Minor | /workspace/apcs/paper/cache_audit/main.tex |
| --- | Minor | UNAVAILABLE |
| --- | Minor | chktex not found. Install with: apt-get install chktex (Linux) or via TeX Live/MiKTeX |
| --- | Minor | MODE] chktex not available |
| --- | Minor | chktex for detailed format checking |

### [Script] GRAMMAR

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | [Script]: goal=grammar strength=minimal |
| --- | Minor | No rule-based issues detected in selected scope. |

### [Script] LOGIC

| Line | Severity | Issue |
|------|----------|-------|
| --- | Major | Abstract, contribution claims, and conclusion may be misaligned. |
| --- | Minor | conclusion missing result; abstract missing contribution claim; conclusion missing contribution response; conclusion missing result evidence. |
| --- | Minor | Make sure all three sections consistently state the problem, method, key results, and contribution. |
| --- | Minor | These sections should tell the same core story with different emphasis, not diverge. |

### [Script] PRESUBMISSION

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | [A1] Abstract five-element check is incomplete; missing background. |
| 36 | Minor | [G2] Long paragraph detected (280 words, 12 sentences); split or add a clearer topic sentence. |
| 140 | Minor | [G2] Long paragraph detected (394 words, 2 sentences); split or add a clearer topic sentence. |
| 216 | Minor | [G2] Long paragraph detected (198 words, 9 sentences); split or add a clearer topic sentence. |
| 369 | Minor | [G2] Long paragraph detected (239 words, 9 sentences); split or add a clearer topic sentence. |
| 476 | Minor | [G2] Long paragraph detected (195 words, 10 sentences); split or add a clearer topic sentence. |
| 380 | Minor | [L4] Numbered equation environment has no label for later reference. |

### [Script] REFERENCES

| Line | Severity | Issue |
|------|----------|-------|
| 104 | Minor | Reference before definition: \ref{fig:framework} at line 104 appears before label definition at line 193 |
| 200 | Minor | Reference before definition: \ref{tab:related} at line 200 appears before label definition at line 288 |

### [Script] SENTENCES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | [Script]: goal=grammar strength=minimal |
| --- | Minor | SENTENCE (Line 69, 37 words, 6 clauses)  [Script] |
| --- | Minor | Recent systems report encouraging results, including Mixture-of-Translators (MoT), which preserves 51.0\% closed-set QA accuracy at 7B 0.5B scale~\citep{mot2026}, Cache-to-Cache semantic communication~\citep{c2c2025}, latent-space communication~\citep{lsc2026}, and selective KV sharing~\citep{kvcomm2025}. |
| --- | Minor | Recent systems report encouraging results. including Mixture-of-Translators (MoT). which preserves 51.0\% closed-set QA accuracy at 7B 0.5B scale~\citep{mot2026}. Cache-to-Cache semantic communication~\citep{c2c2025}. latent-space communication~\citep{lsc2026}. and selective KV sharing~\citep{kvcomm2025}.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 124, 31 words, 4 clauses)  [Script] |
| --- | Minor | \begin{enumerate} \item \textbf{An audit framework.} Identity controls, disjoint calibration/evaluation splits, gold-probability metrics with bootstrap intervals and permutation tests, and oracle probes that bound exploitability without training anything (Section~ ). |
| --- | Minor | \begin{enumerate} \item \textbf{An audit framework.} Identity controls. disjoint calibration/evaluation splits. gold-probability metrics with bootstrap intervals and permutation tests. and oracle probes that bound exploitability without training anything (Section~ ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 124, 34 words, 4 clauses)  [Script] |
| --- | Minor | \item \textbf{A bounded design space.} Five mapper families, a calibration ladder, component ablations, and a residual-anchored translator derived from the two models' projections and shared vocabulary, all under one protocol (Section~  and Section~ ). |
| --- | Minor | \item \textbf{A bounded design space.} Five mapper families. a calibration ladder. component ablations. and a residual-anchored translator derived from the two models' projections and shared vocabulary. all under one protocol (Section~  and Section~ ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 140, 369 words, 47 clauses)  [Script] |
| --- | Minor | \begin{figure}[t] \centering \begin{tikzpicture}[ node distance=4mm and 6mm, box/.style={draw, rounded corners, align=center, inner sep=4pt, minimum width=34mm, font=\scriptsize}, qbox/.style={box, fill=blue!8}, hyp/.style={box, fill=orange!12, minimum width=44mm}, inst/.style={box, fill=green!10, minimum width=44mm}, verdict/.style={box, fill=gray!10, minimum width=44mm}, gate/.style={diamond, draw, aspect=2.2, inner sep=1.5pt, font=\tiny\bfseries}, arr/.style={-{Latex[length=2mm]}, thick}, lab/.style={font=\tiny\bfseries}, ] \node[box, fill=red!8] (mot) {\textbf{Claim under audit}\\ translated teacher cache\\ lifts a frozen student (MoT, C2C, LSC, KVComm)}; \node[qbox, below=5mm of mot] (q) {\textbf{Central question}\\ \emph{Does the teacher's answer-relevant}\\ \emph{capability survive KV-space}\\ \emph{translation under zero re-prefill?}}; \node[hyp, below=6mm of q] (h1) {\textbf{H1 Mechanics}: foreign cache is\\ consumed without loss}; \node[inst, below=4mm of h1] (i1) {Identity control: inject student's\\ \emph{own} cache through the full pipeline}; \node[hyp, below=5mm of i1] (h2) {\textbf{H2 Mapping}: a translator reaches\\ replacement level (no worse than self)}; \node[inst, below=4mm of h2] (i2) {Mapper ladder: ridge / affine / per-layer /\\ task-aware / RAT   calib.\ 30--200}; \node[hyp, below=5mm of i2] (h3) {\textbf{H3 Exploitability}: teacher cache\\ contains student-readable advantage}; \node[inst, below=4mm of h3] (i3) {Oracle probes: blend teacher content into\\ student cache at fraction   / layer windows}; \node[verdict, below=6mm of i3, minimum width=120mm] (v) {\textbf{Verdicts}: H1 \checkmark\ (logit cos.\  ) \quad H2 \textdagger\ (best  ; weak student  ) \quad H3   (monotone probe decline; PPL/acc.\ decoupling)}; \node[box, fill=blue!8, below=4mm of v, minimum width=120mm] (c) {\textbf{Consequence}: replacement is achievable for weak students; capability transfer via cache translation is bounded out\\ for frozen students; audit checklist required for future claims}; \draw[arr] (mot) -- (q); \draw[arr] (q) -- (h1); \draw[arr] (h1) -- node[lab, right] {test} (i1); \draw[arr] (i1) -- node[lab, right] {pass   proceed} (h2); \draw[arr] (h2) -- node[lab, right] {test} (i2); \draw[arr] (i2) -- node[lab, right] {fail   probe ceiling} (h3); \draw[arr] (h3) -- node[lab, right] {test} (i3); \draw[arr] (i3) -- (v); \draw[arr] (v) -- (c); \end{tikzpicture} \caption{The three-hypothesis audit as a decision pipeline. |
| --- | Minor | \begin{figure}[t] \centering \begin{tikzpicture}[ node distance=4mm and 6mm. box/.style={draw. rounded corners. align=center. inner sep=4pt. minimum width=34mm. font=\scriptsize}. qbox/.style={box. fill=blue!8}. hyp/.style={box. fill=orange!12. minimum width=44mm}. inst/.style={box. fill=green!10. minimum width=44mm}. verdict/.style={box. fill=gray!10. minimum width=44mm}. gate/.style={diamond. draw. aspect=2.2. inner sep=1.5pt. font=\tiny\bfseries}. arr/.style={-{Latex[length=2mm]}. thick}. lab/.style={font=\tiny\bfseries}. ] \node[box. fill=red!8] (mot) {\textbf{Claim under audit}\\ translated teacher cache\\ lifts a frozen student (MoT. C2C. LSC. KVComm)}; \node[qbox. below=5mm of mot] (q) {\textbf{Central question}\\ \emph{Does the teacher's answer-relevant}\\ \emph{capability survive KV-space}\\ \emph{translation under zero re-prefill?}}; \node[hyp. below=6mm of q] (h1) {\textbf{H1 Mechanics}: foreign cache is\\ consumed without loss}; \node[inst. below=4mm of h1] (i1) {Identity control: inject student's\\ \emph{own} cache through the full pipeline}; \node[hyp. below=5mm of i1] (h2) {\textbf{H2 Mapping}: a translator reaches\\ replacement level (no worse than self)}; \node[inst. below=4mm of h2] (i2) {Mapper ladder: ridge / affine / per-layer /\\ task-aware / RAT   calib.\ 30--200}; \node[hyp. below=5mm of i2] (h3) {\textbf{H3 Exploitability}: teacher cache\\ contains student-readable advantage}; \node[inst. below=4mm of h3] (i3) {Oracle probes: blend teacher content into\\ student cache at fraction   / layer windows}; \node[verdict. below=6mm of i3. minimum width=120mm] (v) {\textbf{Verdicts}: H1 \checkmark\ (logit cos.\  ) \quad H2 \textdagger\ (best  ; weak student  ) \quad H3   (monotone probe decline; PPL/acc.\ decoupling)}; \node[box. fill=blue!8. below=4mm of v. minimum width=120mm] (c) {\textbf{Consequence}: replacement is achievable for weak students; capability transfer via cache translation is bounded out\\ for frozen students; audit checklist required for future claims}; \draw[arr] (mot) -- (q); \draw[arr] (q) -- (h1); \draw[arr] (h1) -- node[lab. right] {test} (i1); \draw[arr] (i1) -- node[lab. right] {pass   proceed} (h2); \draw[arr] (h2) -- node[lab. right] {test} (i2); \draw[arr] (i2) -- node[lab. right] {fail   probe ceiling} (h3); \draw[arr] (h3) -- node[lab. right] {test} (i3); \draw[arr] (i3) -- (v); \draw[arr] (v) -- (c); \end{tikzpicture} \caption{The three-hypothesis audit as a decision pipeline.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 351, 41 words, 5 clauses)  [Script] |
| --- | Minor | \subsection{Identity Control} At evaluation time we capture the student's own cache over the context (allowed in calibration and probe stages), feed it through the same mapper interface, cache construction, position handling, and scoring path as a translated cache, and score. |
| --- | Minor | \subsection{Identity Control} At evaluation time we capture the student's own cache over the context (allowed in calibration and probe stages). feed it through the same mapper interface. cache construction. position handling. and scoring path as a translated cache. and score.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 358, 45 words, 4 clauses)  [Script] |
| --- | Minor | Zero-prefill counters read the actual cache length before and after the scoring forward pass rather than trusting bookkeeping, and the scoring pass uses an isolated cache because the deployed transformers version mutates a cache even under \texttt{use\_cache=False}, which silently pollutes co-located measurements. |
| --- | Minor | Zero-prefill counters read the actual cache length before and after the scoring forward pass rather than trusting bookkeeping. and the scoring pass uses an isolated cache because the deployed transformers version mutates a cache even under \texttt{use\_cache=False}. which silently pollutes co-located measurements.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 369, 86 words, 12 clauses)  [Script] |
| --- | Minor | \textbf{RAT (residual-anchored translator)}: an architecture-derived map \begin{equation} W^{(s,h)} \;=\; \bigl[\,\pi(W^{S,h}_V)\;R\;\pi(W^{T,h}_V)\,\bigr]^{\!\top}, \qquad R \;=\; \arg\min_R \; \|E_T R - E_S\|_F^2 + \lambda \|R\|_F^2, \end{equation} where   and   are the two models' own value projections at the aligned layer pair,   is the pseudo-inverse pullback from a head's value space to the residual stream, and   aligns the two residual streams through the shared 151{,}936-token embedding matrices, which both Qwen3 models ship. |
| --- | Minor | \textbf{RAT (residual-anchored translator)}: an architecture-derived map \begin{equation} W^{(s. h)} \;=\; \bigl[\. \pi(W^{S. h}_V)\;R\;\pi(W^{T. h}_V)\. \bigr]^{\!\top}. \qquad R \;=\; \arg\min_R \; \|E_T R - E_S\|_F^2 + \lambda \|R\|_F^2. \end{equation} where   and   are the two models' own value projections at the aligned layer pair. is the pseudo-inverse pullback from a head's value space to the residual stream. and   aligns the two residual streams through the shared 151{. }936-token embedding matrices. which both Qwen3 models ship.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 412, 25 words, 4 clauses)  [Script] |
| --- | Minor | \subsection{Pairs, Data, and Protocol Discipline} Primary pair: Qwen3-4B (36 layers, hidden 2560)   Qwen3-1.7B (28 layers, hidden 2048); contrast pair:   Qwen3-0.6B (hidden 1024). |
| --- | Minor | \subsection{Pairs. Data. and Protocol Discipline} Primary pair: Qwen3-4B (36 layers. hidden 2560)   Qwen3-1.7B (28 layers. hidden 2048); contrast pair:   Qwen3-0.6B (hidden 1024).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 412, 11 words, 4 clauses)  [Script] |
| --- | Minor | Both share the same tokenizer, vocabulary,   KV heads,  , and RoPE base . |
| --- | Minor | Both share the same tokenizer. vocabulary. KV heads. and RoPE base .. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 422, 24 words, 4 clauses)  [Script] |
| --- | Minor | The injection pipeline, cache construction, GQA head handling, and position accounting are therefore correct, and every subsequent gap is attributable to the translation itself. |
| --- | Minor | The injection pipeline. cache construction. GQA head handling. and position accounting are therefore correct. and every subsequent gap is attributable to the translation itself.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 452, 174 words, 24 clauses)  [Script] |
| --- | Minor | \small \begin{tabular}{lccc} \toprule Configuration & Acc / Gold / PPL & Gold CHG [95\% CI] \\ \midrule Identity control (self-kv), 1.7B & 0.500 / 0.503 / 21.2 &   [ ] \\ Student self-prefill, 1.7B & 0.500 / 0.503 / 25.4 & -- \\ Teacher (full prefill), 1.7B & 0.767 / 0.693 / -- & -- \\ Native average, 1.7B & 0.167 / 0.169 / 536082.5 &   [ ] \\ Ridge per-head, c=30, 1.7B & 0.200 / 0.207 / 88.5 &   [ ] \\ Affine per-head, c=30, 1.7B & 0.367 / 0.365 / 46.1 &   [ ] \\ Affine per-head, c=200, 1.7B & 0.254 / 0.258 / 56.4 &   [ ] \\ Affine per-layer, c=200, 1.7B & 0.206 / 0.222 / 81441.1 &   [ ] \\ Task-aware, c=200, 1.7B & 0.270 / 0.280 / 142.1 &   [ ] \\ RAT, c=30, 1.7B & 0.233 / 0.235 / 501.6 &   [ ] \\ RAT, c=200, 1.7B & 0.238 / 0.276 / 1008.2 &   [ ] \\ RAT no-anchor rank-32, c=30, 1.7B & 0.267 / 0.280 / 23.7 &   [ ] \\ Affine per-head, c=200, 0.6B & 0.302 / 0.283 / 71.8 &   [ ] \\ RAT, c=200, 0.6B & 0.206 / 0.232 / 790.0 &   [ ] \\ \bottomrule \end{tabular} |
| --- | Minor | \small \begin{tabular}{lccc} \toprule Configuration & Acc / Gold / PPL & Gold CHG [95\% CI] \\ \midrule Identity control (self-kv). 1.7B & 0.500 / 0.503 / 21.2 &   [ ] \\ Student self-prefill. 1.7B & 0.500 / 0.503 / 25.4 & -- \\ Teacher (full prefill). 1.7B & 0.767 / 0.693 / -- & -- \\ Native average. 1.7B & 0.167 / 0.169 / 536082.5 &   [ ] \\ Ridge per-head. c=30. 1.7B & 0.200 / 0.207 / 88.5 &   [ ] \\ Affine per-head. c=30. 1.7B & 0.367 / 0.365 / 46.1 &   [ ] \\ Affine per-head. c=200. 1.7B & 0.254 / 0.258 / 56.4 &   [ ] \\ Affine per-layer. c=200. 1.7B & 0.206 / 0.222 / 81441.1 &   [ ] \\ Task-aware. c=200. 1.7B & 0.270 / 0.280 / 142.1 &   [ ] \\ RAT. c=30. 1.7B & 0.233 / 0.235 / 501.6 &   [ ] \\ RAT. c=200. 1.7B & 0.238 / 0.276 / 1008.2 &   [ ] \\ RAT no-anchor rank-32. c=30. 1.7B & 0.267 / 0.280 / 23.7 &   [ ] \\ Affine per-head. c=200. 0.6B & 0.302 / 0.283 / 71.8 &   [ ] \\ RAT. c=200. 0.6B & 0.206 / 0.232 / 790.0 &   [ ] \\ \bottomrule \end{tabular}. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 476, 47 words, 4 clauses)  [Script] |
| --- | Minor | First, centering is the largest single gain: moving from per-head ridge ( ) to per-head affine ( ) halves the deficit, which confirms that the teacher/student value distributions differ by location and scale before they differ by content (the measured teacher/student norm ratios are   for keys and for values). |
| --- | Minor | First. centering is the largest single gain: moving from per-head ridge ( ) to per-head affine ( ) halves the deficit. which confirms that the teacher/student value distributions differ by location and scale before they differ by content (the measured teacher/student norm ratios are   for keys and for values).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 476, 30 words, 5 clauses)  [Script] |
| --- | Minor | Third, structure and function class change little: per-layer merging ( ), task-aware fine-tuning ( ), RAT (  to  ), and a per-head MLP mapper with two GELU layers (  at  , at  ) all sit inside the same band. |
| --- | Minor | Third. structure and function class change little: per-layer merging ( ). task-aware fine-tuning ( ). RAT (  to  ). and a per-head MLP mapper with two GELU layers (  at. at  ) all sit inside the same band.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 552, 24 words, 4 clauses)  [Script] |
| --- | Minor | The two dissociate, so perplexity cannot serve as evidence of translation quality, a practice that appears in the serving literature because perplexity is cheap. |
| --- | Minor | The two dissociate. so perplexity cannot serve as evidence of translation quality. a practice that appears in the serving literature because perplexity is cheap.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 573, 41 words, 5 clauses)  [Script] |
| --- | Minor | First, the pullback returns only the row-space component of  ; the component orthogonal to  's rows, which can dominate a  -dimensional state observed through a  -dimensional head, is invisible to the teacher's own cache and therefore unavailable to any translator, however nonlinear. |
| --- | Minor | First. the pullback returns only the row-space component of  ; the component orthogonal to  's rows. which can dominate a  -dimensional state observed through a  -dimensional head. is invisible to the teacher's own cache and therefore unavailable to any translator. however nonlinear.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 573, 43 words, 5 clauses)  [Script] |
| --- | Minor | Second, the teacher's advantage over the student lives in weights, not in caches: FFN layers carry the model's factual associations~\citep{ffnmemory2021}, and the capability gap between Qwen3-4B and 1.7B is a gap in these parameters, which a cache does not transport. |
| --- | Minor | Second. the teacher's advantage over the student lives in weights. not in caches: FFN layers carry the model's factual associations~\citep{ffnmemory2021}. and the capability gap between Qwen3-4B and 1.7B is a gap in these parameters. which a cache does not transport.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 36, 50 words, 8 clauses)  [Script] |
| --- | Minor | Across six mapper families (per-head ridge, affine, per-layer affine, task-aware, a residual-anchored translator, and a per-head MLP), a controlled calibration ladder on a fixed evaluation set, and 28 recorded GPU runs, no configuration makes the strong student exceed its own prefill: gold-probability change spans   to   against a self-kv control at  . |
| --- | Minor | Across six mapper families (per-head ridge. affine. per-layer affine. task-aware. a residual-anchored translator. and a per-head MLP). a controlled calibration ladder on a fixed evaluation set. and 28 recorded GPU runs. no configuration makes the strong student exceed its own prefill: gold-probability change spans   to   against a self-kv control at  .. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |

## Decision Signals

- **Committee Score**: 5.4/10
- **Editor Verdict**: Pass to Review
- **Reviewer Recommendation**: Major Revision
- **Issue Bundle**: 1 major / 3 moderate / 0 minor

## Revision Roadmap

### Priority 1 --- Must Address (Blocking)

- [ ] Abstract and conclusion claims need explicit evidence traceability ([LLM]; abstract)

### Priority 2 --- Strongly Recommended

- [ ] Cross-section numeric consistency should be reconciled ([LLM]; abstract)
- [ ] Comparison protocol should make fairness assumptions explicit ([LLM]; method)
- [ ] Novelty claim should be grounded against the closest prior work ([LLM]; related_work)
