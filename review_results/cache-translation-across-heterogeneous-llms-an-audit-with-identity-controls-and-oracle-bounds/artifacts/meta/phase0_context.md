# Phase 0: Automated Audit Results

**File**: `/workspace/apcs/paper/cache_audit/arxiv/main.tex` | **Language**: en | **Mode**: quick-audit
**Generated**: 2026-08-29T19:24:00.370236

## Issue Summary (180 total)
- Major: 5
- Minor: 175

## Issues by Module

### BIB

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Check: /workspace/apcs/paper/cache_audit/arxiv/main.tex |
| 2 | — | Minor | P2 | PASS |
| 3 | — | Minor | P2 | entries: 0 |
| 4 | — | Minor | P2 | entries: 0 |

### CITATIONS

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | No citation stacking issues found. |

### DEAI

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Use --analyze for full analysis |

### EXPERIMENT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 473 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 2 | 414 | Minor | P2 | No ablation or component-level evidence is mentioned; verify that contribution attribution is covered. |
| 3 | 414 | Minor | P2 | No statistical significance, variance, or confidence information is mentioned. |
| 4 | 559 | Major | P1 | Discussion may lack depth: low ratio of explanatory/attribution language (0/23 lines). Add causal analysis explaining why results occur. |
| 5 | 559 | Major | P1 | Discussion may lack layered structure: it should cover at least two categories (mechanism, comparison with prior work, limitations/boundaries, implications/outlook) instead of only restating results. |
| 6 | 627 | Major | P1 | Conclusion lacks limitations or future work discussion. |
| 7 | 627 | Minor | P2 | Conclusion lacks implications or broader impact statement. |
| 8 | 627 | Minor | P2 | Conclusion lacks explicit summary of core findings. |

### FIGURES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | figures in /workspace/apcs/paper/cache_audit/arxiv/main.tex... |
| 2 | — | Minor | P2 | 3 figures. |
| 3 | — | Minor | P2 | Line 429: figures/fig2_mapper_landscape.pdf |
| 4 | — | Minor | P2 | Line 506: figures/fig3_oracle_probes.pdf |
| 5 | — | Minor | P2 | Line 532: figures/fig4_ppl_acc_decoupling.pdf |
| 6 | — | Minor | P2 | All figures passed check. |

### FORMAT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | ============================================================ |
| 2 | — | Minor | P2 | Format Check Report |
| 3 | — | Minor | P2 | ============================================================ |
| 4 | — | Minor | P2 | /workspace/apcs/paper/cache_audit/arxiv/main.tex |
| 5 | — | Minor | P2 | UNAVAILABLE |
| 6 | — | Minor | P2 | chktex not found. Install with: apt-get install chktex (Linux) or via TeX Live/MiKTeX |
| 7 | — | Minor | P2 | MODE] chktex not available |
| 8 | — | Minor | P2 | chktex for detailed format checking |

### GRAMMAR

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | [Script]: goal=grammar strength=minimal |
| 2 | — | Minor | P2 | No rule-based issues detected in selected scope. |

### LOGIC

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Major | P1 | Abstract, contribution claims, and conclusion may be misaligned. |
| 2 | — | Minor | P2 | conclusion missing result; abstract missing contribution claim; conclusion missing contribution response; conclusion missing result evidence. |
| 3 | — | Minor | P2 | Make sure all three sections consistently state the problem, method, key results, and contribution. |
| 4 | — | Minor | P2 | These sections should tell the same core story with different emphasis, not diverge. |

### PRESUBMISSION

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | [A1] Abstract five-element check is incomplete; missing background. |
| 2 | 36 | Minor | P2 | [G2] Long paragraph detected (257 words, 11 sentences); split or add a clearer topic sentence. |
| 3 | 138 | Minor | P2 | [G2] Long paragraph detected (394 words, 2 sentences); split or add a clearer topic sentence. |
| 4 | 214 | Minor | P2 | [G2] Long paragraph detected (198 words, 9 sentences); split or add a clearer topic sentence. |
| 5 | 365 | Minor | P2 | [G2] Long paragraph detected (239 words, 9 sentences); split or add a clearer topic sentence. |
| 6 | 562 | Minor | P2 | [G2] Long paragraph detected (223 words, 10 sentences); split or add a clearer topic sentence. |
| 7 | 376 | Minor | P2 | [L4] Numbered equation environment has no label for later reference. |

### REFERENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 102 | Minor | P2 | Reference before definition: \ref{fig:framework} at line 102 appears before label definition at line 191 |
| 2 | 198 | Minor | P2 | Reference before definition: \ref{tab:related} at line 198 appears before label definition at line 286 |

### SENTENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | [Script]: goal=grammar strength=minimal |
| 2 | — | Minor | P2 | SENTENCE (Line 67, 37 words, 6 clauses)  [Script] |
| 3 | — | Minor | P2 | Recent systems report encouraging results, including Mixture-of-Translators (MoT), which preserves 51.0\% closed-set QA accuracy at 7B 0.5B scale~\citep{mot2026}, Cache-to-Cache semantic communication~\citep{c2c2025}, latent-space communication~\citep{lsc2026}, and selective KV sharing~\citep{kvcomm2025}. |
| 4 | — | Minor | P2 | Recent systems report encouraging results. including Mixture-of-Translators (MoT). which preserves 51.0\% closed-set QA accuracy at 7B 0.5B scale~\citep{mot2026}. Cache-to-Cache semantic communication~\citep{c2c2025}. latent-space communication~\citep{lsc2026}. and selective KV sharing~\citep{kvcomm2025}.. |
| 5 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 6 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 7 | — | Minor | P2 | none |
| 8 | — | Minor | P2 | Check: NEEDS-LLM |
| 9 | — | Minor | P2 | Flags:    not-assessed |
| 10 | — | Minor | P2 | SENTENCE (Line 122, 31 words, 4 clauses)  [Script] |
| 11 | — | Minor | P2 | \begin{enumerate} \item \textbf{An audit framework.} Identity controls, disjoint calibration/evaluation splits, gold-probability metrics with bootstrap intervals and permutation tests, and oracle probes that bound exploitability without training anything (Section~ ). |
| 12 | — | Minor | P2 | \begin{enumerate} \item \textbf{An audit framework.} Identity controls. disjoint calibration/evaluation splits. gold-probability metrics with bootstrap intervals and permutation tests. and oracle probes that bound exploitability without training anything (Section~ ).. |
| 13 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 14 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 15 | — | Minor | P2 | none |
| 16 | — | Minor | P2 | Check: NEEDS-LLM |
| 17 | — | Minor | P2 | Flags:    not-assessed |
| 18 | — | Minor | P2 | SENTENCE (Line 122, 34 words, 4 clauses)  [Script] |
| 19 | — | Minor | P2 | \item \textbf{A bounded design space.} Five mapper families, a calibration ladder, component ablations, and a residual-anchored translator derived from the two models' projections and shared vocabulary, all under one protocol (Section~  and Section~ ). |
| 20 | — | Minor | P2 | \item \textbf{A bounded design space.} Five mapper families. a calibration ladder. component ablations. and a residual-anchored translator derived from the two models' projections and shared vocabulary. all under one protocol (Section~  and Section~ ).. |
| 21 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 22 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 23 | — | Minor | P2 | none |
| 24 | — | Minor | P2 | Check: NEEDS-LLM |
| 25 | — | Minor | P2 | Flags:    not-assessed |
| 26 | — | Minor | P2 | SENTENCE (Line 138, 369 words, 47 clauses)  [Script] |
| 27 | — | Minor | P2 | \begin{figure}[t] \centering \begin{tikzpicture}[ node distance=4mm and 6mm, box/.style={draw, rounded corners, align=center, inner sep=4pt, minimum width=34mm, font=\scriptsize}, qbox/.style={box, fill=blue!8}, hyp/.style={box, fill=orange!12, minimum width=44mm}, inst/.style={box, fill=green!10, minimum width=44mm}, verdict/.style={box, fill=gray!10, minimum width=44mm}, gate/.style={diamond, draw, aspect=2.2, inner sep=1.5pt, font=\tiny\bfseries}, arr/.style={-{Latex[length=2mm]}, thick}, lab/.style={font=\tiny\bfseries}, ] \node[box, fill=red!8] (mot) {\textbf{Claim under audit}\\ translated teacher cache\\ lifts a frozen student (MoT, C2C, LSC, KVComm)}; \node[qbox, below=5mm of mot] (q) {\textbf{Central question}\\ \emph{Does the teacher's answer-relevant}\\ \emph{capability survive KV-space}\\ \emph{translation under zero re-prefill?}}; \node[hyp, below=6mm of q] (h1) {\textbf{H1 Mechanics}: foreign cache is\\ consumed without loss}; \node[inst, below=4mm of h1] (i1) {Identity control: inject student's\\ \emph{own} cache through the full pipeline}; \node[hyp, below=5mm of i1] (h2) {\textbf{H2 Mapping}: a translator reaches\\ replacement level (no worse than self)}; \node[inst, below=4mm of h2] (i2) {Mapper ladder: ridge / affine / per-layer /\\ task-aware / RAT   calib.\ 30--200}; \node[hyp, below=5mm of i2] (h3) {\textbf{H3 Exploitability}: teacher cache\\ contains student-readable advantage}; \node[inst, below=4mm of h3] (i3) {Oracle probes: blend teacher content into\\ student cache at fraction   / layer windows}; \node[verdict, below=6mm of i3, minimum width=120mm] (v) {\textbf{Verdicts}: H1 \checkmark\ (logit cos.\  ) \quad H2 \textdagger\ (best  ; weak student  ) \quad H3   (monotone probe decline; PPL/acc.\ decoupling)}; \node[box, fill=blue!8, below=4mm of v, minimum width=120mm] (c) {\textbf{Consequence}: replacement is achievable for weak students; capability transfer via cache translation is bounded out\\ for frozen students; audit checklist required for future claims}; \draw[arr] (mot) -- (q); \draw[arr] (q) -- (h1); \draw[arr] (h1) -- node[lab, right] {test} (i1); \draw[arr] (i1) -- node[lab, right] {pass   proceed} (h2); \draw[arr] (h2) -- node[lab, right] {test} (i2); \draw[arr] (i2) -- node[lab, right] {fail   probe ceiling} (h3); \draw[arr] (h3) -- node[lab, right] {test} (i3); \draw[arr] (i3) -- (v); \draw[arr] (v) -- (c); \end{tikzpicture} \caption{The three-hypothesis audit as a decision pipeline. |
| 28 | — | Minor | P2 | \begin{figure}[t] \centering \begin{tikzpicture}[ node distance=4mm and 6mm. box/.style={draw. rounded corners. align=center. inner sep=4pt. minimum width=34mm. font=\scriptsize}. qbox/.style={box. fill=blue!8}. hyp/.style={box. fill=orange!12. minimum width=44mm}. inst/.style={box. fill=green!10. minimum width=44mm}. verdict/.style={box. fill=gray!10. minimum width=44mm}. gate/.style={diamond. draw. aspect=2.2. inner sep=1.5pt. font=\tiny\bfseries}. arr/.style={-{Latex[length=2mm]}. thick}. lab/.style={font=\tiny\bfseries}. ] \node[box. fill=red!8] (mot) {\textbf{Claim under audit}\\ translated teacher cache\\ lifts a frozen student (MoT. C2C. LSC. KVComm)}; \node[qbox. below=5mm of mot] (q) {\textbf{Central question}\\ \emph{Does the teacher's answer-relevant}\\ \emph{capability survive KV-space}\\ \emph{translation under zero re-prefill?}}; \node[hyp. below=6mm of q] (h1) {\textbf{H1 Mechanics}: foreign cache is\\ consumed without loss}; \node[inst. below=4mm of h1] (i1) {Identity control: inject student's\\ \emph{own} cache through the full pipeline}; \node[hyp. below=5mm of i1] (h2) {\textbf{H2 Mapping}: a translator reaches\\ replacement level (no worse than self)}; \node[inst. below=4mm of h2] (i2) {Mapper ladder: ridge / affine / per-layer /\\ task-aware / RAT   calib.\ 30--200}; \node[hyp. below=5mm of i2] (h3) {\textbf{H3 Exploitability}: teacher cache\\ contains student-readable advantage}; \node[inst. below=4mm of h3] (i3) {Oracle probes: blend teacher content into\\ student cache at fraction   / layer windows}; \node[verdict. below=6mm of i3. minimum width=120mm] (v) {\textbf{Verdicts}: H1 \checkmark\ (logit cos.\  ) \quad H2 \textdagger\ (best  ; weak student  ) \quad H3   (monotone probe decline; PPL/acc.\ decoupling)}; \node[box. fill=blue!8. below=4mm of v. minimum width=120mm] (c) {\textbf{Consequence}: replacement is achievable for weak students; capability transfer via cache translation is bounded out\\ for frozen students; audit checklist required for future claims}; \draw[arr] (mot) -- (q); \draw[arr] (q) -- (h1); \draw[arr] (h1) -- node[lab. right] {test} (i1); \draw[arr] (i1) -- node[lab. right] {pass   proceed} (h2); \draw[arr] (h2) -- node[lab. right] {test} (i2); \draw[arr] (i2) -- node[lab. right] {fail   probe ceiling} (h3); \draw[arr] (h3) -- node[lab. right] {test} (i3); \draw[arr] (i3) -- (v); \draw[arr] (v) -- (c); \end{tikzpicture} \caption{The three-hypothesis audit as a decision pipeline.. |
| 29 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 30 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 31 | — | Minor | P2 | none |
| 32 | — | Minor | P2 | Check: NEEDS-LLM |
| 33 | — | Minor | P2 | Flags:    not-assessed |
| 34 | — | Minor | P2 | SENTENCE (Line 347, 41 words, 5 clauses)  [Script] |
| 35 | — | Minor | P2 | \subsection{Identity Control} At evaluation time we capture the student's own cache over the context (allowed in calibration and probe stages), feed it through the same mapper interface, cache construction, position handling, and scoring path as a translated cache, and score. |
| 36 | — | Minor | P2 | \subsection{Identity Control} At evaluation time we capture the student's own cache over the context (allowed in calibration and probe stages). feed it through the same mapper interface. cache construction. position handling. and scoring path as a translated cache. and score.. |
| 37 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 38 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 39 | — | Minor | P2 | none |
| 40 | — | Minor | P2 | Check: NEEDS-LLM |
| 41 | — | Minor | P2 | Flags:    not-assessed |
| 42 | — | Minor | P2 | SENTENCE (Line 354, 45 words, 4 clauses)  [Script] |
| 43 | — | Minor | P2 | Zero-prefill counters read the actual cache length before and after the scoring forward pass rather than trusting bookkeeping, and the scoring pass uses an isolated cache because the deployed transformers version mutates a cache even under \texttt{use\_cache=False}, which silently pollutes co-located measurements. |
| 44 | — | Minor | P2 | Zero-prefill counters read the actual cache length before and after the scoring forward pass rather than trusting bookkeeping. and the scoring pass uses an isolated cache because the deployed transformers version mutates a cache even under \texttt{use\_cache=False}. which silently pollutes co-located measurements.. |
| 45 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 46 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 47 | — | Minor | P2 | none |
| 48 | — | Minor | P2 | Check: NEEDS-LLM |
| 49 | — | Minor | P2 | Flags:    not-assessed |
| 50 | — | Minor | P2 | SENTENCE (Line 365, 86 words, 12 clauses)  [Script] |
| 51 | — | Minor | P2 | \textbf{RAT (residual-anchored translator)}: an architecture-derived map \begin{equation} W^{(s,h)} \;=\; \bigl[\,\pi(W^{S,h}_V)\;R\;\pi(W^{T,h}_V)\,\bigr]^{\!\top}, \qquad R \;=\; \arg\min_R \; \|E_T R - E_S\|_F^2 + \lambda \|R\|_F^2, \end{equation} where   and   are the two models' own value projections at the aligned layer pair,   is the pseudo-inverse pullback from a head's value space to the residual stream, and   aligns the two residual streams through the shared 151{,}936-token embedding matrices, which both Qwen3 models ship. |
| 52 | — | Minor | P2 | \textbf{RAT (residual-anchored translator)}: an architecture-derived map \begin{equation} W^{(s. h)} \;=\; \bigl[\. \pi(W^{S. h}_V)\;R\;\pi(W^{T. h}_V)\. \bigr]^{\!\top}. \qquad R \;=\; \arg\min_R \; \|E_T R - E_S\|_F^2 + \lambda \|R\|_F^2. \end{equation} where   and   are the two models' own value projections at the aligned layer pair. is the pseudo-inverse pullback from a head's value space to the residual stream. and   aligns the two residual streams through the shared 151{. }936-token embedding matrices. which both Qwen3 models ship.. |
| 53 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 54 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 55 | — | Minor | P2 | none |
| 56 | — | Minor | P2 | Check: NEEDS-LLM |
| 57 | — | Minor | P2 | Flags:    not-assessed |
| 58 | — | Minor | P2 | SENTENCE (Line 407, 25 words, 4 clauses)  [Script] |
| 59 | — | Minor | P2 | \subsection{Pairs, Data, and Protocol Discipline} Primary pair: Qwen3-4B (36 layers, hidden 2560)   Qwen3-1.7B (28 layers, hidden 2048); contrast pair:   Qwen3-0.6B (hidden 1024). |
| 60 | — | Minor | P2 | \subsection{Pairs. Data. and Protocol Discipline} Primary pair: Qwen3-4B (36 layers. hidden 2560)   Qwen3-1.7B (28 layers. hidden 2048); contrast pair:   Qwen3-0.6B (hidden 1024).. |
| 61 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 62 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 63 | — | Minor | P2 | none |
| 64 | — | Minor | P2 | Check: NEEDS-LLM |
| 65 | — | Minor | P2 | Flags:    not-assessed |
| 66 | — | Minor | P2 | SENTENCE (Line 407, 11 words, 4 clauses)  [Script] |
| 67 | — | Minor | P2 | Both share the same tokenizer, vocabulary,   KV heads,  , and RoPE base . |
| 68 | — | Minor | P2 | Both share the same tokenizer. vocabulary. KV heads. and RoPE base .. |
| 69 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 70 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 71 | — | Minor | P2 | none |
| 72 | — | Minor | P2 | Check: NEEDS-LLM |
| 73 | — | Minor | P2 | Flags:    not-assessed |
| 74 | — | Minor | P2 | SENTENCE (Line 417, 24 words, 4 clauses)  [Script] |
| 75 | — | Minor | P2 | The injection pipeline, cache construction, GQA head handling, and position accounting are therefore correct, and every subsequent gap is attributable to the translation itself. |
| 76 | — | Minor | P2 | The injection pipeline. cache construction. GQA head handling. and position accounting are therefore correct. and every subsequent gap is attributable to the translation itself.. |
| 77 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 78 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 79 | — | Minor | P2 | none |
| 80 | — | Minor | P2 | Check: NEEDS-LLM |
| 81 | — | Minor | P2 | Flags:    not-assessed |
| 82 | — | Minor | P2 | SENTENCE (Line 447, 174 words, 24 clauses)  [Script] |
| 83 | — | Minor | P2 | \small \begin{tabular}{lccc} \toprule Configuration & Acc / Gold / PPL & Gold CHG [95\% CI] \\ \midrule Identity control (self-kv), 1.7B & 0.500 / 0.503 / 21.2 &   [ ] \\ Student self-prefill, 1.7B & 0.500 / 0.503 / 25.4 & -- \\ Teacher (full prefill), 1.7B & 0.767 / 0.693 / -- & -- \\ Native average, 1.7B & 0.167 / 0.169 / 536082.5 &   [ ] \\ Ridge per-head, c=30, 1.7B & 0.200 / 0.207 / 88.5 &   [ ] \\ Affine per-head, c=30, 1.7B & 0.367 / 0.365 / 46.1 &   [ ] \\ Affine per-head, c=200, 1.7B & 0.254 / 0.258 / 56.4 &   [ ] \\ Affine per-layer, c=200, 1.7B & 0.206 / 0.222 / 81441.1 &   [ ] \\ Task-aware, c=200, 1.7B & 0.270 / 0.280 / 142.1 &   [ ] \\ RAT, c=30, 1.7B & 0.233 / 0.235 / 501.6 &   [ ] \\ RAT, c=200, 1.7B & 0.238 / 0.276 / 1008.2 &   [ ] \\ RAT no-anchor rank-32, c=30, 1.7B & 0.267 / 0.280 / 23.7 &   [ ] \\ Affine per-head, c=200, 0.6B & 0.302 / 0.283 / 71.8 &   [ ] \\ RAT, c=200, 0.6B & 0.206 / 0.232 / 790.0 &   [ ] \\ \bottomrule \end{tabular} |
| 84 | — | Minor | P2 | \small \begin{tabular}{lccc} \toprule Configuration & Acc / Gold / PPL & Gold CHG [95\% CI] \\ \midrule Identity control (self-kv). 1.7B & 0.500 / 0.503 / 21.2 &   [ ] \\ Student self-prefill. 1.7B & 0.500 / 0.503 / 25.4 & -- \\ Teacher (full prefill). 1.7B & 0.767 / 0.693 / -- & -- \\ Native average. 1.7B & 0.167 / 0.169 / 536082.5 &   [ ] \\ Ridge per-head. c=30. 1.7B & 0.200 / 0.207 / 88.5 &   [ ] \\ Affine per-head. c=30. 1.7B & 0.367 / 0.365 / 46.1 &   [ ] \\ Affine per-head. c=200. 1.7B & 0.254 / 0.258 / 56.4 &   [ ] \\ Affine per-layer. c=200. 1.7B & 0.206 / 0.222 / 81441.1 &   [ ] \\ Task-aware. c=200. 1.7B & 0.270 / 0.280 / 142.1 &   [ ] \\ RAT. c=30. 1.7B & 0.233 / 0.235 / 501.6 &   [ ] \\ RAT. c=200. 1.7B & 0.238 / 0.276 / 1008.2 &   [ ] \\ RAT no-anchor rank-32. c=30. 1.7B & 0.267 / 0.280 / 23.7 &   [ ] \\ Affine per-head. c=200. 0.6B & 0.302 / 0.283 / 71.8 &   [ ] \\ RAT. c=200. 0.6B & 0.206 / 0.232 / 790.0 &   [ ] \\ \bottomrule \end{tabular}. |
| 85 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 86 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 87 | — | Minor | P2 | none |
| 88 | — | Minor | P2 | Check: NEEDS-LLM |
| 89 | — | Minor | P2 | Flags:    not-assessed |
| 90 | — | Minor | P2 | SENTENCE (Line 471, 47 words, 4 clauses)  [Script] |
| 91 | — | Minor | P2 | First, centering is the largest single gain: moving from per-head ridge ( ) to per-head affine ( ) halves the deficit, which confirms that the teacher/student value distributions differ by location and scale before they differ by content (the measured teacher/student norm ratios are   for keys and for values). |
| 92 | — | Minor | P2 | First. centering is the largest single gain: moving from per-head ridge ( ) to per-head affine ( ) halves the deficit. which confirms that the teacher/student value distributions differ by location and scale before they differ by content (the measured teacher/student norm ratios are   for keys and for values).. |
| 93 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 94 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 95 | — | Minor | P2 | none |
| 96 | — | Minor | P2 | Check: NEEDS-LLM |
| 97 | — | Minor | P2 | Flags:    not-assessed |
| 98 | — | Minor | P2 | SENTENCE (Line 471, 17 words, 4 clauses)  [Script] |
| 99 | — | Minor | P2 | Third, structure changes little: per-layer merging ( ), task-aware fine-tuning ( ), and RAT (  at  , at  ) sit inside the same band. |
| 100 | — | Minor | P2 | Third. structure changes little: per-layer merging ( ). task-aware fine-tuning ( ). and RAT (  at. at  ) sit inside the same band.. |
| 101 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 102 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 103 | — | Minor | P2 | none |
| 104 | — | Minor | P2 | Check: NEEDS-LLM |
| 105 | — | Minor | P2 | Flags:    not-assessed |
| 106 | — | Minor | P2 | SENTENCE (Line 541, 24 words, 4 clauses)  [Script] |
| 107 | — | Minor | P2 | The two dissociate, so perplexity cannot serve as evidence of translation quality, a practice that appears in the serving literature because perplexity is cheap. |
| 108 | — | Minor | P2 | The two dissociate. so perplexity cannot serve as evidence of translation quality. a practice that appears in the serving literature because perplexity is cheap.. |
| 109 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 110 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 111 | — | Minor | P2 | none |
| 112 | — | Minor | P2 | Check: NEEDS-LLM |
| 113 | — | Minor | P2 | Flags:    not-assessed |
| 114 | — | Minor | P2 | SENTENCE (Line 562, 41 words, 5 clauses)  [Script] |
| 115 | — | Minor | P2 | First, the pullback returns only the row-space component of  ; the component orthogonal to  's rows, which can dominate a  -dimensional state observed through a  -dimensional head, is invisible to the teacher's own cache and therefore unavailable to any translator, however nonlinear. |
| 116 | — | Minor | P2 | First. the pullback returns only the row-space component of  ; the component orthogonal to  's rows. which can dominate a  -dimensional state observed through a  -dimensional head. is invisible to the teacher's own cache and therefore unavailable to any translator. however nonlinear.. |
| 117 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 118 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 119 | — | Minor | P2 | none |
| 120 | — | Minor | P2 | Check: NEEDS-LLM |
| 121 | — | Minor | P2 | Flags:    not-assessed |
| 122 | — | Minor | P2 | SENTENCE (Line 562, 43 words, 5 clauses)  [Script] |
| 123 | — | Minor | P2 | Second, the teacher's advantage over the student lives in weights, not in caches: FFN layers carry the model's factual associations~\citep{ffnmemory2021}, and the capability gap between Qwen3-4B and 1.7B is a gap in these parameters, which a cache does not transport. |
| 124 | — | Minor | P2 | Second. the teacher's advantage over the student lives in weights. not in caches: FFN layers carry the model's factual associations~\citep{ffnmemory2021}. and the capability gap between Qwen3-4B and 1.7B is a gap in these parameters. which a cache does not transport.. |
| 125 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 126 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 127 | — | Minor | P2 | none |
| 128 | — | Minor | P2 | Check: NEEDS-LLM |
| 129 | — | Minor | P2 | Flags:    not-assessed |
| 130 | — | Minor | P2 | SENTENCE (Line 36, 49 words, 7 clauses)  [Script] |
| 131 | — | Minor | P2 | Across five mapper families (per-head ridge, affine, per-layer affine, task-aware, and a residual-anchored translator built from the two models' own projections), a 200-example calibration ladder, and 21 recorded GPU runs, no configuration makes the strong student exceed its own prefill: gold-probability change spans   to   against a self-kv control at  . |
| 132 | — | Minor | P2 | Across five mapper families (per-head ridge. affine. per-layer affine. task-aware. and a residual-anchored translator built from the two models' own projections). a 200-example calibration ladder. and 21 recorded GPU runs. no configuration makes the strong student exceed its own prefill: gold-probability change spans   to   against a self-kv control at  .. |
| 133 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 134 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 135 | — | Minor | P2 | none |
| 136 | — | Minor | P2 | Check: NEEDS-LLM |
| 137 | — | Minor | P2 | Flags:    not-assessed |

## Pre-Submission Checklist

- [x] No placeholder text (TODO, FIXME, XXX)
- [x] All figures referenced in text
- [x] All tables referenced in text
- [ ] Anonymous submission (blind review check) — Author information detected — verify if blind review required
- [x] Consistent math notation
- [ ] Acronyms defined on first use — Potentially undefined: ['PPL', 'QA', 'LSC', 'ARC', 'RAG']
