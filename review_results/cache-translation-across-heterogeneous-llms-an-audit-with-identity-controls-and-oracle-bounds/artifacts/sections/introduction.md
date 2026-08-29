## Introduction
sec:intro
Multi-model serving systems want to reuse the teacher's work. A large teacher
prefills a long context once; a smaller student should inherit that state and
answer at teacher quality without re-reading the context. Cache translation
formalizes this idea: learn a map from the teacher's cache space into the
student's. Recent systems report encouraging results, including
Mixture-of-Translators (MoT), which preserves 51.0\% closed-set QA accuracy at
7B0.5B scale~mot2026, Cache-to-Cache semantic
communication~c2c2025, latent-space
communication~lsc2026, and selective KV
sharing~kvcomm2025.
These reports conflate three questions that can fail independently. Does the
injection pipeline consume a foreign cache correctly (mechanics)? Does
the map replace the student's own cache at replacement level (mapping)?
Does the teacher's cache carry student-readable content that lifts the student
above its own prefill (exploitability)? Prior evaluations score the
translated end-to-end pipeline against the student, so a positive delta could
come from any of the three; a null result cannot be attributed. The line also
lacks an identity control, a probe of the exploitability ceiling, and any
report of what happens to accuracy when perplexity is repaired.
We evaluate cache translation as an audit with three gated hypotheses.
H1 (mechanics) asks whether a foreign cache is consumed without loss;
its instrument is an identity control that injects the student's own cache
through the full pipeline. H2 (mapping) asks whether any translator
reaches replacement level; its instrument is a ladder of five mapper families
under a disjoint calibration ladder of 30 to 200 examples, including a
residual-anchored translator (RAT) that we derive from the two models' own
projections and a shared vocabulary alignment. H3 (exploitability)
asks whether the teacher's cache contains any student-readable advantage at
all; its instrument is an oracle probe that mixes teacher-cache content into
the student's own cache at controlled fractions and layer windows, which upper
bounds every translator that could be trained. Figure~fig:framework
shows the gates and the instruments.
The audit returns three verdicts on Qwen3 pairs. H1 passes: the identity
control reproduces the student's own prefill with per-sample logit cosine
 and a gold-probability change of 
( CI ), so mechanics is not the failure mode. H2
fails for the strong student: across the mapper ladder, the gold-probability
change over the student's own prefill spans to , and the best
configuration reaches against a student baseline of ; the weak
0.6B student reaches parity (, CI ). H3 returns a
principled negative: with student cache and only teacher content,
accuracy already drops from to , the decline is monotone in the
teacher fraction, and replacing the top third of layers is the only harmless
configuration. A translator that restores near-native fluency (PPL 23.7
versus 21.2 for the identity control) still leaves accuracy at 0.267. The
teacher's advantage does not ride in its cache.
Our contributions are threefold.
enumerate
 An audit framework. Identity controls, disjoint
calibration/evaluation splits, gold-probability metrics with bootstrap
intervals and permutation tests, and oracle probes that bound exploitability
without training anything (Section~sec:method).
 A bounded design space. Five mapper families, a calibration
ladder, component ablations, and a residual-anchored translator derived from
the two models' projections and shared vocabulary, all under one protocol
(Section~sec:method and Section~sec:results).
 A verdict with consequences. Mechanics passes, replacement
fails for the strong student and holds for the weak one, and exploitability
is bounded at zero; perplexity cannot substitute for accuracy as evidence
(Section~sec:results and Section~sec:implications).
enumerate