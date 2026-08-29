## The Three-Hypothesis Audit Framework
sec:framework
### Setup
Let a teacher prefill a context and form a cache
 with layers, KV
heads, and head dimension . A translator maps it to
 in the space of a student with layers. The
student then answers a query with only the query tokens as input,
, under strict zero re-prefill: the
context never enters the student's forward pass. Three references anchor the
audit. Student self-prefill is the baseline the
handoff must not degrade. Identity injection feeds itself
through the injection pipeline; it isolates mechanics from mapping.
Teacher full is the capability upper bound. We
measure answer quality by the gold-letter probability , accuracy, and
the change , with
95\% bootstrap intervals and sign-flip permutation tests; perplexity over the
scoring suffix is recorded but never accepted as evidence of capability
(Section~sec:decoupling).
### Hypotheses, Instruments, Gates
Figure~fig:framework states the three hypotheses with their instruments
and gates. H1: the identity injection matches the student's own
prefill per sample (gate: mean logit cosine within float tolerance).
H2: some reaches replacement level (gate: with
CI excluding zero, across the mapper ladder at calibration budgets up to 200
examples). H3: the teacher's cache contains student-readable
advantage (gate: some oracle probe configuration with teacher content
improves over the student's own cache). H3's probe is decisive because it
does not depend on training a translator; it directly measures what a frozen
student can extract from teacher-cache content under the most favorable
mixing.