## Audit Methodology
sec:method
### Identity Control
At evaluation time we capture the student's own cache over the context
(allowed in calibration and probe stages), feed it through the same mapper
interface, cache construction, position handling, and scoring path as a
translated cache, and score. Any deviation from the student's own prefill is
mechanics loss. This is the control that prior evaluations omit.
### Disjoint Splits, Unified Prompts, and Honest Counters
Mapper calibration and evaluation draws are disjoint samples of the same
distribution (HellaSwag and ARC-Challenge, shuffled then split). All methods
share one token-level prompt format, so the only difference between the
handoff and the student baseline is where the context representation comes
from. Zero-prefill counters read the actual cache length before and after the
scoring forward pass rather than trusting bookkeeping, and the scoring pass
uses an isolated cache because the deployed transformers version mutates a
cache even under use\_cache=False, which silently pollutes
co-located measurements.
### Mapper Ladder
We audit five families under one interface. Ridge (per-head): a
 ridge map per student layer and head with Gram aggregation
over variable-length calibration samples. Affine (per-head): the
same with mean centering and an intercept, fit in closed form over
concatenated calibration samples. Affine (per-layer): heads merged
into rows, eight times fewer parameters. Task-aware: ridge
initialization plus a bounded task fine-tune that mixes reconstruction with
gold-answer cross-entropy (weight ) under a diagonal
parameterization with a per-layer bounded gate. RAT
(residual-anchored translator): an architecture-derived map
where and are the two models' own value projections
at the aligned layer pair, is the pseudo-inverse pullback from a
head's value space to the residual stream, and aligns the two residual
streams through the shared 151,936-token embedding matrices, which both
Qwen3 models ship. A closed-form rank- correction fits the calibration
residual on top of the analytic core, a Hungarian matching on head covariance
profiles replaces the identity head correspondence, and the sink position
(position 0) is overwritten with the student's calibration statistics because
its translated value is a PPL catastrophe (Section~sec:ratresults).
Keys follow the same construction on . Every family is evaluated at
calibration budgets 30 and 200 with disjoint evaluation samples
( and respectively).
### Oracle Probes
The probes assume an oracle that no training can beat: they hand the student
its own cache and mix in teacher content at a controlled budget.
Fraction probe: for
 with the student's own cache.
Window probe: translated content in only the bottom, middle, or top
third of student layers, student content elsewhere. Because every
configuration keeps the student's own cache dominant or localized, any
improvement over would indicate exploitable teacher-cache
content; monotone degradation under the tested translators rules out
exploitable content in this regime. Probes are
flagged non-deployable (they require the student's own prefill) and exist
only as measurement instruments.
### Pairs, Data, and Protocol Discipline
Primary pair: Qwen3-4B (36 layers, hidden 2560) Qwen3-1.7B (28 layers,
hidden 2048); contrast pair: Qwen3-0.6B (hidden 1024). Both share the
same tokenizer, vocabulary, KV heads, , and RoPE base
. Every run records its git commit and protocol version in its
artifacts; 21 runs back the numbers below.