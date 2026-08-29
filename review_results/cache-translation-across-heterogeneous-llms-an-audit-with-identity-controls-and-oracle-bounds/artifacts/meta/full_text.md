article
=1
geometry
inputenc
fontenc
lmodern 
amsmath,amssymb,amsfonts
graphicx
booktabs
multirow
xcolor
tikz
shapes.geometric,arrows.meta,positioning,fit,calc
natbib
microtype
hyperref
KV
CHG
p_gold
Cache Translation Across Heterogeneous LLMs:\\
An Audit with Identity Controls and Oracle Bounds
 Anonymous\\
 Draft, compiled \\
 Code and all run artifacts: apcs repository (21 recorded GPU runs)
document
abstract
Cache translation maps the key-value (KV) cache that a large teacher model
forms over a context into the cache space of a smaller student model, so that
the student answers without re-prefilling the context. A growing line of work
reports quality-preserving translation across heterogeneous models. We audit
this claim on Qwen3 pairs (4B1.7B and 4B0.6B) with a three-hypothesis
framework and two instruments that prior evaluations omit. First, an
identity control injects the student's own cache through the full
translation-and-injection pipeline: it reproduces the student's own prefill
exactly (per-sample logit cosine ), which isolates mechanics from
mapping. Second, oracle probes mix teacher-cache content into the
student's own cache and bound what any translator could extract. Across six
mapper families (per-head ridge, affine, per-layer affine, task-aware, a
residual-anchored translator, and a per-head MLP), a controlled calibration
ladder on a fixed evaluation set, and 28 recorded GPU runs, no configuration
makes the strong student exceed its own prefill: gold-probability change
spans to against a self-kv control at . Calibration
budget is inert (CHG at vs at on the
same evaluation set), and nonlinearity does not help. The weak
student reaches parity (, CI ). Oracle probes show a
monotone decline as teacher content replaces student content in early and
mid layers, robust to translator quality, and a perplexity--accuracy decoupling: a translator that restores
near-native fluency (PPL 23.7 versus 21.2) still leaves accuracy at 0.267
versus 0.500. We conclude that the teacher's answer-relevant advantage does
not survive KV-space translation under zero re-prefill: it lives in the
teacher's weights, not in its cache. We release the audit protocol as a
required checklist for cache-translation claims.
abstract
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
## A Survey of Cache-Level Communication Between Models
sec:survey
We organize prior work into four lines and close each with the assumption it
leaves untested. Table~tab:related summarizes the comparison.
### KV Reuse Within a Single Model
Prompt caching, cache-augmented generation, and streaming attention reuse a
model's own cache across requests. Streaming attention identifies attention
sinks at early positions whose KV carries outsized attention
mass~streamingsink2023. CacheGen compresses and streams a KV cache
across a network~cachegen2024; CacheBlend fuses several cached
fragments with selective attention recomputation for
RAG~cacheblend2025; KVSharer shares caches across layers that behave
dissimilarly~kvsharer2024. These systems assume a single model on
both sides of the cache. The reuse question is therefore about storage and
attention cost, not about representation compatibility. Our audit inherits
their serving motivation but moves the mismatch to the model axis, where the
compatibility assumption breaks.
### Cross-Model Cache Translation
The line closest to our audit translates caches between models. C2C proposes
direct cache-to-cache semantic communication with a depth-ratio layer
mapping~c2c2025. LSC aligns caches through a shared latent space with
a cross-attention translator~lsc2026. MoT combines gated translator
modules over depth-ratio channel windows with a Context Correction Loss that
aligns a replayed target trajectory with the native one, and reports the
strongest numbers to date: 51.0\% average closed-set QA accuracy at
Qwen2.5-7B0.5B and 96.3\% retention in long-context cache-augmented
generation~mot2026. KVComm shares attention-ranked KV
layers~kvcomm2025; Interlat communicates last-hidden-state
messages~interlat2025; HCache restores caches from hidden-state
translations~hcache2025. Two properties of this line matter for our
audit. First, its evaluations score the translated pipeline against the
student baseline, which conflates mechanics, mapping, and exploitability.
Second, its strongest variants do not operate under strict zero re-prefill:
MoT reconstructs the target-side cache by replaying the context through the
target model with source-guided sparse attention, so the reported quality
includes a correction pass that a cold-start deployment would not get. Our
audit holds the zero-re-prefill constraint fixed and asks what a translator
alone can deliver.
### Cross-Model Representation Alignment
A separate line asks whether two trained networks' representations can be
aligned at all. Model stitching connects frozen networks with a learned
linear layer and shows that stitching quality varies sharply across training
runs and layers~stitching2021. Relative representations replace
absolute coordinates with anchor-based similarity spaces and enable zero-shot
communication between encoders~relrep2022. The Platonic Representation
Hypothesis argues that representations converge across models up to linear
transform~platonic2024, with refinements and proofs
following~aristotelian2026,perfectplatonic2025. vec2vec translates
embeddings between models without paired data by exploiting this shared
geometry~vec2vec2025. These results motivate the hypothesis that a
linear map between cache spaces should exist. Our results qualify the
conclusion for caches: the convergence evidence comes from pooled embedding
spaces, while a cache is a per-layer, per-head, position-indexed state whose
student-relevant content is not determined by its teacher counterpart
(Section~sec:why).
### Frozen-Model Controllability
Work on steering shows that a frozen model's behavior can be moved by
injecting activations: task vectors compress in-context learning into a
single mid-layer activation~taskvectors2023; activation steering adds
behavioral directions to the residual stream; gist tokens compress prompts
into a few virtual tokens~gist2023. The tuned lens reads intermediate
layers through per-layer affine probes~tunedlens2023, and the FFN-as-
memory view assigns the factual capacity of a model to its feed-forward
weights~ffnmemory2021. This line supports the intuition that a frozen
student can absorb injected information, which makes the H3 null result
informative rather than tautological: the failure is not that frozen students
cannot be steered, but that the teacher's cache does not carry the steering
signal in student-readable form. It also suggests where transferable signal
might live: in residual-stream directions and mid-layer task vectors, not in
KV entries.
### What the Survey Does Not Establish
Across the four lines, no work (i) separates mechanics from mapping from
exploitability, (ii) injects the student's own cache as an identity control,
(iii) reports accuracy at matched perplexity, or (iv) bounds exploitability
with an oracle that requires no training. The claims of the translation line
therefore rest on pipelines whose components have never been isolated. The
audit below supplies the missing instruments.
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
H2: some reaches replacement level (gate: gold CHG
 with the CI lower bound above a small tolerance ; a
stricter gain gate additionally requires the CI lower bound ).
Replacement asks only that the student is not degraded; gain asks that the
teacher's cache provides a measurable lift. H3: the teacher's cache contains student-readable
advantage (gate: some oracle probe configuration with teacher content
improves over the student's own cache). H3's probe is decisive because it
does not depend on training a translator; it directly measures what a frozen
student can extract from teacher-cache content under the most favorable
mixing.
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
## Results
sec:results
### H1 Passes: Mechanics Is Not the Failure Mode
The identity control reproduces the student's own prefill exactly: per-sample
logit cosine , gold CHG , . The
injection pipeline, cache construction, GQA head handling, and position
accounting are therefore correct, and every subsequent gap is attributable to
the translation itself. The same control verified the disk-persistence path:
a cold-start evaluation that loads a persisted cache and mapper parameters
reproduces the offline run bit-for-bit.
### H2 Fails for the Strong Student and Holds for the Weak One
Figure~fig:landscape and Table~tab:main give the landscape.
Three regularities hold across all families. First, centering is the largest
single gain: moving from per-head ridge () to per-head affine
() halves the deficit, which confirms that the teacher/student value
distributions differ by location and scale before they differ by content
(the measured teacher/student norm ratios are for keys and
 for values). Second, the calibration budget is inert. Under a controlled comparison
that fixes the evaluation set () and varies only the calibration
budget, affine mapping yields CHG at and at
: a difference of . The mapper has converged; more
calibration data cannot close a gap that is set by the information content
of the translated cache, not by estimation variance. Third, structure and function class change little: per-layer merging
(), task-aware fine-tuning (), RAT ( to ),
and a per-head MLP mapper with two GELU layers ( at ,
 at ) all sit inside the same band. Nonlinearity does not
help; the bottleneck is not the function class. The strong student never
approaches its own prefill (); the best mapper reaches . The
weak 0.6B student tells the opposite story: affine at gives
 , statistically indistinguishable from its own
prefill, which is replacement-level service for a weak student.
### RAT Component Ablations Separate Fluency From Capability
sec:ratresults
RAT ablations attribute the failure inside the architecture-anchored design.
Removing the sink override explodes suffix perplexity to :
the translated position-0 entry, which carries outsized attention mass, is a
catastrophe unless overwritten with the student's own statistics. Removing
the analytic core and raising the correction rank to 32 repairs
fluency: PPL drops to , within of the identity control. Accuracy
stays at . The analytic core's min-norm pullback is harmful on real
models, and more importantly, fluency recovery does not touch the answer
deficit. The cache can be made to look right to the language-modeling
objective without becoming right for the task.
### H3: Oracle Probes Bound Exploitability at Zero
The probes answer H3 without training anything
(Figure~fig:probes). With student cache and teacher
content, gold probability already falls from to ; the decay
is monotone through . Window injection localizes the damage:
replacing the top third of layers is indistinguishable from the student
( versus ), while the middle and bottom thirds are strongly
harmful (, ). The reading is architectural. Answer-relevant
routing runs through early and mid layers, exactly where the teacher's cache
is least compatible; the top layers, where the translated content is least
damaging, are not where the answer is decided.
The probes were run with two translators: the RAT mapper reported above and
the affine mapper (the best in the ladder, gold ). Both show the same
monotone decline. With affine at , gold CHG is 
(CI ); at , (CI ).
The verdict is robust to translator quality: even the best available mapper
produces content that the frozen student cannot exploit.
### Perplexity and Accuracy Decouple
sec:decoupling
Figure~fig:decoupling plots perplexity against accuracy for every
method and run. The identity control and the student occupy the top-left
regime. Translated caches populate a broad band, including a point with
near-native perplexity and a 23-point accuracy deficit. Native injection
sits at extreme perplexity () with low accuracy. Perplexity
measures whether the cache permits fluent continuation; accuracy measures
whether the cache supports the task. The two dissociate, so perplexity
cannot serve as evidence of translation quality, a practice that appears in
the serving literature because perplexity is cheap.
### Channel Asymmetry
Channel ablations at rank value-only () above key-only
() above both () on accuracy, while perplexity orders them in
reverse ( versus versus ). Mapping keys repairs
fluency and damages routing: the mapped keys steer attention to plausible
but task-irrelevant positions. We report the asymmetry as a finding and
leave the mechanism open.
## Why Translation Fails: An Architectural Analysis
sec:why
The audit's verdicts fit the architecture. A cache entry is a linear
projection of a residual-stream state . Translation must recover
 from , move it across the residual-space gap ( to ), and
re-project. Two facts bound the recovery. First, the pullback
 returns only the row-space component of ; the component
orthogonal to 's rows, which can dominate a -dimensional state
observed through a -dimensional head, is invisible to the teacher's own
cache and therefore unavailable to any translator, however nonlinear. The
student's corresponding value depends on that invisible
component, so is not a function of : the mapping task is
intrinsically underdetermined, and a learned translator can at best track the
conditional mean. Second, the teacher's advantage over the student lives in
weights, not in caches: FFN layers carry the model's factual
associations~ffnmemory2021, and the capability gap between Qwen3-4B
and 1.7B is a gap in these parameters, which a cache does not transport. The
oracle probes make the consequence concrete: even with the student's own
cache holding of the mass, the teacher's contribution degrades
the answer. MoT's own design corroborates the analysis: its reported quality
includes a target-side replay with a correction loss, which is exactly the
upper-layer correction that strict zero re-prefill denies. Under a cold
start, the translator's output is the whole story, and the whole story does
not contain the teacher's capability.
## Implications for Cache-Translation Research
sec:implications
For claims. Any cache-translation evaluation should ship an identity
control, a gold-probability metric with intervals, and an oracle probe. A
pipeline delta over a student baseline is not evidence of translation; a
perplexity gain is not evidence of capability. Our instruments are one
evaluation run each and compose with any serving stack.
For deployment. Two regimes survive the audit. For weak students
relative to the teacher (here 0.6B under 4B), translated injection reaches
parity with self-prefill, so cached handoff saves the student's prefill at no
measured quality cost. Top-layer injection is harmless; if a system must
spend translation capacity, the top layers are where it can be spent, though
our probes also show that spending it there buys nothing. For strong
students, cache translation is currently a net negative, and the honest
alternative is the text channel: the teacher's summary of the context scores
 against the student's full-text baseline, so compression
loses less than translation loses on the strong student.
For the direction. The survivable versions of the idea relocate
work from the cache to weights or to the replay: allow a target-side
correction pass as MoT does, train student-side adapters, or translate in a
shared latent space anchored by task vectors rather than per-entry caches.
Each option trades away part of the cold-start guarantee, and each should be
re-audited with the identity control before its delta is attributed to
translation.
## Limitations
Our audit covers one model family (Qwen3), two pairs, and two multiple-choice
tasks with -- evaluation samples per run; the CI widths reflect
that. The oracle probe bounds exploitability under linear blending and
three-layer windows; a nonlinear exploit path outside this family would not
be detected, though it would also need to survive the same measurement
protocol. RAT's analytic core assumes value projections are informative
pullbacks of the residual stream; the synthetic validation confirms the
mathematics, and the real-model failure is itself an audit finding. We did
not re-implement MoT's full replay pipeline; our claims concern the
zero-re-prefill regime that its predecessors and our deployment target
share. The mapper-ladder results use a single seed per configuration; a
three-seed sweep of the best mapper (affine, ) confirms the negative
sign across seeds 42, 43, and 44 (, , ), with seed 42
being the most favorable case.
## Conclusion
Cache translation promises to move a teacher's work into a smaller model. The
audit separates that promise into three testable parts. The mechanism works:
an identity control reproduces the student's own prefill exactly. The map
does not: five mapper families, a calibration ladder, and an
architecture-anchored translator all leave the strong student below its own
prefill, and the weak student at parity. The reason is not the map: oracle
probes bound the student-readable advantage of the teacher's cache at zero,
and a near-native-perplexity cache still fails the task. The teacher's
capability is in its weights. We release the audit framework so that the
next cache-translation claim can be checked in one run.
plainnat
references
## Reproducibility
All numbers come from the apcs repository's recorded runs
(reports/runs/*/inject-eval/), each carrying its git commit,
protocol version, configuration, and per-sample scores. The audit protocol
is implemented in apcs.inference.evaluator with the flags
calib\_eval\_split, calib\_shuffle, probe\_mode,
and native\_rescale; mapper families live in apcs.mapper
(math, joint, task\_aware, rat);
persistence and the cold-start online phase in
apcs.inference.kv\_store. Unit tests cover the audit counters, the
store round-trip, and the mapper interface.
## Additional Measurements
The teacher/student KV norm ratios are (keys) and 
(values), measured over 30 calibration contexts per layer. Norm rescaling of
the native cache repairs neither its perplexity () nor its
accuracy, ruling out scale as the primary failure. The unified prompt
renders choices once; the earlier protocol rendered them twice for the
student baseline, which inflated the confidence-based deltas reported in our
own preliminary sweeps and motivated the unified format in the audit.
document