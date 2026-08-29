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