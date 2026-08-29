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