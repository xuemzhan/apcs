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