# Paper Summary: Cache Translation Across Heterogeneous LLMs:\\ An Audit with Identity Controls and Oracle Bounds

## Research Question
- Cache translation maps the key-value (KV) cache that a large teacher model forms over a context into the cache space of a smaller student model, so that the student answers without re-prefilling the context

## Core Thesis
- We conclude that the teacher's answer-relevant advantage does not survive KV-space translation under zero re-prefill: it lives in the teacher's weights, not in its cache.

## Headline Claims
- We conclude that the teacher's answer-relevant advantage does not survive KV-space translation under zero re-prefill: it lives in the teacher's weights, not in its cache.

## Section Map
- abstract (36-61): 251 words
- introduction (63-193): 925 words
- method (310-343): 273 words
- method_2 (344-413): 545 words
- result (414-558): 1089 words
- discussion (559-584): 243 words
- conclusion (627-643): 116 words

## Closure Targets
- No closure target was extracted automatically.