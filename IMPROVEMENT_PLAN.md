# APCS Improvement Plan

**Status**: ✅ All improvements completed

---

## Completed Improvements

### 1. Statistical Enhancement ✅
- [x] Expanded evaluation: 10 → 20 held-out prompts
- [x] Added TGRR metric
- [x] Added permutation p-values
- [x] Multi-seed validation (3 seeds)

### 2. Baseline Comparisons ✅
- [x] Random projection baseline
- [x] Student baseline (no teacher)
- [x] K-only ablation
- [x] V-only ablation

### 3. Analysis Enhancement ✅
- [x] Per-layer CHG analysis
- [x] Multi-seed stability analysis
- [x] KV channel ablation

### 4. Bug Fixes ✅
- [x] ValueMapper bias/scale formula
- [x] Evaluator stale reference
- [x] _lowrank_factor SVD redundancy

### 5. Paper Updates ✅
- [x] Reorganized structure
- [x] Added figures (7 total)
- [x] Updated results tables
- [x] Added analysis section

---

## Future Work (Not in Current Scope)

### Short-term
- [ ] Cross-family model pairs (Llama → Qwen)
- [ ] End-to-end timing (PSR measurement)
- [ ] Long-context scaling (512-8192 tokens)

### Medium-term
- [ ] Layer-importance weighting
- [ ] Neural projection comparison (C2C, MoT)
- [ ] Multi-turn stability testing

---

## Lessons Learned

1. **Sample size matters**: 10 → 20 samples reversed K-only/V-only conclusions
2. **Simple methods can win**: Native averaging outperforms complex ridge mapping
3. **Bug fixes change conclusions**: ValueMapper fix made V-only positive
4. **Negative results are valuable**: K-only failure reveals asymmetry
