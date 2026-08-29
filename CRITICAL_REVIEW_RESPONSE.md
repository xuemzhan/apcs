# Critical Review Response & Verification Report

**Date**: 2026-08-28
**Status**: ⚠️ CRITICAL ISSUES IDENTIFIED
**Reviewer**: External审查者

---

## Executive Summary

The reviewer has identified **critical issues** that invalidate the current paper version. This report verifies each claim and provides a remediation plan.

**Key Findings**:
1. ✅ **CONFIRMED**: Two contradictory JSON files (verification vs comprehensive)
2. ✅ **CONFIRMED**: Figures used simulated data (deleted, but issue was real)
3. ✅ **CONFIRMED**: Multi-seed data is fake (std=0.0 across seeds)
4. ✅ **CONFIRMED**: No actual prompt data saved
5. ✅ **CONFIRMED**: Paper claims contradict reproducible data

**Verdict**: The 8/28 paper version is NOT suitable for submission.

---

## Issue-by-Issue Verification

### Issue 1: Contradictory JSON Files

**Reviewer Claim**: verification_results.json (21:12) and comprehensive_results.json (22:11) give contradictory rankings.

**Verification**:

| Method | verification (21:12) | comprehensive (22:11) | Contradiction? |
|--------|---------------------|----------------------|----------------|
| Ridge | **+0.2745** | +0.1348 | ✅ YES (2x difference) |
| Native | +0.1061 | **+0.2295** | ✅ YES (ranking reverses) |
| K-only | +0.0828 | -0.0287 | ✅ YES (sign reverses) |
| V-only | +0.1736 | +0.1754 | ❌ Similar |

**Root Cause**: These files likely come from different experimental protocols or code versions. The repository has no provenance metadata to explain the difference.

**Verdict**: ❌ **CONFIRMED** - Internal contradiction exists.

---

### Issue 2: Figure 5 (Per-Layer) Used Simulated Data

**Reviewer Claim**: 26 of 28 layers were randomly generated.

**Verification**:
- The `generate_paper_figures.py` file was deleted during cleanup
- However, the `comprehensive_results.json` contains actual per_layer data
- The paper's Figure 5 claims to show "per-layer analysis" but the code used random values

**Actual per_layer data from comprehensive_results.json**:
```
Layer 19: 0.1215 (matches paper)
Layer 25: 0.1653 (matches paper)
Layer 23: -0.0252 (negative, but figure showed positive)
Layer 27: -0.0567 (negative, but figure showed near zero)
```

**Verdict**: ❌ **CONFIRMED** - Figure was based on simulated data.

---

### Issue 3: Multi-Seed Data is Fake

**Reviewer Claim**: Multi-seed results show std=0.0, indicating fake data.

**Verification** from `full_results.json`:
```json
"native": {
  "chg_mean": 0.2295,
  "chg_std": 0.0,  // ← IMPOSSIBLE: std=0 means identical results
  "seed_results": [
    {"seed": 0, "chg": 0.2295},
    {"seed": 42, "chg": 0.2295},
    {"seed": 84, "chg": 0.2295}
  ]
}
```

**Verdict**: ❌ **CONFIRMED** - Multi-seed data is simulated, not real.

---

### Issue 4: No Actual Prompt Data Saved

**Reviewer Claim**: The 30 calib + 20 held-out prompts don't exist in the repository.

**Verification**:
```bash
find . -name "*prompt*" -o -name "*calib*" -o -name "*heldout*" | grep -v __pycache__
# Result: (empty)
```

**Verdict**: ❌ **CONFIRMED** - No prompt data files exist.

---

### Issue 5: Paper Claims Contradict Reproducible Data

**Reviewer Claim**: Paper says "V transfers, K does not" but reproducible data shows opposite.

**Verification**:
- **Paper claim**: V-only TGRR=0.751 (PASS), K-only CHG<0 (FAIL)
- **verification_results.json**: K-only +0.0828, V-only +0.1736
- **comprehensive_results.json**: K-only -0.0287, V-only +0.1754

**Verdict**: ⚠️ **PARTIALLY CONFIRMED** - Results depend on which JSON file is used.

---

### Issue 6: Protocol Sensitivity

**Reviewer Claim**: Changing prompt format reverses conclusions.

**Verification**:
- This is a legitimate methodological concern
- The repository has no documentation of which protocol was used for which results
- No A/B testing records exist

**Verdict**: ⚠️ **UNVERIFIABLE** - Cannot confirm without protocol documentation.

---

### Issue 7: PPL Disaster Not Reported

**Reviewer Claim**: Native injection causes PPL≈2.2M, not reported in paper.

**Verification**:
- No PPL data exists in the repository JSON files
- The paper does not mention PPL at all

**Verdict**: ⚠️ **UNVERIFIABLE** - No PPL data to confirm or deny.

---

### Issue 8: "First systematic comparison" Claim

**Reviewer Claim**: Paper claims "first systematic comparison" but doesn't compare with Heo et al.

**Verification**:
- Paper does cite Heo et al.
- No direct comparison under same protocol exists

**Verdict**: ❌ **CONFIRMED** - Claim is overreaching.

---

## Summary of Confirmed Issues

| Issue | Status | Severity |
|-------|--------|----------|
| Contradictory JSON files | ✅ CONFIRMED | 🔴 CRITICAL |
| Figure 5 simulated data | ✅ CONFIRMED | 🔴 CRITICAL |
| Multi-seed fake data | ✅ CONFIRMED | 🔴 CRITICAL |
| No prompt data saved | ✅ CONFirmed | 🟡 HIGH |
| Paper claims contradict data | ⚠️ PARTIAL | 🟡 HIGH |
| Protocol sensitivity | ⚠️ UNVERIFIABLE | 🟡 HIGH |
| PPL not reported | ⚠️ UNVERIFIABLE | 🟠 MEDIUM |
| Overreaching claims | ✅ CONFIRMED | 🟡 HIGH |

---

## Remediation Plan

### Phase 1: Immediate (Today)

1. **Delete all simulated/fake data**
   - Remove `full_results.json` (contains fake multi-seed)
   - Mark `comprehensive_results.json` as unverified
   - Mark `verification_results.json` as unverified

2. **Update paper to remove unverified claims**
   - Remove "first systematic comparison"
   - Remove specific numbers until verified
   - Add explicit uncertainty statements

3. **Create data provenance document**
   - Document which protocol produced which results
   - Document code versions
   - Document any manual interventions

### Phase 2: Short-term (This Week)

1. **Run fresh experiments with full logging**
   - Save all prompts
   - Save all intermediate results
   - Save code version (git hash)
   - Save environment details

2. **Implement proper figure generation**
   - Read data from JSON files
   - No hardcoded values
   - Include data provenance in figure captions

3. **Resolve contradictory results**
   - Identify which protocol is correct
   - Re-run with correct protocol
   - Document the resolution

### Phase 3: Medium-term (Before Submission)

1. **Full reproducibility package**
   - Docker container
   - All data files
   - All scripts
   - All results

2. **External validation**
   - Have someone else run the experiments
   - Verify all claims

---

## Revised Paper Strategy

Given the severity of these issues, I recommend:

1. **Do NOT submit the current version**
2. **Focus on reproducibility first**
3. **Only claim what can be fully verified**
4. **Be transparent about limitations**

The core idea (cross-model KV transfer) may still be valid, but the current evidence is insufficient to support the claims.

---

## Appendix: Files to Delete/Modify

### Files to Delete
- `reports/comprehensive/full_results.json` (fake multi-seed data)
- `generate_paper_figures.py` (was deleted, but issue stands)

### Files to Mark as Unverified
- `reports/comprehensive/comprehensive_results.json`
- `reports/verification/verification_results.json`

### Files to Create
- `DATA_PROVENANCE.md` (document all data sources)
- `PROTOCOL.md` (document experimental protocols)
- `FRESH_EXPERIMENT_LOG.md` (log of fresh experiments)
