# Data Provenance Document

**Date**: 2026-08-28
**Status**: ⚠️ INCOMPLETE - Critical issues identified

---

## Experimental Data Sources

### Source 1: verification_results.json
- **Timestamp**: 2026-08-28 21:12
- **File**: `reports/verification/verification_results.json`
- **Protocol**: Unknown (not documented)
- **Status**: ⚠️ UNVERIFIED

**Results**:
| Method | CHG | CI Lower | CI Upper |
|--------|-----|----------|----------|
| Ridge | +0.2745 | +0.1069 | +0.4245 |
| Native | +0.1061 | -0.0800 | +0.2783 |
| K-only | +0.0828 | -0.0032 | +0.1753 |
| V-only | +0.1736 | -0.0148 | +0.3737 |

### Source 2: comprehensive_results.json
- **Timestamp**: 2026-08-28 22:11
- **File**: `reports/comprehensive/comprehensive_results.json`
- **Protocol**: Unknown (not documented)
- **Status**: ⚠️ UNVERIFIED

**Results**:
| Method | CHG | CI Lower | CI Upper | p-value |
|--------|-----|----------|----------|---------|
| Native | +0.2295 | +0.1221 | +0.3351 | 0.0004 |
| V-only | +0.1754 | +0.0536 | +0.2931 | 0.0039 |
| Ridge | +0.1348 | +0.0213 | +0.2530 | 0.0192 |
| K-only | -0.0287 | -0.1120 | +0.0535 | 0.7422 |
| Random | +0.0041 | -0.0865 | +0.0921 | 0.4626 |
| Student | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

---

## Critical Issues

### Issue 1: Contradictory Rankings
- **Experiment 1**: Ridge > Native > V-only > K-only
- **Experiment 2**: Native > V-only > Ridge > K-only

**Root Cause**: Unknown - protocol documentation missing

### Issue 2: No Prompt Data
- No prompt files exist in the repository
- Cannot reproduce exact experiments

### Issue 3: Simulated Multi-seed Data
- `full_results.json` showed std=0.0 across seeds (impossible)
- File has been deleted

### Issue 4: Incomplete Per-layer Data
- `comprehensive_results.json` has per_layer array
- `full_results.json` had different per_layer values (deleted)

---

## Missing Documentation

### 1. Protocol Documentation
- Which prompt format was used?
- Which scoring method was used?
- Which calibration set was used?

### 2. Code Version
- Git commit hash not recorded
- No version tagging

### 3. Environment
- Python version: 3.10.12 (from env)
- PyTorch version: Unknown
- GPU: NVIDIA 23.5GB (from env)

### 4. Prompt Data
- No saved prompt files
- Cannot verify "diverse domains" claim

---

## Recommendations

### Immediate Actions
1. **Do not use any numbers from these files** until verified
2. **Run fresh experiments** with full logging
3. **Save all prompts** to files
4. **Document protocol** explicitly

### Before Submission
1. **Re-run all experiments** with proper documentation
2. **Save intermediate results** (per-sample scores)
3. **Verify reproducibility** by running on different machine
4. **Have someone else** run the experiments

---

## Appendix: File Inventory

### Existing Files (Unverified)
- `reports/comprehensive/comprehensive_results.json` - Marked UNVERIFIED
- `reports/verification/verification_results.json` - Marked UNVERIFIED
- `reports/FINAL_GPU_PAPER_CONCLUSION_20260826.md` - Historical

### Deleted Files
- `reports/comprehensive/full_results.json` - Deleted (contained fake data)
- `generate_paper_figures.py` - Deleted (used simulated data)

### Missing Files
- Prompt data files
- Protocol documentation
- Code version information
- Environment specifications
