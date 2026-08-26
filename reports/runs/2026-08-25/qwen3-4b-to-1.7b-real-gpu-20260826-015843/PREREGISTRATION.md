# PREREGISTRATION.md

_Generated at: 2026-08-26T02:19:56.288292_

## 1. Models

- Teacher: `Qwen/Qwen3-4B` @ `main`
  - dtype: `bfloat16`
  - attention: `sdpa`
- Student: `Qwen/Qwen3-1.7B` @ `main`
  - dtype: `bfloat16`
  - attention: `sdpa`
  - freeze: `True`

## 2. Dataset

- Primary teacher-advantage set: `mmlu`
- Splits: `['train', 'validation', 'test']`
- Fidelity set: `['hellaswag', 'arc_challenge', 'winogrande']`
- Long-context set: `None`
- Behavior-sensitive set: `None`

## 3. Hyperparameters

- Rank candidates: 8 / 16 / 32 (A1)
- source_top_k: `2`
- α_max: `0.5` (Bounded, A10)
- de-RoPE: `True` (A6)
- shared basis: `True` (A2)
- separate K/V: `True` (A8)
- advantage rank: `16`
- advantage separate KV: K=`lowrank`, V=`lowrank`
- RMS calibration: `True` (A9)
- Query gate: `False` (A11)

## 4. Seeds

- seeds: `[0, 1, 2]`

## 5. Statistics (§51)

- bootstrap_n: `1000`
- CI: `0.95`
- paired_test: `permutation`

## 6. Gates (§5, §6, §7)

- Gate 0 (Self-KV Replay): all samples PASS
- Gate 1 (Retention): `≥ 0.75` PASS
                     `≥ 0.9` STRONG
- Gate 2A (CHG): `> 0` AND bootstrap CI lower > 0 AND TGRR > 0
- Gate 2A (PSR_A): `> 0`

## 7. Negative Result Policy

CHG ≤ 0 时 **不重新筛选 test dataset**；保留全部负向结果。
若 Replacement 不稳定（Retention < 0.80），停止 Capability Transfer 扩展，
优先研究 Alignment / Direction Asymmetry / Geometry。

## 8. Forbidden (§52)

1. Student 在主实验中重新读取 X（除 calibration / eval 阶段）。
2. 微调 Student 主体后仍称 Runtime State Transfer。
3. Test 调参。
4. 只挑 Teacher-win Test Sample。
5. 隐藏 Teacher Prefill 成本。
6. 隐藏 H2D / Cache Load。
7. 用 R² / Cosine / CKA 代替 CHG。
8. Cache 注入失败后 Silent Re-prefill。