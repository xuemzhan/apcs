# Cross-Model KV State Handoff Experiment Results

## Summary

| Teacher | Student | Avg CHG | Range | Verdict |
|---------|---------|---------|-------|---------|
| 4B | 1.7B | **+0.00** | [-0.09, +0.10] | Mixed |
| 4B | 0.6B | **+0.18** | [+0.18, +0.19] | Strong positive |
| 8B | 1.7B | **+0.003** | [-0.07, +0.10] | Mixed |
| 8B | 0.6B | **+0.18** | [+0.15, +0.20] | Strong positive |

## Detailed Results by Seed and Lambda

### 4B → 1.7B

| Seed | λ=1e-4 | λ=1e-3 | λ=1e-2 |
|------|--------|--------|--------|
| 42 | **+0.10** | **+0.10** | **+0.09** |
| 123 | **-0.08** | **-0.08** | **-0.09** |
| 456 | **-0.02** | **-0.02** | **-0.02** |

Teacher: 0.78, Student self: 0.69-0.80 (varies by seed)

### 4B → 0.6B

| Seed | λ=1e-4 | λ=1e-3 | λ=1e-2 |
|------|--------|--------|--------|
| 42 | **+0.18** | **+0.18** | **+0.18** |
| 123 | **+0.19** | **+0.19** | **+0.18** |
| 456 | **+0.18** | **+0.19** | **+0.19** |

Teacher: 0.78, Student self: 0.53-0.59 (varies by seed)

### 8B → 1.7B

| Seed | λ=1e-4 | λ=1e-3 | λ=1e-2 |
|------|--------|--------|--------|
| 42 | **+0.10** | **+0.10** | **+0.10** |
| 123 | **-0.02** | **-0.02** | **-0.02** |
| 456 | **-0.07** | timeout | **-0.06** |

Teacher: 0.70, Student self: 0.69-0.80 (varies by seed)

### 8B → 0.6B

| Seed | λ=1e-4 | λ=1e-3 | λ=1e-2 |
|------|--------|--------|--------|
| 42 | **+0.18** | **+0.18** | **+0.18** |
| 123 | **+0.15** | **+0.15** | **+0.16** |
| 456 | **+0.20** | **+0.20** | **+0.20** |

Teacher: 0.70, Student self: 0.53-0.59 (varies by seed)

## Analysis

### Key Finding: CHG Correlates with Teacher-Student Gap

The technique works best when there's a **large performance gap** between teacher and student:

- **0.6B student** (weak): CHG = +0.18 consistently
- **1.7B student** (stronger): CHG ≈ 0 (mixed)

This is consistent with first principles:
1. A weak student has more to learn from teacher's representations
2. A strong student already captures much of the context itself
3. The mapper can only transfer information the student doesn't already have

### Robustness Analysis

- **Lambda**: Minimal effect (1e-4 to 1e-2 all give similar results)
- **Seed**: Significant effect on 1.7B students, minimal on 0.6B students
- **Consistency**: 0.6B student results are highly consistent across all conditions

### Implications for Paper

1. **Main result**: KV handoff can improve weak students by +18% on MCQ tasks
2. **Boundary condition**: The technique is less effective when student is already strong
3. **Practical value**: Most useful for deploying small models that benefit from large model knowledge
4. **Theoretical insight**: The mapper learns a meaningful transformation that transfers task-relevant information
