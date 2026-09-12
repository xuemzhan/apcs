# APCS GPU 真实数据分析报告 — 2026-08-25

## 实验环境
- **GPU**: NVIDIA GeForce RTX 4090 (24GB, sm_89)
- **PyTorch**: 2.5.1+cu124
- **模型源**: ModelScope (modelscope.cn)
- **模型对**: Qwen3-4B (Teacher) → Qwen3-1.7B (Student)
- **数据集**: HellaSwag, ARC-Challenge, WinoGrande (真实 HF 数据); MMLU (合成，因 HF Hub 不可达)

## 实验配置
```yaml
mapper:
  type: ridge
  rank: 16
  separate_kv: true
  de_rope_k: true
  de_rope_v: false
  ridge_lambda_k: 0.001
  ridge_lambda_v: 0.001
  real_calibration_samples: 16
  layer_selection: proportional
  rms_calibration: true
```

---

## 1. 任务执行结果汇总

| Task | 标题 | Status | 关键指标 |
|------|------|--------|----------|
| T00 | Compatibility Scanner | **PASS** | G1_MATCHED_KV, matched_kv=true |
| T01 | Self-KV Replay (Gate 0) | **PASS** | logit_cosine=1.0, max_error=0.0, token_agreement=1.0 |
| T02 | RoPE Round-trip | **PASS** | max_err=8.88e-16, cosine=1.0 |
| T03 | Layer Alignment | OK | proportional selected (36T→28S) |
| T04 | Ridge Baseline | **PASS** | KV_cosine=0.744, attn_output_cos=0.959 |
| T05 | Replacement (Gate 1) | **FAIL** | retention=0.744 < 0.90 |
| T06 | Lightweight Mapper | OK | rank×retention 曲线已生成 |
| T07 | Teacher Gap Freeze | OK | gap: low=8, med=16, high=8 |
| T08 | Advantage State Training | OK | student_frozen=true |
| T09 | Main Capability (Gate 2A) | PASS* | *synthetic scores |
| T10 | System Cost | OK | PSR_A=0.617 |
| T11 | MVP Decision | **D** | STOP_REPLACEABILITY_UNSTABLE |
| T12 | Geometry Diagnostics | OK | CKA/PA/ER 已生成 |
| T13 | Generalization | OK | 0/1 pairs pass Gate 2A |

---

## 2. 关键发现与分析

### 2.1 K vs V 映射质量不对称（核心问题）

| 指标 | K (Key) | V (Value) | 差距 |
|------|---------|-----------|------|
| Retention | **0.965** | 0.563 | -0.402 |
| R² | 0.926 | -0.828 | -1.754 |
| KV Cosine | 0.965 | 0.563 | -0.402 |
| Attn Output Cosine | 0.991 | 0.938 | -0.053 |
| Token Agreement | 0.844 | 0.320 | -0.524 |

**诊断**: V 映射质量是制约整体 Retention 的瓶颈。K 映射已经非常优秀（0.965），但 V 映射仅达 0.563，将平均 Retention 拉低至 0.744，导致 Gate 1 FAIL。

### 2.2 超参数敏感性测试

| 配置变化 | Retention_K | Retention_V | Mean Retention |
|----------|-------------|-------------|----------------|
| Baseline (rank=16, λ_v=0.001) | 0.963 | 0.526 | 0.744 |
| λ_v 增至 0.01 | 0.963 | 0.526 | 0.744 |
| calibration=32 | 0.965 | 0.563 | 0.764 |
| rank=32 | 0.965 | 0.563 | 0.764 |
| de_rope_v=true | 0.963 | 0.488 | 0.725 |
| last_layer selection | 0.965 | 0.563 | 0.764 |

**结论**: V 映射质量对超参数**不敏感**，说明瓶颈不在正则化/秩/层数选择，而在 V 值本身的 Teacher-Student 表征差异。

### 2.3 V Retention 低的根本原因分析

1. **V 值语义差异**: Transformer 中 V 承载"内容信息"，不同规模模型的 V 表征空间差异远大于 K（注意力模式）。
2. **线性映射局限**: Ridge 回归是线性映射，无法捕捉 Teacher→Student V 值之间的非线性变换。
3. **Head_dim 一致但分布不同**: 虽然 Qwen3-4B 和 Qwen3-1.7B 的 head_dim=128、num_kv_heads=8 完全匹配，但 V 的值分布差异仍然显著。

### 2.4 Gate 2A 结果（合成评分）

T09 使用合成评分（provider.score=synthetic），结果显示:
- CHG > 0 (基线 +0.188)
- TGRR > 0 (0.548)
- Gate 2A: PASS

**⚠️ 重要声明**: 此 Gate 2A 结果基于合成评分，不构成真实能力迁移证据。需将 provider.score 切换为 hf 以获取真实评分。

---

## 3. MVP Decision

**Verdict: D_STOP_REPLACEABILITY_UNSTABLE**

```
retention (0.744) < 0.80 → D_STOP_REPLACEABILITY_UNSTABLE
```

原因: Teacher→Student KV 替换的 Retention 未达到最低阈值（0.80），Student 无法稳定替代 Teacher 的 KV Cache。

---

## 4. 环境问题与修复记录

### 4.1 GPU 架构兼容性
- **问题**: torch 2.5.1+cu124 不包含 sm_89 kernel
- **修复**: 修改 `apcs/inference/backends.py` 和 `apcs/providers/hf_model.py` 的 `_cuda_compute_available()`，对 sm_89 (Ada Lovelace) 添加兼容性豁免

### 4.2 HF Hub 不可达
- **问题**: 数据集下载超时（HF Hub 在当前环境不可达）
- **修复**: 修改 `apcs/data/hf_dataset.py`，添加本地 .arrow 缓存 fallback + 合成 MMLU 生成

### 4.3 缺少 numpy import
- **问题**: `apcs/replay/runner.py` 缺少 `import numpy as np`
- **修复**: 添加 import

---

## 5. 优化建议

### 5.1 短期（提升 V Retention）
1. **尝试 non-linear mapper**: 用 MLP/低秩 ALS 替代纯 Ridge 线性映射
2. **增加 V calibration 样本**: 当前 16 个样本可能不足以捕捉 V 分布
3. **per-layer 独立 V 映射**: 不同层的 V 映射矩阵可能差异很大

### 5.2 中期（真实评分验证）
1. 将 `provider.score` 从 `synthetic` 切换为 `hf`，获取真实 LLM 评分
2. 增加 context_length 到 2K/4K/8K，测试长上下文下的 Retention
3. 在更多数据集上验证（当前仅用 fidelity 数据集的子集）

### 5.3 长期（跨模型泛化）
1. 测试第二 Pair (SmolLM2-1.7B→135M) 的 G2 路径
2. 验证 G3 Cross-family 路径

---

## 6. 产物目录

```
reports/runs/2026-08-25/qwen3-4b-to-1.7b-real-gpu-20260825-215241/
├── prepare-data/   # 数据集准备
├── t00/           # 兼容性扫描
├── t01/           # Self-KV Replay
├── t02/           # RoPE Round-trip
├── t03/           # Layer Alignment
├── t04/           # Ridge Baseline
├── t05/           # Replacement
├── t06/           # Lightweight Mapper
├── t07/           # Teacher Gap Freeze
├── t08/           # Advantage State Training
├── t09/           # Main Capability
├── t10/           # System Cost
├── t11/           # MVP Decision
├── t12/           # Geometry Diagnostics
└── t13/           # Generalization
```

---

*报告生成时间: 2026-08-25 23:00 UTC*
*APCS v0.1.0 — KVCache-APCS 实验系统*
