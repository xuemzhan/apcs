# Experimental Protocol

**Date**: 2026-08-28（v1.0） / 2026-08-29（v1.1 修订见 §11）
**Status**: ACTIVE
**Purpose**: Ensure reproducible experiments with proper provenance

---

## 1. Overview

This document defines the experimental protocol for cross-model KV Cache transfer evaluation. All experiments must follow this protocol exactly.

---

## 2. Models

### 2.1 Primary Pair
- **Teacher**: Qwen3-4B
  - Path: `~/.cache/modelscope/models/Qwen--Qwen3-4B/snapshots/master`
  - Layers: 36
  - Hidden size: 2560
  - KV heads: 8
  - Head dim: 128

- **Student**: Qwen3-1.7B
  - Path: `~/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master`
  - Layers: 28
  - Hidden size: 2048
  - KV heads: 8
  - Head dim: 128

### 2.2 Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

teacher = AutoModelForCausalLM.from_pretrained(TEACHER_PATH)
student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH)
tokenizer = AutoTokenizer.from_pretrained(STUDENT_PATH)
```

---

## 3. Prompts

### 3.1 Prompt Sources
All prompts must be:
1. Saved to files before experiments
2. From diverse domains
3. Not overlapping between calibration and held-out sets

### 3.2 Prompt Format
Each prompt is a multiple-choice question with 4 options (A, B, C, D).

**Format**:
```
Question text?
(A) Option A
(B) Option B
(C) Option C
(D) Option D
```

### 3.3 Prompt Files
- `data/calibration_prompts.json`: 30 prompts for calibration
- `data/heldout_prompts.json`: 20 prompts for evaluation
- `data/all_prompts.json`: All 50 prompts with metadata

### 3.4 Prompt Metadata
Each prompt includes:
```json
{
  "id": "prompt_001",
  "domain": "factual",
  "question": "What is the capital of France?",
  "choices": ["Paris", "London", "Berlin", "Madrid"],
  "answer": "A",
  "difficulty": "easy"
}
```

---

## 4. Methods

### 4.1 Student Baseline
- Use student's own prefill cache
- No teacher information
- Expected CHG: 0.0

### 4.2 Native Average
- Average teacher layers per student layer
- No learned parameters
- Formula: $\widehat K^S_{\ell h} = \frac{1}{n} \sum_{i \in \mathcal{A}(\ell)} K^T_{ih}$

### 4.3 Ridge KV Mapping
- Per-head ridge regression
- Aggregate Gram-matrix training
- Regularization: λ=1e-5
- Formula: $\widehat K^S_{\ell h} = Z_{\ell h}^{K} W^K_{\ell h}$

### 4.4 V-Only
- Map values only
- Pass keys through unchanged

### 4.5 K-Only
- Map keys only
- Pass values through unchanged

### 4.6 Random Projection
- Random orthogonal matrix
- Tests if alignment is necessary

---

## 5. Evaluation Protocol

### 5.1 Scoring Method
**CRITICAL**: Use full-text scoring, NOT letter-only scoring.

```python
def score_choices(logits, choices, tokenizer):
    """Score using full text of choices, not just letters."""
    scores = []
    for choice in choices:
        # Tokenize choice text
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        # Get logits for choice tokens
        choice_logits = logits[-len(choice_ids):]
        # Compute log probability
        log_prob = sum(log_prob[i] for i, choice_id in enumerate(choice_ids))
        scores.append(log_prob)
    return scores
```

### 5.2 CHG Calculation
```python
def chg(score_handoff, score_student):
    """Capability Gain = Handoff - Student"""
    return score_handoff - score_student
```

### 5.3 TGRR Calculation
```python
def tgrr(score_handoff, score_student, score_teacher):
    """Teacher Gap Recovery Rate"""
    gap = score_teacher - score_student
    if gap <= 0:
        return 0.0
    return (score_handoff - score_student) / gap
```

### 5.4 Statistical Tests
- **Bootstrap CI**: 1000 resamples, 95% confidence
- **Permutation test**: 1000 permutations, H0: mean(CHG) = 0

---

## 6. Data Logging

### 6.1 Required Outputs
Every experiment must produce:

1. **Prompt file**: `prompts_{timestamp}.json`
2. **Results file**: `results_{timestamp}.json`
3. **Log file**: `log_{timestamp}.txt`
4. **Code version**: `git_hash.txt`

### 6.2 Results Format
```json
{
  "experiment_id": "exp_20260828_001",
  "timestamp": "2026-08-28T22:00:00",
  "git_hash": "abc123",
  "protocol_version": "1.0",
  "models": {
    "teacher": "Qwen3-4B",
    "student": "Qwen3-1.7B"
  },
  "config": {
    "n_calibration": 30,
    "n_heldout": 20,
    "scoring_method": "full_text",
    "seed": 42
  },
  "results": {
    "student_baseline": {
      "mean_score": 0.546,
      "scores": [0.5, 0.6, ...]
    },
    "native": {
      "mean_score": 0.776,
      "chg": 0.230,
      "tgrr": 0.768,
      "p_value": 0.0004,
      "ci_lower": 0.122,
      "ci_upper": 0.335,
      "scores": [0.7, 0.8, ...]
    }
  },
  "per_prompt": [
    {
      "prompt_id": "prompt_001",
      "student_score": 0.5,
      "teacher_score": 0.8,
      "native_score": 0.7,
      "ridge_score": 0.65,
      "v_only_score": 0.68,
      "k_only_score": 0.52,
      "random_score": 0.51
    }
  ]
}
```

### 6.3 Log Format
```
[2026-08-28 22:00:00] Experiment started: exp_20260828_001
[2026-08-28 22:00:01] Loading teacher model: Qwen3-4B
[2026-08-28 22:00:05] Loading student model: Qwen3-1.7B
[2026-08-28 22:00:10] Loading prompts: 30 calibration, 20 held-out
[2026-08-28 22:00:15] Running method: student_baseline
[2026-08-28 22:00:30] Method completed: mean_score=0.546
[2026-08-28 22:00:35] Running method: native
[2026-08-28 22:01:00] Method completed: mean_score=0.776, chg=+0.230
...
[2026-08-28 22:05:00] Experiment completed
```

---

## 7. Reproducibility Requirements

### 7.1 Code Version
- Record git hash before running
- Never modify code during experiment

### 7.2 Random Seeds
- Use fixed seeds for reproducibility
- Record all seeds used

### 7.3 Environment
- Record Python version
- Record PyTorch version
- Record CUDA version
- Record GPU model

### 7.4 Data Version
- Hash all prompt files
- Record any data preprocessing

---

## 8. Quality Checks

### 8.1 Before Running
- [ ] All prompts saved to files
- [ ] Git hash recorded
- [ ] Environment documented
- [ ] Protocol version recorded

### 8.2 During Running
- [ ] Log file growing
- [ ] No errors in output
- [ ] Memory usage reasonable

### 8.3 After Running
- [ ] All results files created
- [ ] Per-prompt scores saved
- [ ] Statistical tests completed
- [ ] Results match expectations (e.g., student_baseline CHG ≈ 0)

---

## 9. Common Pitfalls

### 9.1 Letter-Only Scoring
**WRONG**: Score only "A", "B", "C", "D" tokens
**RIGHT**: Score full choice text

### 9.2 Missing Calibration
**WRONG**: Fit mapper on held-out data
**RIGHT**: Fit mapper only on calibration data

### 9.3 No Prompt Saving
**WRONG**: Generate prompts in memory
**RIGHT**: Save prompts to files before experiment

### 9.4 Single Seed
**WRONG**: Run once and report
**RIGHT**: Run multiple seeds and report mean ± std

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-28 | Initial protocol |
| 1.1 | 2026-08-29 | 见 §11 |

---

## 11. Protocol v1.1 修订（2026-08-29）

针对外部审查确认的证据缺陷，v1.1 强制以下变更（实现位于 `apcs/inference/`）：

1. **主指标换轨（§5 修订）**：能力结论只认 gold 概率 / accuracy / TGRR；
   逐样本记录 `gold_prob`；bootstrap CI（1000 次）+ 符号翻转 permutation p
   （`apcs.metrics.permutation_p`）。置信度（argmax 字母 softmax）降级为
   兼容字段，**禁止**作为能力证据。
2. **校准/评估互斥切分**：`inject_eval.calib_eval_split=true`（默认开），
   mapper 仅用前 `calib_samples` 行拟合，评估行与校准行同分布互斥；
   关闭时 provenance 如实标注 `calib_eval_disjoint=false`。
3. **基线强制**：默认消融含 `native`（无参数层均值教师 KV）；
   `random_proj` / `k_only` / `v_only` 可选。任何"mapper 有效"的结论
   必须同时报告 native 基线。
4. **KV 持久化与两阶段**：`persist_kv=true` 将 Teacher KV（逐样本 npz +
   sha256 manifest）、rows、teacher 分数、mapper 参数落盘至
   `kv_store/`；`online_kv_dir` 触发在线阶段（仅加载 Student、从盘冷启动，
   `evaluate_online`）。mapper 参数优先从盘恢复（`mapper_source=
   offline_store`），在线拟合必须如实标注。
5. **PSR 核算**：逐样本计时 student self-prefill 与冷启动 handoff，
   输出 `psr`（注明含测量用 PPL forward 的保守性）。
6. **RoPE 对齐模式**：`mapper.rope_align=unrotated`（fit 目标 de-RoPE、
   注入前 re-RoPE，§23 本意）为推荐值；`rotated`（W 吸收位置旋转）保留
   作对照。两者对比必须同协议运行。
7. **审计真实化**：zero-prefill 断言数据源改为实际 cache 长度
   （`cache_seq_len`），forward 前后增量计数；禁止手工赋值。
8. **版本固定**：所有 artifacts 携带 `protocol_version` 与 `git_hash`；
   无 git hash 的历史结果视为不可复现。

---

## 12. Protocol v1.2 / v1.3（2026-08-29 下午）：三假说框架与规模阶梯

### 12.1 三假说分解（能力声明的唯一合法路径）

| 假说 | 判定实验 | 通过判据 |
|---|---|---|
| H1 机制无损 | `self_kv` 恒等注入（学生自 KV 走完整注入路径） vs student_self | 逐样本 logit 余弦 ≈ 1.0 |
| H2 映射质量 | mapper kv_both vs self_kv | gold CHG → 0（PPL 比值 → 1） |
| H3 可利用性 | 混合目标 task-aware vs self_kv 上限 | held-out gold CHG > 0 且 CI > 0 |

**H1 已判定通过**（logit 余弦 = 1.000）。**H3 的判定以 H2 达标为前提**；
H2 的第一杠杆是校准样本规模（§32 原目标 100–500）。

### 12.2 v1.2 强制项（在 v1.1 基础上追加）

1. **`self_kv` 恒等对照**必须出现在每次评估的消融列表中；
2. **统一 prompt 格式**：所有方法共用 `tokenize(ctx)+tokenize(query+"\n\nAnswer:")`，
   方法间唯一差异是 context 来源（消除格式混淆）；
3. **KV 范数诊断**（`kv_norm_diagnostics`）必须落盘；`native_rescale=true`
   用于分离尺度失配与语义失配；
4. **logit 余弦**逐样本记录（`logit_cos_vs_student`）——行为层保真度；
5. **teacher-summary 文本基线**（`summary_baseline=true`）——KV 路线的
   实用性竞争者，任何"KV 迁移有价值"的结论必须优于它。

### 12.3 v1.3 规模阶梯（当前执行）

- mapper 结构：`affine`（per-head 带截距）与 **`affine_layer`**
  （逐层中心化岭回归，参数少 H=8 倍，行/参数比改善 8 倍）；
- 校准阶梯：30 → 200（→500）；评估 ≥100（检出 δ≈0.07 的功效）；
- 已排除：unrotated 与 affine 不叠加（V 通道一阶问题由中心化解决）；
  task-aware 在 30 样本下过拟合（PPL 88→18.7k），需规模前置。

### 12.4 已证伪/已排除的假设（防止重复走弯路）

- RoPE 缩放/对齐方式：全家族 θ=1e6 无 scaling，de-RoPE 数学正确；
  unrotated 模式不带来额外收益（V 是瓶颈，K 不是）；
- 尺度失配为唯一原因：native 重标定后仍 PPL 5.4e5 ⇒ 语义/结构失配为主；
- 机制损失：self_kv 恒等排除注入/位置/GQA/持久化链路问题。

---

## 13. Protocol v1.4（2026-08-29 晚）：RAT、Oracle 探针与 H3 终审

### 13.1 相关文献定位

cache-translation 已成为活跃方向：MoT (arXiv 2607.28979)、LSC (2601.06123)、
C2C (2510.03215)、KVComm、Interlat、HCache。**MoT 的 baseline "C2C-Project"
（单投影映射 ≈ 本仓库 affine 族）在其评测中近随机** —— 与本仓库 H2 负
结果独立互证。MoT 的有效成分是 cross-attention translator + Context
Correction Loss + **窗口翻译+目标侧重放**（非全层替换、非严格零-prefill）。

### 13.2 RAT（Residual-Anchored Translator）

架构依据：KV = W_V·h（残差流线性投影）⇒ 跨模型翻译可分解为
`W = pinv(w_t).T @ R @ w_s.T`，R 为共享词表 embedding 的最小二乘对齐
（Qwen3 同词表，免配对数据）；头对应用协方差轮廓 + 匈牙利匹配；
position 0（注意力汇）用学生统计覆盖。**实测：解析核有害（去锚后
PPL 502→23.7）** —— min-norm pullback 在真实模型上不成立，诚实记录。

### 13.3 Oracle 探针（H3 上界判定，probe_mode 门控、非部署路径）

- `mix_aXX`：cache = (1−α)·学生自KV + α·翻译KV —— **实测随 α 单调劣化**；
- `win_{low,mid,high}`：MoT 注入模式的探针版 —— **顶层 1/3 无害，
  中/低层有害**。
- **判读规则**：探针给出"教师 KV 可利用性"的上界；上界为零时，任何
  mapper 投入都不可能产生 CHG>0，应转向 replacement 叙事。本仓库
  已触发该规则（2026-08-29）。

### 13.4 PPL/accuracy 脱钩现象

RAT noanchor(rank32) 达到 PPL 23.7（≈自 prefill 21.2）但 acc 0.267：
cache 翻译可恢复"语言流畅性"，不可恢复"教师级上下文理解"。
**PPL 不得单独作为 cache 翻译质量的证据**（写入评审 checklist）。
