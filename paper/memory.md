# MEMORY.md
## 跨模型 KV Cache 运行时能力迁移研究 Session 总结

**Session 截止时间：2026-08-08 21:16（UTC+8）**

> 本文档用于后续会话连续研究，汇总本次 Session 形成的研究问题、文献边界、方法版本、资源约束、实验方案、论文框架、审稿修订、先验推断数据及后续创新方向。
>
> **重要说明**：本 Session 中部分论文结果数字与图表属于 **Prior Projection / Simulated / NOT Measured**，不是实测数据。后续真实实验必须逐项替换，不能作为科学证据。

---

# 1. 研究问题的最终收敛

研究从“让小模型复用大模型 KV Cache”逐步收敛为：

> 一个更强的 Teacher LLM 已经处理长上下文 `X` 并形成 KV Runtime State。能否把该状态转换给更小的 Student LLM，使 Student **完全不重新 Prefill 原上下文 X**，直接继续推理；进一步，能否让 Student 不仅恢复自身 Self-Prefill 水平，还继承部分 Teacher 的任务优势？

核心对象：

```text
Teacher: T
Student: S
Long context: X
New query: q
Teacher runtime state: C_T(X)
Student self-prefill state: C_S(X)
Handoff state: C*_S(X,q)
```

严格约束：

```text
Student MUST NOT full-prefill original context X.
Student MAY process the short new query q.
Student backbone parameters remain frozen.
```

最终目标不是：

```text
C*_S ≈ C_S
```

而是：

```text
Score(S | C*_S) > Score(S | C_S)
```

即：

```text
CHG > 0
```

该问题最终被命名为：

**Runtime Capability Transfer via KV State Handoff**

---

# 2. 三类研究范式的严格区分

## 2.1 Information Injection

形式：

```text
C_S + F(C_T) → C'_S
```

特点：

- Student 已经处理过 `X`；
- Student 已有 Self-KV；
- Teacher 信息只是在已有 Student State 上增强。

代表方向：C2C、LCF 等。

---

## 2.2 Cache Emulation / Replacement

形式：

```text
C_T → Ĉ_S ≈ C_S
```

特点：

- Student 不重新 Prefill `X`；
- Mapper 尽量恢复 Student 自己会生成的状态/行为；
- 核心目标是高 Retention，而不是超过 Student。

代表：Cross-Model KV Cache Transfer、Ridge/MLP Mapper。

---

## 2.3 Advantage-Preserving State Substitution

本研究主线：

```text
C_T → C*_S
```

要求：

```text
Student never materializes C_S(X)
```

同时：

```text
Score(S | C*_S) > Score(S | C_S)
```

即从“Prefill 替代”升级到“运行时能力迁移”。

---

# 3. 核心指标

## 3.1 Retention

```text
Retention = Score_handoff / Score_student-self
```

解释：高 Retention 只能证明 **State Compatibility**，不能证明 Capability Transfer。

## 3.2 CHG — Capability Handoff Gain

```text
CHG = Score_handoff - Score_student-self
```

这是整个研究最重要的科学端点。

只有 `CHG > 0` 才能支持 Runtime Capability Transfer。

## 3.3 TGRR — Teacher Gap Recovery Rate

```text
TGRR =
(Score_handoff - Score_student)
/
(Score_teacher - Score_student)
```

衡量 Teacher–Student 原始能力差中有多少被 Runtime State 恢复。

## 3.4 PSR_A — Scenario A Prefill Saving Ratio

只适用于 Teacher Cache 已因上游业务自然存在：

```text
PSR_A =
1 -
(T_map + T_load + T_query)
/
T_prefill,S(X)
```

## 3.5 PCR — Parameter Compression Ratio

```text
PCR = MapperSize_ours / MapperSize_ridge
```

---

# 4. 系统成本的三种场景

这是审稿优化后必须严格区分的口径。

## Scenario A — Natural Handoff / Mid-session Downgrade

Teacher 已因业务处理过 `X`，Teacher Prefill 为沉没成本。

合法报告：`PSR_A`。

这是主论文系统场景。

## Scenario B — Teacher-for-Transfer

Teacher 仅为了给 Student 生成 Transfer State 才运行。

必须计算：

```text
Cost_B =
T_prefill,T
+ T_map
+ T_load
+ T_query
```

禁止用 `PSR_A` 声称端到端收益。

## Scenario C — One-Teacher-Many-Student / Repeated Reuse

一次 Teacher Prefill 被多个 Student / 多次 Query 复用。

```text
Cost_baseline(N) = N * T_prefill,S
```

```text
Cost_handoff(N) =
T_prefill,T
+ T_map
+ N(T_load + T_query)
```

Break-even：

```text
N_BE =
min { N : Cost_handoff(N) < Cost_baseline(N) }
```

必须在同一 Hardware、Batch、Cache Residency 下实测。

---

# 5. 方法版本演进

## 5.1 V1.1

早期思路包括：

- Cross-model compatibility mapper；
- Residual；
- Query-conditioned gate；
- No Student re-prefill。

主要风险：Query Gate 若每次扫描完整 Cache，会产生 `O(|X|)` per-query 成本；同时 Low-rank / Residual / Gate 都容易与已有工作重叠。

---

# 6. V1.2：APCS 正式形成

研究主名义：

**Runtime Capability Transfer via KV State Handoff**

Paper 1：

**Advantage-Preserving Large-to-Small KV Handoff**

候选题目：

**Beyond Cache Emulation: Advantage-Preserving Large-to-Small KV Handoff**

方法命名：

**APCS — Advantage-Preserving Cache Synthesis**

核心思想：Compatibility 与 Advantage 解耦。

一次性 Context Synthesis：

```text
C_base(X) = F_compat(C_T(X))
R_adv(X) = E_adv(C_T(X))
```

轻量 Query Adaptation：

```text
g(q) = G(E_q(q))
```

```text
C*_S(X,q) = C_base(X) + Expand[g(q)] ⊙ R_adv(X)
```

系统约束：

- 所有 `O(|X|)` 工作只发生一次；
- Per-query adaptation 必须很轻；
- 禁止每个 Query 重扫长 Cache。

---

# 7. 最新 APCS V2 参数化

经过审稿意见后，`R_adv` 不再是抽象黑箱。

正式结构：

```text
Teacher aligned layers
        ↓
Source-Layer Mixer
        ↓
Separate K/V Low-Rank Residual Adapters
        ↓
Residual RMS Calibration
        ↓
Bounded Layer-wise Injection
        ↓
C*_S
```

## 7.1 Source-Layer Mixer

对于 Student Layer `l`、Head `h`：

```text
A(l) = selected teacher layers
```

学习 `w_(l,i,h)`：

```text
Z_(l,h) = Σ_i w_(l,i,h) C_T^(i,h)
```

Layer Alignment 至少比较：

- Proportional
- Last-Layer
- Data-driven Top-k
- Geometry-aware Top-k

## 7.2 K / V 独立 Low-Rank Residual

```text
R^K_(l,h) = A^K_(l,h) σ(B^K_(l,h) Z^K_(l,h))
```

```text
R^V_(l,h) = A^V_(l,h) σ(B^V_(l,h) Z^V_(l,h))
```

Rank Sweep：

```text
r ∈ {8,16,32}
```

默认：`r=16`；可选 `r=64` 上限对照。

## 7.3 de-RoPE Key Mapping

Key 优先：

```text
Teacher K
→ de-RoPE
→ map/residual
→ Student-side RoPE
```

强制消融：

```text
Correct de-RoPE mapping
vs
Direct RoPE-space mapping
```

## 7.4 Residual RMS Calibration

限制：

```text
RMS(R_adv) / RMS(C_base)
```

避免 Residual Norm Explosion、粗暴放大激活造成伪 CHG 或 KL/JCR 崩坏。

## 7.5 Bounded Layer-wise Injection

```text
alpha_l = alpha_max * tanh(a_l)
```

最终：

```text
C*_S = C_base + alpha_l R_adv
```

`alpha_max` 只能由 Validation 决定，Test 前冻结。

## 7.6 Query Gate

首轮默认关闭。

只有证明不扫描长 `X` Cache 时才允许启用。

Query Gate 只能读取短 `q` 并输出 Layer/Head/Latent-slot 系数。

---

# 8. Training Objective

最新版：

```text
L =
lambda_task * L_task
+ lambda_self * L_self
+ lambda_teacher * w * L_teacher
+ lambda_att * L_att
+ lambda_reg * L_reg
```

- `L_task`：真实任务答案 / suffix；
- `L_self`：保护 Student 稳定行为；
- `L_teacher`：Teacher Guidance，只使用 Train Split Margin；
- `L_att`：Attention Output / Logit 行为约束；
- `L_reg`：控制 Rank、Residual Norm、Alpha、Mapper Size、Sparsity。

---

# 9. Paper 1 只保留三项 Contribution

## Contribution 1 — Problem Formulation

Advantage-Preserving Large-to-Small KV Handoff：

- No Student X-Prefill；
- 同时评价 Teacher Advantage Recovery 与真实 Prefill Saving。

## Contribution 2 — Method

APCS：

```text
Compatibility Base
+
Advantage-Carrying State
+
Bounded lightweight adaptation
```

## Contribution 3 — Empirical/System Characterization

联合研究：

```text
model gap
× task
× context length
× mapper budget
× geometry
× behavior
× deployment cost
```

得到 CHG / TGRR / PSR_A / PCR / failure taxonomy / geometry-capability coupling。

---

# 10. 明确不作为 Paper 1 核心创新的内容

以下只作为实现机制或消融：

- Low-rank Mapper
- Residual
- Query Gate
- Attention-output Loss
- Shared Latent
- CCIR
- Cache Store
- Router
- Agent Platform

原因：2026 年相关工作已覆盖大量相似单项机制。

---

# 11. 最新文献边界（截至 2026-08）

## 11.1 Cross-Model KV Cache Transfer

**Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**

arXiv:2608.03893。

关键事实：

- Source KV → Receiver-compatible KV；
- Receiver 可完全跳过自身 Prefill；
- 同 Family Matched-KV；
- Ridge/MLP；
- Mapper 相对 re-prefill 约快 2.7–25×；
- 部分 Pair Retention 73–98%；
- Large→Small 方向可能脆弱、不对称；
- Attention-output cosine 比 Raw R² 更预测 Retention；
- Pairwise Mapper 可能达到 B 参数级、GB 存储级。

最重要边界：

> **Receiver-free 本身不是创新。**

## 11.2 C2C

**Cache-to-Cache: Direct Semantic Communication Between Large Language Models**，ICLR 2026。

属于 Information Injection：Source KV 与 Target Self-KV 融合。

## 11.3 LatentAlign

共享 KV Latent Space，因此 Shared Latent / CCIR 本身不够新。

## 11.4 LCF

Lightweight latent communication，包含压缩、Residual、Gate 等，因此“轻量 + Residual + Gate”也不能单独作为创新。

## 11.5 KaVa

Teacher compressed KV 作为 Student latent reasoning 的训练监督，因此“Teacher KV Distillation”本身不是新问题。

## 11.6 DroidSpeak

同架构 / Fine-tuned variants，Selective Layer Recompute + Reuse；不是任意异构 Large→Small State Synthesis。

## 11.7 ICaRus / PrefillShare

通过架构或训练期共设计共享 Prefill，不是 post-hoc Teacher→Student Runtime Handoff。

## 11.8 When KV Cache Reuse Fails in Multi-Agent Systems

ACL 2026。

关键发现：

- End-task Accuracy 可能稳定；
- Judge / Candidate Selection 行为仍可能明显漂移。

引出 JCR，因此最新版实验必须加入：

- JCR-style Consistency
- Next-token KL
- Behavior-sensitive Benchmark

---

# 12. 最新研究假设

- **H0 Compatibility**：Large→Small Matched-KV State 可高 Retention 替代 Student Self-Prefill。
- **H1 Capability Transfer**：Teacher Advantage 一部分存在于可迁移 Runtime State 中。
- **H2 Low-dimensional Advantage**：优势可能集中于低维/稀疏子空间，不需 B 级 Mapper。
- **H3 Behavior-Aligned Objective**：Attention/Behavior Objective 比 Raw KV Reconstruction 更重要。
- **H4 Teacher Gap Relationship**：Teacher Gap 与 CHG/TGRR 有规律。
- **H5 Long-context Pareto**：一次 Context Synthesis + 多 Query 复用在长 Context 可能形成 Pareto。
- **H6 Behavior Stability**：能力迁移必须与行为稳定性共同评价。

---

# 13. Geometry–Capability 机制研究

最新版论文把“几何不兼容”升级为正式实验。

必须测：

- Attention-output cosine
- Linear CKA
- Principal Subspace Angle
- Effective Rank
- Head-wise correlation

分析：

```text
Geometry metric ↔ Retention
Geometry metric ↔ CHG
```

三类机制：

### Type I

```text
Low numerical similarity
High task fidelity
```

结论：Raw KV Similarity 不是必要条件。

### Type II

```text
High Retention
CHG <= 0
Poor deep-layer geometry
```

结论：State Compatibility ≠ Capability Compatibility。

### Type III

```text
CHG > 0
Localized aligned subspace
```

结论：Student 可能只消费 Teacher Advantage 中与自身 Geometry 可兼容的局部子空间。

---

# 14. 最小资源研究配置

资源有限，因此研究压缩到单卡。

推荐：

```text
GPU: 1×24GB NVIDIA
RAM: 64–128GB
NVMe: 1–2TB
Teacher: Qwen3-4B
Student: Qwen3-1.7B
```

原因：

- Same Family
- Tokenizer compatible
- Matched KV heads/head_dim
- RoPE compatible
- 仍有 Layer/Hidden Size Gap
- 单 24GB GPU 可通过 Sequential Residency 验证机制

极限调试：

```text
Qwen3-1.7B → Qwen3-0.6B
```

只适合 Hook / Replay / RoPE / Injection / Mapper / Decode Pipeline，不适合最终论文结论。

---

# 15. 单卡运行策略

```text
Teacher Load
→ Teacher Forward / Capture
→ Necessary State to CPU
→ Teacher Unload
→ CUDA Cleanup
→ Student Load
→ Mapper / Inject
→ Decode / Evaluate
```

Teacher 与 Student 不长期同时驻留 GPU。

---

# 16. KV 存储策略

禁止默认保存：

```text
all samples × all layers × all heads × all tokens × K/V
```

优先 Streaming：

```text
Input Batch
→ Teacher Forward
→ Capture Current KV
→ Mapper / Adapter Update
→ Release Raw KV
```

完整 Debug KV：`<=10 samples`。

---

# 17. 最小数据规模

Calibration：

```text
100–500 samples
512–1024 tokens
```

Capability：

```text
Train: 500–2000
Validation: 200–500
Test: 500–1000
```

目标先判断 `CHG > 0 ?`，不是一次性完成最终顶会规模。

---

# 18. 数据集分层

## Fidelity Set

候选：HellaSwag、ARC-Challenge、MMLU 子集、WinoGrande。

目的：Replacement 是否破坏 Student 基础能力。

## Teacher-Advantage Set

只能 Dataset-level 选择 Teacher 稳定优于 Student 的任务。

禁止 Test 后挑 “Teacher 对 / Student 错” 样本。

## Long-Context Set

候选：LongBench、RULER、Multi-document QA、Multi-hop、Needle/Retrieval。

```text
4K → 8K → 16K → 32K
```

最小资源阶段先 8K/16K。

## Behavior-Sensitive Set

Judge / Ranker / Multi-candidate，指标 JCR / KL / Ranking Agreement。

---

# 19. Baseline 体系

- **B0 Student Full Prefill**：绝对 Reference。
- **B1 Teacher Full Inference**：Teacher Gap Reference。
- **B2 Teacher Text/Summary Handoff**：传统文本通信。
- **B3 Cross-Model Ridge**：最关键 Replacement Baseline。
- **B4 Cross-Model MLP**：非线性修复。
- **B5 C2C-style Fusion**：能力增强参考，但 Receiver 有 Self-KV，成本不同。
- **B6 LatentAlign / LCF**：代码可用时公平复现。
- **O1 APCS Base Only**：Compatibility-only。
- **O2 Base + Advantage**：核心方法。
- **O3 Full APCS**：仅 Query Gate 满足复杂度约束时开启。

---

# 20. 实验任务链 T00–T13

## T00 Compatibility Scanner

扫描 tokenizer、layers、heads、kv_heads、head_dim、hidden_size、RoPE、dtype、cache layout、attention implementation。

## T01 Student Self-KV Replay

必须先证明：

```text
Native Student Prefill
≈
External Replay of Student Self-KV
```

FAIL → 后续 Cross-Model 实验全部暂停。

## T02 RoPE Round-trip

```text
K_rope → deRoPE → reRoPE ≈ K_rope
```

## T03 Layer Alignment

比较：Proportional、Last-Layer、Data-driven Top-k、后续 Geometry-aware。

## T04 Ridge Mapper

强 Replacement Baseline。

## T05 Replacement Fidelity

Context：512 / 1K / 2K / 4K。

Gate：

```text
Retention >=90% → PASS
80–90% → CONDITIONAL
<80% → FAIL
```

## T06 Lightweight Mapper

测试 Ridge、Rank 8/16/32、Shared Basis、Head/Layer Sharing。

论文 Fig.1：PCR vs Retention。

## T07 Teacher Gap Freeze

用 Validation 跑 Teacher / Student，冻结 Dataset / Split / Gap Strata。

## T08 Advantage State Training

首轮固定：

```text
Source-Layer Mixer
+ Separate K/V Low-rank
+ Rank 16
+ RMS Calibration
+ Bounded Alpha
+ No Query Gate
```

Student 全冻结。

## T09 Main Capability Experiment

比较 Student、Teacher、Text Handoff、Ridge、Base、Base+Advantage、Full APCS。

核心：CHG / TGRR。

必须：3 Seeds、Full Test、95% CI、Paired Bootstrap/Permutation。

## T10 System Cost

Context：1K / 4K / 8K / 16K，最终 32K。

测：Teacher Prefill、Map、Load/H2D、Query Prefill、Decode、Student Full Prefill、VRAM/RAM。

输出：PSR_A、Scenario B End-to-End Cost、N_BE。

## T11 MVP Decision

只能输出：

```text
A. GO — Runtime Capability Transfer
B. GO — Efficient State Handoff
C. STOP / REDESIGN
```

## T12 Geometry + Behavior Diagnostics

Geometry：Attention-output cosine、Linear CKA、Principal Angle、Effective Rank、Head correlation。

Behavior：Next-token KL、JCR-style、Multi-turn。

## T13 Generalization

顺序：

```text
Second Large→Small Pair
→ Mismatched Head Pilot
→ Cross-Family Pilot
```

---

# 21. 必做消融

1. Rank 8 / 16 / 32
2. Shared Basis
3. Base vs Base + Advantage
4. Teacher Guidance On/Off
5. Self Stability On/Off
6. de-RoPE vs Direct Mapping
7. Layer Selection Strategy
8. Separate K/V vs Shared K/V
9. RMS Calibration On/Off
10. Bounded Alpha vs Unbounded
11. Query Gate On/Off（仅复杂度满足时）

---

# 22. Behavior Stability

单轮：Task Score、Next-token KL、Top-k overlap。

Judge / Ranker：

```text
JCR =
# Handoff decisions matching Student Self-Prefill reference
/
N
```

Multi-turn：1 / 5 / 10 / 20 turns，每轮记录 CHG、KL、JCR、Task Score、Latency。

---

# 23. 统计与学术约束

- 所有训练型 Mapper/Adapter >= 3 Seeds；
- Test 不得用于 Teacher Gap、Rank、Alpha、Top-k、Loss、Teacher Weight、Early Stop、Sample Filter；
- 禁止只报 Teacher-correct / Student-wrong 样本；
- KV R²、Cosine、CKA、Attention Similarity、Retention 都不能代替 CHG；
- Cache Injection 失败后禁止 Silent Re-prefill；
- Scenario B/C 必须计入 Teacher Prefill；
- H2D / Load 必须计入系统成本。

---

# 24. 论文框架

两份论文母稿按 ICLR 2027-style 组织：

1. Abstract
2. Introduction
3. Background and Related Work
4. Problem Formulation
5. Method — APCS
6. Experimental Setup
7. Results
8. Ablation
9. Analysis and Discussion
10. Limitations
11. Conclusion
12. Ethics / Security / Reproducibility
13. References
14. Appendix

中文标题：

**超越缓存仿真：无需重预填充的大模型到小模型优势保持型 KV 状态接力**

英文标题：

**Beyond Cache Emulation: Advantage-Preserving Large-to-Small KV State Handoff without Re-Prefill**

Word 仅是内容母稿；最终投稿需迁移到官方 LaTeX Template。公式要求 Word OMML、可编辑、非图片。

---

# 25. 先验推断数据（NOT MEASURED）

为了线下实验前冻结预期，构造过一组保守模拟值。**不能作为科学证据。**

Qwen3-4B → Qwen3-1.7B / 4K：

```text
Student: 54.8
Teacher: 64.7
```

Ridge：

```text
Handoff ≈ 51.5
Retention ≈ 94.0%
CHG ≈ -3.3 pp
PCR = 1.00
```

Low-rank Base：

```text
Handoff ≈ 50.8
Retention ≈ 92.7%
CHG ≈ -4.0 pp
PCR ≈ 0.23
```

APCS：

```text
Handoff ≈ 56.6
Retention ≈ 103.3%
CHG ≈ +1.8 pp
TGRR ≈ 18.2%
PSR_A ≈ 12%
PCR ≈ 0.27
```

这些是 Prior Projection / NOT Measured。

---

# 26. 先验系统 Scaling（NOT MEASURED）

模拟：

```text
1K:  PSR_A ≈ -86%
4K:  PSR_A ≈ +12%
8K:  PSR_A ≈ +35%
16K: PSR_A ≈ +48%
```

含义：短 Context 下 Map+Load 可能不划算，长 Context 才形成 Pareto。

---

# 27. 先验 Multi-turn（NOT MEASURED）

模拟：

```text
Turn 1:  CHG ≈ +1.8 pp
Turn 20: CHG ≈ +0.6 pp
```

JCR-style：

```text
98.7% → 92.8%
```

KL 随轮次上升。

用途仅为冻结“缓慢漂移”的先验形态。

---

# 28. 审稿报告与论文 V2 修订

审稿总体倾向：**Conditional Accept / Strong Draft**。

主要肯定：

- 问题定义强；
- 学术诚信高；
- Information Injection / Emulation / State Substitution 边界清晰；
- CHG / TGRR / PSR / PCR 指标完备；
- 可证伪 Gate 设计合理；
- 图表与科学问题映射清楚。

主要修改意见：

1. Advantage State 参数化需具体；
2. Teacher Cache 自然存在 vs Teacher 专为 Transfer 运行的成本必须拆开；
3. 最终需要第二 Large→Small Pair；
4. 需要讨论 Mismatched Head / Cross-architecture；
5. Behavior Stability 加 JCR / KL；
6. Geometric Incompatibility 是关键科学风险。

V2 已逐项纳入：

- APCS 具体化；
- Scenario A/B/C；
- N_BE；
- Geometry–Capability；
- JCR/KL；
- G1/G2/G3 泛化层级。

---

# 29. 论文最终需要的图表

## Table 1 — Main Results

至少 2 Large→Small Pairs；Student、Teacher、Text Handoff、Ridge、Low-rank Base、APCS。

指标：Score、Retention、CHG、TGRR、PSR_A、PCR，必要时 KL/JCR。

## Figure 1 — PCR–Retention

证明 Mapper Budget 与 Compatibility 的折中。

## Figure 2 — Teacher Gap–CHG/TGRR

证明或否证 Runtime Advantage 的迁移规律。

## Figure 3 — CHG/TGRR–PSR_A Pareto

仅 Scenario A；右上象限才是论文目标。

## Figure 4 — Context Length Scaling

绝对时延分解。

## Figure 5 — Multi-turn Stability

同时 CHG、KL、JCR。

## Figure 6 — Ablation

## Figure 7（建议新增）— Geometry–Capability Map

可含 CKA Heatmap、Principal Angle vs CHG、Attention-output cosine vs Retention、Effective Rank vs CHG。

---

# 30. 研究结果可能走向的四条路径

## Path A — Runtime Capability Transfer

```text
Retention high
CHG > 0
TGRR > 0
PSR_A > 0
Behavior stable
Geometry interpretable
Second Pair replicates
```

主张：**Runtime Capability Transfer via KV State Handoff**。

## Path B — Efficient State Handoff

```text
Retention high
CHG <= 0
PCR low
PSR_A high
```

转：**Efficient Cross-Model State Handoff**。

## Path C — Mechanism Boundary

```text
Retention high
CHG <= 0
Geometry mismatch explains failure
```

转：**State Compatibility Does Not Imply Capability Compatibility**。

## Path D — Stop / Redesign

Replacement 不稳定，则优先研究 Pair Asymmetry、Layer/Head Alignment、Cache Geometry。

---

# 31. 已生成的主要文件

## 研究方案 V1.2

`跨模型KV_Cache认知状态迁移研究方案_V1.2_顶会审稿校准版.docx`

特点：23页；Runtime Capability Transfer；APCS；三项 Contribution；双论文路线；4–6周 Sprint。

## 最小资源研究包 V1.3

单文件：

`跨模型KV_Handoff_最小资源研究方案_V1.3.md`

ZIP：

`跨模型KV_Handoff_最小资源研究包_V1.3.zip`

包含 README.md、RESEARCH_PLAN.md、AGENTS.md、TASKS.md、EXPERIMENT_SPEC.md。

## PDF 详细方案 V1.4

`跨模型KV_Cache运行时能力迁移详细研究与智能体实施方案_V1.4.pdf`

定位：研究立项方案 + 实验手册 + Agent 工程规范。

## 中文论文 Draft

- `论文框架_中文_ICLR2027风格_KV_Runtime_Capability_Transfer.docx`
- `论文Draft_中文_ICLR2027风格_KV_Runtime_Capability_Transfer_先验推演版.docx`
- `论文Draft_中文_ICLR2027风格_KV_Runtime_Capability_Transfer_审查优化版_V2.docx`

## 英文论文 Draft

- `Paper_Draft_English_ICLR2027_Style_KV_Runtime_Capability_Transfer.docx`
- `Paper_Draft_English_ICLR2027_Style_KV_Runtime_Capability_Transfer_Prior_Projection.docx`
- `Paper_Draft_English_ICLR2027_KV_Runtime_Capability_Transfer_Review_Optimized_V2.docx`

## 先验模拟数据

`推断实验数据_先验推演_NotMeasured.json`

---

# 32. 资源有限时实验优先级

## P0 — 绝对必须

T00、T01、T02、T04、T05、T07、T08、T09；Student/Teacher/Ridge/APCS；3 Seeds；Full Test；CHG/TGRR。

## P1 — 顶会强烈需要

T06、T10、T12；KL/JCR；8K/16K；Full Ablation；Second Pair；Geometry。

## P2 — 有资源再做

MLP、完整 C2C/LCF、32K、Mismatched Head、Cross-Family、Query Gate、NVMe Serving。

---

# 33. 6 周实验节奏

## Week 1

T00–T03：Compatibility、Self-KV Replay、RoPE、Layer Alignment。

## Week 2

T04–T06：Ridge、Low-rank、PCR–Retention。

## Week 3

T07–T08：Teacher Gap Freeze、APCS Advantage Training。

## Week 4

T09：Full Test、3 Seeds、CHG/TGRR、Main Table、Fig.2。

这是第一次真正的科学 Go/No-Go。

## Week 5

若 CHG 有信号：T10，做 8K/16K、Scenario A/B/C、Pareto、Context Scaling。

## Week 6

T12 + Ablation：Geometry、KL、JCR、Multi-turn、Fig.5/6/7。

---

# 34. 更高创新方向

用户进一步讨论了：如果想让该方向更具创新性，应从“更好的 Mapper”升级为“Runtime Capability Engineering”。

重点方向如下。

---

# 35. 创新 1：Causal Advantage Subspace Transfer

这是**当前最推荐的下一步**。

当前 APCS：

```text
C*_S = C_base + R_adv
```

Reviewer 会进一步问：

> 如何证明 `R_adv` 真的是 Teacher Advantage，而不是碰巧提升 Benchmark 的 Residual？

提出：

**Causal Advantage Subspace (CAS)**

设：

```text
C_T = C_shared + C_adv + C_nuisance
```

- `C_shared`：Teacher/Student 共有；
- `C_adv`：Teacher-exclusive + task-causal + student-consumable；
- `C_nuisance`：无关或 Student 无法消费。

可定义：

```text
CCS(z) =
Score(S | C_base + z)
-
Score(S | C_base)
```

目标升级为：

```text
Teacher KV
→ Shared State
+ Causal Advantage Subspace
```

它与当前 T12 Geometry Diagnostics 高度衔接，创新强度最高、资源增量相对可控。

---

# 36. 创新 2：Universal Runtime State Codec

Pairwise Mapper 存在潜在 `O(N^2)` 问题。

更大的问题：能否构造模型无关 Runtime State IR？

```text
Qwen KV
Llama KV
Mistral KV
      ↓
Universal Runtime State
      ↓
Receiver-specific Decoder
```

```text
Z_runtime = E_T(C_T)
C*_S = D_S(Z_runtime)
```

希望：

```text
I(Z; Task) ↑
I(Z; ModelID) ↓
```

从 Cache Translation 升级到 Model-independent Runtime Representation。

---

# 37. 创新 3：Capability Capsule

如果 Teacher Advantage 只集中在少量 Layer/Head/低秩 Subspace，则不需要传完整 KV。

构造：

**Runtime Capability Capsule**

```text
Teacher(X)
→ Capability Capsule
→ Store
→ Student A / B / C
```

Capsule 可包含 Semantic Memory、Reasoning State、Task-specific Advantage、Confidence、Applicable Query Domain。

把 KV Cache 升级为可持久化的 Runtime Capability Artifact。

---

# 38. 创新 4：Composable Capability Capsules

研究：

```text
Capsule_A + Capsule_B
```

是否可组合。

例如：Legal Capsule + Financial Capsule → Combined Capability。

可研究 Addition、Subtraction、Intersection、Composition，形成 Runtime State Algebra。

---

# 39. 创新 5：Capability Rate–Distortion

研究：恢复多少 Teacher Advantage，需要多少 Runtime State 信息？

```text
R = Bits(Z_adv)
D = TeacherScore - HandoffScore
```

研究 `R(D)`。

或：

```text
R*(tau) =
min Size(Z)
s.t. TGRR(Z) >= tau
```

候选独立论文：

**How Many Bits Does Runtime Capability Need?**

---

# 40. 创新 6：Multi-Teacher State Composition

多个 Teacher：Math / Code / Long-context，各自输出 `Z1/Z2/Z3`。

Student：

```text
C*_S = F(Z1,Z2,Z3)
```

可发展为：

**Runtime Mixture-of-Experts without Expert Decoding / State-space MoE**。

---

# 41. 创新 7：Capability-Aware Handoff Router

不是所有 Query 都应 Transfer。

Router 输入：

- Teacher Gap Prediction
- Geometry Compatibility
- Query Type
- Context Length
- PSR Prediction
- Student Uncertainty

输出：

```text
STATE TRANSFER
STUDENT RE-PREFILL
TEXT HANDOFF
TEACHER DECODE
```

优化：`Quality - lambda * Cost`。

---

# 42. 创新 8：Selective Re-Prefill

科学主论文要求 Zero Student X-Prefill，但系统最优可能是：

```text
95% transferred state + 5% selective recomputation
```

研究最小 Layer/Head/Token 重算预算，使 `CHG > 0`。

这是很强的系统论文扩展。

---

# 43. 创新 9：Editable / Reversible Runtime Capability

Runtime State 可进一步研究 Add / Remove / Modify。

例如：

```text
Teacher State
- Harmful State
+ Useful Capability
→ Student
```

形成：

**Safe Capability Transplantation**。

目标是迁移有用能力而不迁移错误、偏见、Prompt Injection 或其他不希望行为。

---

# 44. 创新 10：Runtime Capability OS

长期终极方向：

模型不再只是：

```text
Weights + Prompt
```

而是：

```text
Model = Weights + Context + Runtime Capability
```

形成 Capability Store → Runtime State IR → 多种 Student Model 的体系。

最终概念：**Runtime Capability OS**。

---

# 45. 推荐长期演进链

```text
KV Handoff
    ↓
Advantage-Preserving Handoff
    ↓
Causal Advantage Subspace
    ↓
Capability Capsule
    ↓
Universal Runtime State
    ↓
Runtime Capability OS
```

当前最推荐的下一步：**Causal Advantage Subspace Transfer**。

理由：

1. 与现有 APCS 只有一步距离；
2. 不要求立刻扩大模型；
3. 可直接利用 T12 Geometry 数据；
4. 能回答“Teacher Advantage 到底在哪里、是否 task-causal、是否 Student-consumable”；
5. 即使 CHG 不稳定，也能产出机制结论。

建议：Paper 1 保持 APCS 聚焦；线下实验同时记录 Layer/Head/Geometry/Intervention 数据，为 CAS 下一篇论文准备。

---

# 46. 当前研究最重要的科学判断

这个方向真正重要的问题不是：

> 能不能把一个 KV Tensor 映射成另一个 KV Tensor？

而是：

> **Context-conditioned Runtime State 中是否存在可跨模型迁移的能力成分？**

如果答案为 Yes：

- 模型能力的一部分可以在运行时形成；
- 可以被保存、压缩、迁移、组合；
- 可能形成新的模型间协作与 Serving 范式。

如果答案为 No：

也可以形成重要科学边界：

> **State Compatibility 不代表 Capability Compatibility。**

---

# 47. 后续 Session 恢复上下文时最重要的 10 点

1. 当前主论文：APCS / Advantage-Preserving Handoff。
2. 当前主模型 Pair：Qwen3-4B → Qwen3-1.7B。
3. 当前首要 Gate：Self-KV Replay → Ridge Replacement → CHG。
4. 所有先验数字均 NOT MEASURED。
5. 正式论文最终需要第二 Large→Small Pair。
6. Geometry / JCR / KL 已升级为强制实验。
7. Scenario A/B/C 成本必须严格拆开。
8. 当前最推荐的下一创新：Causal Advantage Subspace。
9. 若 CHG≤0 但 Retention 高，优先研究 State Compatibility ≠ Capability Compatibility，不要继续盲目堆模块。
10. 所有后续科学结论必须由真实实验决定，禁止反向修改 Test 目标。

---

# 48. 一句话 Session 总结

> 本 Session 已把“让小模型加载大模型 KV Cache”的初始设想，发展为严格的 **Runtime Capability Transfer** 研究路线：以 APCS 实现 Large→Small、No Student X-Prefill 的 State Substitution，以 CHG/TGRR/PSR_A/PCR 衡量能力与系统价值，以 Geometry、JCR、KL 判断状态是否真正可消费，并进一步提出 Causal Advantage Subspace、Capability Capsule、Universal Runtime State 与 Runtime Capability OS 等更高层创新方向。
