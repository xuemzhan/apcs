# 跨模型 KV Cache 运行时能力迁移实验方案 V2.1

## ——面向 ICLR 2027 论文证据链的实验执行版

> **对应论文**：Beyond Cache Emulation: Advantage-Preserving Large-to-Small KV State Handoff without Re-Prefill
> **核心方法**：Advantage-Preserving Cache Synthesis（APCS）
> **当前主模型对**：Qwen3-4B → Qwen3-1.7B
> **最低推荐资源**：1×24GB NVIDIA GPU + 64–128GB RAM + 1–2TB NVMe
> **当前研究阶段**：Matched-KV / Same-family 核心机制验证
> **最终投稿要求**：至少两个 Large→Small Pair，优先覆盖两个 Model Family
> **首要科学指标**：CHG
> **首要系统指标**：PSR_A
> **核心原则**：先证明工程正确，再证明 Replacement，再讨论 Capability Transfer。

---

# 1. 实验总体目标

本实验不以“成功实现跨模型 KV 映射”为最终目标。

需要依次回答六个问题：

1. **工程正确性**：外部 KV 是否真正被 Student 正确消费？
2. **Replacement Fidelity**：Teacher KV 转换后是否能够替代 Student Self-Prefill？
3. **Capability Transfer**：Student 是否能够超过自己的 Self-Prefill Baseline？
4. **Geometry–Capability Coupling**：为什么某些 Teacher 状态能够或不能被 Student 消费？
5. **Behavior Stability**：能力增益是否以行为漂移为代价？
6. **System Cost**：在哪一种部署场景和 Context Length 下，Handoff 才真正节省计算？

研究逻辑必须严格按照：

```text
Engineering Correctness
        ↓
Replacement
        ↓
Capability Transfer
        ↓
Behavior Stability
        ↓
Geometry Mechanism
        ↓
System Cost
        ↓
Generality
```

禁止在前一层证据没有成立时直接讨论后一层结论。

---

# 2. 核心科学问题

给定长上下文：

[
X
]

Teacher：

[
T
]

Student：

[
S
]

以及后续 Query：

[
q
]

Teacher 已经处理：

[
X
]

并得到：

[
C_T(X)
]

本文要求 Student 不重新 Prefill 原上下文：

[
Prefill_S(X)=0
]

而是通过：

[
C_T(X)
\rightarrow
C_S^*(X)
]

然后执行：

[
\hat y=S(q;C_S^*(X))
]

标准 Student Baseline 为：

[
C_S(X)=Prefill_S(X)
]

以及：

[
y_S=S(q;C_S(X))
]

最终研究的不是：

[
C_S^*\approx C_S
]

而是：

[
Score(S|C_S^*)>
Score(S|C_S)
]

---

# 3. 四个论文核心指标

## 3.1 Retention

用于判断 Replacement Fidelity：

[
Retention=
\frac{Score_{handoff}}
{Score_{student-self}}
]

Retention 高只能说明：

> Handoff State 可以近似替代 Student State。

不能说明 Teacher Advantage 已经迁移。

---

## 3.2 CHG：Capability Handoff Gain

[
CHG=
Score_{handoff}
---------------

Score_{student-self}
]

这是论文最重要的科学端点。

只有：

[
CHG>0
]

才能讨论 Runtime Capability Transfer。

---

## 3.3 TGRR：Teacher Gap Recovery Rate

[
TGRR=
\frac{
Score_{handoff}-Score_{student}
}{
Score_{teacher}-Score_{student}
}
]

用于回答：

> Teacher 与 Student 原来的能力差距，有多少通过 Runtime State 被 Student 恢复？

---

## 3.4 PCR：Parameter Compression Ratio

[
PCR=
\frac{MapperSize_{ours}}
{MapperSize_{ridge}}
]

用于评价 APCS / Lightweight Mapper 的部署规模。

---

# 4. 系统收益必须严格区分三种场景

这是本轮论文修改后必须严格执行的实验约束。

---

## 4.1 Scenario A：Natural Handoff

Teacher 因业务本来就已经处理过 `X`。

例如：

```text
Teacher负责前期复杂分析
        ↓
已经得到C_T(X)
        ↓
系统因成本/角色切换到Student
```

此时 Teacher Prefill 属于沉没成本。

定义：

[
PSR_A=
1-
\frac{
T_{map}+T_{load}+T_{query}
}{
T_{prefill,S}(X)
}
]

论文的：

```text
CHG–PSR Pareto
```

主图只能使用这一口径。

---

## 4.2 Scenario B：Teacher-for-Transfer

Teacher 本来不需要运行，只是为了产生 Cache 而执行。

此时：

[
Cost_B=
T_{prefill,T}
+
T_{map}
+
T_{load}
+
T_{query}
]

必须与：

[
T_{prefill,S}
]

比较。

禁止使用：

```text
PSR_A
```

声称 Scenario B 具有端到端收益。

---

## 4.3 Scenario C：One-Teacher-Many-Student

一次 Teacher Prefill 被多个 Student 或多个 Query 复用。

Baseline：

[
Cost_{baseline}(N)
==================

N T_{prefill,S}
]

Handoff：

[
Cost_{handoff}(N)
=================

T_{prefill,T}
+
T_{map}
+
N(T_{load}+T_{query})
]

定义：

[
N_{BE}
======

\min
\left{
N:
Cost_{handoff}(N)<Cost_{baseline}(N)
\right}
]

必须通过真实硬件实验得到：

```text
N_BE
```

而不是理论估算后直接作为论文结果。

---

# 5. 研究假设与 Gate

## H0：外部 KV Replay 正确

Student 自己产生的 KV：

```text
Native Self-KV
```

被保存、重新注入后，必须与 Native Inference 基本等价。

### Gate 0

若 Self-KV Replay 失败：

> 禁止进行 Cross-Model Mapper 实验。

---

# 6. H1：Replacement 可成立

Teacher KV 经 Mapper：

[
C_T(X)
\rightarrow
\hat C_S(X)
]

Student 不重新处理 `X`。

期望：

```text
Retention ≥ 90%
```

强目标：

```text
Retention ≥ 95%
```

### Gate 1

```text
Retention ≥90%
    → PASS

80%–90%
    → CONDITIONAL

<80%
    → FAIL
```

如果 `<80%`：

优先解决：

* Cache Layout；
* RoPE；
* Layer Alignment；
* Mapping；
* 当前 Model Pair；

而不是继续做 Advantage State。

---

# 7. H2：Teacher Advantage 可以迁移

Compatibility Base 建立后：

[
C_S^*
=====

C_{base}
+
R_{adv}
]

如果：

[
CHG>0
]

则说明至少部分 Teacher Advantage 能被冻结的 Student 消费。

### Gate 2A

进入 Runtime Capability Transfer 的最低条件：

* Full Test `CHG>0`；
* 至少 3 Seeds；
* 95% CI 支持正向趋势；
* TGRR>0；
* 无明显行为失稳；
* Scenario A 下 PSR_A>0。

---

# 8. H3：State Compatibility 不一定意味着 Capability Compatibility

可能出现：

```text
Retention = 95%
CHG = -1.5pp
```

即：

> Student 能使用 Cache，但无法利用 Teacher 特有能力。

这时必须启动 Geometry Diagnostics。

如果：

* Replacement 高；
* CHG≤0；
* Deep Layer Geometry 明显不兼容；

可形成：

> **State Compatibility Does Not Imply Capability Compatibility**

这一独立科学结论。

---

# 9. H4：Behavior Stability

即便：

[
CHG>0
]

也必须验证：

* Token Distribution；
* Judge Decision；
* Ranking；
* Refusal；
* Multi-turn behavior；

没有严重漂移。

因此：

> Accuracy / CHG 不能作为唯一质量指标。

---

# 10. 模型实验层级

## G1：Same-Family Matched-KV

当前主实验：

```text
Teacher: Qwen3-4B
Student: Qwen3-1.7B
```

这是最小资源科学验证 Pair。

Context：

```text
512
1K
2K
4K
```

核心信号成立后：

```text
8K
16K
32K
```

---

# 11. 第二模型 Pair

最终论文必须至少增加一组独立 Large→Small Pair。

优先：

> 第二模型 Family。

选择要求：

* Open Weight；
* 能获取 `past_key_values`；
* 可 Hook Attention；
* Tokenizer / Position 可审计；
* GPU 资源可承受。

第二 Pair 的主要目的不是提高样本数量，而是回答：

> APCS 是否只是 Qwen3-4B→1.7B 的特例？

---

# 12. G2：Mismatched Head / Dimension

核心 Matched-KV 实验完成后再做。

例如：

[
H_T\ne H_S
]

或：

[
d_{head,T}\ne d_{head,S}
]

增加：

[
P_H
]

Head Projection，以及：

[
P_d
]

Dimension Projection。

G2：

* 属于泛化实验；
* 不作为核心机制成立的前置条件。

---

# 13. G3：Cross-Family

只有完成：

* Tokenizer Audit；
* Position Audit；
* RoPE Audit；
* Cache Layout Audit；
* Prompt Semantics Audit；

后才执行。

资源有限时属于 Optional Pilot。

---

# 14. 数据集分层

不能使用同一个 Benchmark 同时回答所有问题。

至少分四类。

---

# 15. Fidelity Set

目的：

> Handoff 是否破坏 Student 基础能力？

候选：

* HellaSwag；
* ARC-Challenge；
* MMLU 子集；
* WinoGrande。

输出：

* Score；
* Retention；
* KL；
* Token Agreement。

---

# 16. Teacher-Advantage Set

目的：

> Teacher 相对 Student 的优势能否迁移？

允许：

> 在 Dataset-level 上选择 Teacher 稳定优于 Student 的任务。

禁止：

> Test 后选择“Teacher 对 / Student 错”的题。

必须严格分：

```text
Train
Validation
Test
```

其中：

Train：

* 计算 Teacher Margin；
* Teacher Guidance Weight。

Validation：

* Rank；
* α_max；
* Top-k；
* Loss Weight。

Test：

> 完全冻结。

---

# 17. Long-Context Set

建议：

* RULER 类；
* LongBench 类；
* Multi-document QA；
* Multi-hop QA；
* Retrieval / Needle。

长度：

```text
4K
8K
16K
32K
```

最小资源阶段先做到：

```text
8K / 16K
```

即可。

---

# 18. Behavior-Sensitive Set

单独设置：

* LLM Judge；
* Ranker；
* Multi-candidate comparison。

目的：

> 检查 Accuracy 不变情况下 Decision Behavior 是否变化。

---

# 19. Baselines

## B0：Student Full Prefill

绝对 Reference。

---

## B1：Teacher Full Inference

用于：

* Teacher Gap；
* TGRR；
* Performance Reference。

---

## B2：Teacher Text/Summary Handoff

用于比较 Runtime State 与自然语言通信。

---

## B3：Cross-Model Ridge

最重要 Replacement Baseline。

必须实现：

* Top-k Teacher Layers；
* de-RoPE Key；
* per-head；
* K/V Separate Mapping。

---

## B4：Cross-Model MLP

只有 Ridge 明显不足时使用。

---

## B5：C2C-style

属于 Information Injection。

Receiver 已有：

[
C_S(X)
]

因此不能与 APCS 的 Zero Student X-Prefill 成本直接等价比较。

---

## B6：LatentAlign / LCF

代码可用时尽量复现。

无法公平复现时：

> 明确说明比较边界。

---

## O1：APCS Base Only

仅 Compatibility。

---

## O2：Base + Advantage State

最重要实验组。

---

## O3：Full APCS

只有 Query Gate 被证明不会重新扫描 `X` 时才开启。

---

# 20. Compatibility Base

定义：

[
C_{base}
========

F_{compat}(C_T)
]

首先实现：

### Stage 1

Full Ridge。

### Stage 2

Low-rank：

```text
r = 8
r = 16
r = 32
```

可选：

```text
r = 64
```

### Stage 3

Shared Basis。

### Stage 4

Head / Layer Sharing。

---

# 21. Source-Layer Mixer

对于 Student Layer：

[
l
]

选择 Teacher Layers：

[
\mathcal A(l)
]

然后：

[
Z_{l,h}
=======

\sum_{i\in\mathcal A(l)}
w_{l,i,h}
C_T^{i,h}
]

其中：

[
\sum_iw_{l,i,h}=1
]

Layer Alignment 至少比较：

1. Proportional；
2. Last-Layer；
3. Data-driven Top-k；
4. Geometry-aware Top-k。

---

# 22. Advantage Residual

K 和 V 必须独立参数化。

Key：

[
R^K_{l,h}
=========

A^K_{l,h}
\sigma
\left(
B^K_{l,h}Z^K_{l,h}
\right)
]

Value：

[
R^V_{l,h}
=========

A^V_{l,h}
\sigma
\left(
B^V_{l,h}Z^V_{l,h}
\right)
]

Rank：

```text
8 / 16 / 32
```

默认：

```text
16
```

---

# 23. de-RoPE 强制实验

Key 必须执行：

```text
Teacher K
↓
de-RoPE
↓
Mapper / Residual
↓
Student RoPE
```

必须设置 Ablation：

```text
de-RoPE Mapping
vs
Direct RoPE-space Mapping
```

这是核心工程正确性验证之一。

---

# 24. Residual RMS Calibration

需要分别计算：

[
RMS(C_{base})
]

和：

[
RMS(R_{adv})
]

限制 Advantage State 不能通过简单放大 Norm 获得表面 CHG。

同时记录：

* Layer Residual RMS；
* Base RMS；
* Ratio；
* CHG；
* KL。

---

# 25. Bounded Layer-wise Injection

使用：

[
\alpha_l
========

\alpha_{max}\tanh(a_l)
]

最终：

[
C_S^*
=====

C_{base}
+
\alpha_lR_{adv}
]

`α_max`：

* Validation 决定；
* Test 前冻结。

---

# 26. Query Gate

首轮：

> 默认关闭。

若启用：

只能：

```text
q
→ lightweight encoder
→ layer/head gate
```

禁止：

```text
q
→ scan all C_T(X)
```

Per-query complexity 不得与：

[
|X|
]

线性增长。

---

# 27. Loss 设计

完整目标：

[
L
=

\lambda_{task}L_{task}
+
\lambda_{self}L_{self}
+
\lambda_{teacher}wL_{teacher}
+
\lambda_{att}L_{att}
+
\lambda_{reg}L_{reg}
]

---

## L_task

直接 Task/Suffix Loss。

---

## L_self

用于防止 Teacher Advantage 注入导致 Student 稳定行为被破坏。

---

## L_teacher

Teacher Guidance。

权重：

[
w
]

只能由 Train Split Teacher–Student Margin 计算。

---

## L_att

Attention Output Alignment。

优先于：

> 只优化 Raw KV MSE。

---

## L_reg

控制：

* Rank；
* Residual Norm；
* α；
* Mapper Size。

---

# 28. T00：Compatibility Scanner

自动读取：

```text
tokenizer
vocab
layers
attention_heads
kv_heads
head_dim
hidden_size
rope
position
dtype
cache_layout
attention_implementation
```

输出：

```text
model_compatibility.json
```

---

# 29. T01：Self-KV Replay

步骤：

```text
Student Prefill(X)
↓
save C_S(X)
↓
Native Decode(q)
↓
clear
↓
inject C_S(X)
↓
q only
↓
Decode
```

比较：

* Logit Cosine；
* Max Error；
* Token Agreement。

如果不一致：

> 停止后续所有实验。

---

# 30. T02：RoPE Round-trip

验证：

[
K_{rope}
\rightarrow
deRoPE
\rightarrow
reRoPE
\approx
K_{rope}
]

至少抽样：

* 3 Layers；
* 3 Heads；
* 128 Tokens。

---

# 31. T03：Layer Alignment

构建：

```text
Teacher Layer
→
Student Layer
```

输出：

* Layer Similarity Matrix；
* Top-k Mapping；
* Attention-output Similarity。

---

# 32. T04：Ridge Baseline

Calibration：

```text
100–500 samples
```

Context：

```text
512 / 1K
```

记录：

* Mapper Parameters；
* Size；
* Train Time；
* Map Time；
* R²；
* Cosine；
* Attention-output Cosine。

---

# 33. T05：Replacement

Context：

```text
512
1K
2K
4K
```

方法：

* Student；
* Teacher；
* Ridge。

输出：

* Score；
* Retention；
* KL；
* Token Agreement；
* Latency。

Gate：

```text
≥90% PASS
80–90% CONDITIONAL
<80% FAIL
```

---

# 34. T06：Lightweight Mapper

测试：

```text
Ridge
Rank-8
Rank-16
Rank-32
Shared Basis
```

输出论文：

> Figure 1 — PCR vs Retention。

---

# 35. T07：Teacher Gap Freeze

先运行 Teacher 与 Student 的 Validation。

冻结：

* Dataset；
* Task；
* Split；
* Gap Strata。

此后禁止根据 Test 修改。

---

# 36. T08：Advantage State Training

第一版固定：

```text
Source-Layer Mixer
+
Separate K/V Low-rank
+
Rank 16
+
RMS Calibration
+
Bounded α
+
No Query Gate
```

Student：

> 全冻结。

---

# 37. T09：Main Capability Experiment

必须比较：

```text
Student
Teacher
Text Handoff
Ridge
Base Only
Base + Advantage
Full APCS
```

核心：

[
CHG
]

和：

[
TGRR
]

统计：

```text
3 Seeds
Mean
Std
95% CI
Paired Bootstrap / Permutation
```

---

# 38. T10：System Cost

至少：

```text
1K
4K
8K
16K
```

最终：

```text
32K
```

测量：

```text
teacher_prefill
map
H2D/load
query_prefill
decode
student_full_prefill
VRAM
RAM
cache_bytes
```

分别计算：

* Scenario A：PSR_A；
* Scenario B：Full Cost；
* Scenario C：N_BE。

---

# 39. T11：MVP Decision

只能输出三类结论：

### A

```text
GO — Runtime Capability Transfer
```

### B

```text
GO — Efficient State Handoff
```

### C

```text
STOP / REDESIGN
```

---

# 40. T12：Geometry Diagnostics

这是最新版论文新增的强制实验。

每个 Layer / Head：

* Attention-output Cosine；
* Linear CKA；
* Principal Subspace Angle；
* Effective Rank；
* Head Correlation。

然后分析：

```text
Geometry ↔ Retention
Geometry ↔ CHG
```

---

# 41. Geometry Type I

```text
Numerical similarity low
Task fidelity high
```

说明：

> KV numerical reconstruction 不是必要条件。

---

# 42. Geometry Type II

```text
Retention high
CHG ≤ 0
Deep geometry poor
```

支持：

> State Compatibility ≠ Capability Compatibility。

---

# 43. Geometry Type III

```text
CHG > 0
Aligned local subspace
```

支持：

> Teacher Advantage 主要存在于 Student 可消费的局部子空间。

---

# 44. T13：Generalization

顺序：

```text
Second Model Pair
↓
Mismatched Head Pilot
↓
Cross-family Pilot
```

禁止在第一 Pair 未通过 Gate 2 时大量投入 G2/G3。

---

# 45. JCR-style Behavior Metric

以 Student Self-Prefill 为 Reference。

定义：

[
JCR=
\frac{
\text{Handoff 与 Self-Prefill 做出相同决策的次数}
}{
N
}
]

用于：

* Judge；
* Ranker；
* Multi-choice preference。

---

# 46. Multi-turn

建议：

```text
1
5
10
20
```

轮。

每轮记录：

* CHG；
* KL；
* JCR；
* Task Score；
* Latency。

最终：

> Figure 5。

---

# 47. 必做消融

## A1 Rank

```text
8 / 16 / 32
```

---

## A2 Shared Basis

```text
On / Off
```

---

## A3 Advantage State

```text
Base
vs
Base + Advantage
```

最关键消融。

---

## A4 Teacher Guidance

```text
With / Without
```

---

## A5 Self Stability

重点同时看：

```text
CHG
KL
JCR
```

---

## A6 de-RoPE

```text
Correct
vs
Direct Mapping
```

---

## A7 Layer Selection

```text
Proportional
Last-layer
Data-driven
Geometry-aware
```

---

## A8 K/V Adapter

```text
Separate
vs
Shared
```

---

## A9 RMS Calibration

```text
On / Off
```

---

## A10 Bounded Alpha

```text
Bounded
vs
Unbounded
```

---

## A11 Query Gate

仅证明不扫描 `X` 后执行。

---

# 48. Teacher Gap 分析

定义：

[
Gap=
Score_T-Score_S
]

使用 Validation 预先定义：

```text
Low
Medium
High
```

Gap。

每个区间报告：

* N；
* Gap；
* CHG；
* TGRR；
* CI。

但始终同时报告：

> Full Test Overall。

---

# 49. 系统计时规范

固定：

* Hardware；
* CUDA；
* Torch；
* Transformers；
* Attention Implementation；
* Dtype；
* Batch；
* Context；
* Residency。

计时：

1. Warm-up；
2. `torch.cuda.synchronize()`；
3. ≥10 次正式重复；
4. P50；
5. P95。

禁止只报告：

> Matrix Multiplication Time。

---

# 50. Cache Residency

必须区分：

### R0

GPU-resident。

### R1

CPU-resident。

### R2

NVMe / Persistent。

主最小资源实验主要是：

> R1。

---

# 51. 统计约束

训练型实验至少：

```text
3 Seeds
```

核心结果：

* Mean；
* Std；
* 95% CI；
* Paired Test。

多 Benchmark：

* 单任务；
* Macro Average；
* Positive / Negative Task Count。

---

# 52. 严格禁止事项

## 禁止 1

Student 在主实验中重新读取 `X`。

---

## 禁止 2

微调 Student 主体后仍称 Runtime State Transfer。

---

## 禁止 3

Test 调参。

---

## 禁止 4

只挑 Teacher-win Test Sample。

---

## 禁止 5

隐藏 Teacher Prefill 成本。

---

## 禁止 6

隐藏 H2D / Cache Load。

---

## 禁止 7

用：

* R²；
* Cosine；
* CKA；

代替 CHG。

---

## 禁止 8

Cache 注入失败后 Silent Re-prefill。

---

# 53. 单卡执行策略

```text
Teacher Load
↓
Forward
↓
Capture
↓
CPU Offload
↓
Teacher Unload
↓
CUDA Cleanup
↓
Student Load
↓
Map
↓
Inject
↓
Decode
```

---

# 54. KV 存储策略

禁止：

```text
Dataset
× All Layers
× All Heads
× All Tokens
```

全部保存。

采用：

```text
Streaming
```

完整 Debug KV：

```text
≤10 samples
```

---

# 55. 主论文 Table 1

最终至少包含：

```text
2 Large→Small Pairs
```

方法：

* Student；
* Teacher；
* Text；
* Ridge；
* Low-rank Base；
* APCS。

指标：

* Score；
* Retention；
* CHG；
* TGRR；
* PSR_A；
* PCR。

必要时增加：

* KL；
* JCR。

---

# 56. Figure 1

## PCR–Retention

回答：

> Mapper 可以压缩多少而不明显破坏 Replacement？

来源：

```text
T04–T06
```

---

# 57. Figure 2

## Teacher Gap–CHG/TGRR

回答：

> Teacher 越强，运行时 Advantage 是否越容易迁移？

来源：

```text
T07/T09
```

---

# 58. Figure 3

## CHG–PSR_A Pareto

仅 Scenario A。

画：

```text
CHG = 0
PSR_A = 0
```

两条轴线。

只有：

> 右上象限

才支持论文目标。

---

# 59. Figure 4

## Context Length Scaling

至少：

```text
1K
4K
8K
16K
```

报告：

* Student Prefill；
* Map；
* Load；
* Query；
* PSR_A。

---

# 60. Figure 5

## Multi-turn Stability

至少同时画：

* CHG；
* KL；
* JCR。

---

# 61. Figure 6

## Ablation

必须突出：

* Advantage State；
* Teacher Guidance；
* Self Stability；
* de-RoPE；
* RMS Calibration；
* Bounded α。

---

# 62. Figure 7——建议新增

## Geometry–Capability

推荐形式：

### Fig.7a

Layer × Layer CKA。

### Fig.7b

Principal Angle × CHG。

### Fig.7c

Attention-output Cosine × Retention。

### Fig.7d

Effective Rank × CHG。

如果结果明显，这张图可能成为论文机制分析中最有价值的一张图。

---

# 63. Run 数据规范

每个 Run：

```text
reports/runs/<run_id>/
├── config.yaml
├── metrics.json
├── system.json
├── geometry.json
├── stdout.log
└── summary.md
```

---

# 64. config.yaml 必须记录

```text
teacher_model
teacher_commit

student_model
student_commit

tokenizer

dtype
attention_implementation

rope_config
cache_layout

context_length
dataset
split

seed

mapper
rank

source_top_k
alpha_max

loss_weights

hardware
cache_residency
```

---

# 65. metrics.json

建议字段：

```json
{
  "student_score": null,
  "teacher_score": null,
  "handoff_score": null,

  "retention": null,
  "chg": null,
  "tgrr": null,
  "pcr": null,
  "psr_a": null,

  "next_token_kl": null,
  "jcr": null,

  "attention_output_cosine": null,
  "linear_cka": null,
  "principal_angle": null,
  "effective_rank": null
}
```

---

# 66. 推荐实验矩阵

| ID  | Method         | Context | Seeds | Purpose      |
| --- | -------------- | ------: | ----: | ------------ |
| E00 | Self-KV Replay |     512 |     1 | Injection    |
| E01 | RoPE           |     512 |     1 | Position     |
| E02 | Ridge          |     512 |     1 | Feasibility  |
| E03 | Ridge          |      1K |     1 | Replacement  |
| E04 | Ridge          |      4K |     1 | Replacement  |
| E05 | Rank 8         |   1K/4K |     3 | PCR          |
| E06 | Rank 16        |   1K/4K |     3 | PCR          |
| E07 | Rank 32        |   1K/4K |     3 | PCR          |
| E08 | Advantage      |      1K |     3 | CHG Probe    |
| E09 | APCS           |      4K |     3 | Main CHG     |
| E10 | APCS           |      8K |     3 | Long Context |
| E11 | APCS           |     16K |     3 | PSR          |
| E12 | Multi-turn     |   4K/8K |     3 | Stability    |
| E13 | Geometry       |    Best |     3 | Mechanism    |
| E14 | Scenario B/C   |  4K–16K |    ≥1 | N_BE         |
| E15 | Second Pair    |   4K/8K |     3 | Generality   |

---

# 67. 资源有限时优先级

## P0：绝对必须

* T00；
* T01；
* T02；
* T04；
* T05；
* T07；
* T08；
* T09；
* Student；
* Teacher；
* Ridge；
* APCS；
* 3 Seeds；
* Full Test；
* CHG；
* TGRR。

---

## P1：顶会论文强烈需要

* Low-rank；
* PCR；
* 8K/16K；
* PSR_A；
* KL；
* JCR；
* Geometry；
* Ablation；
* Second Pair。

---

## P2：有资源再做

* MLP；
* 完整 C2C；
* 完整 LCF；
* 32K；
* Mismatched Head；
* Cross-Family；
* Query Gate；
* NVMe Serving。

---

# 68. 6 周执行节奏

## Week 1

完成：

```text
T00–T03
```

目标：

> 实验基础可信。

---

## Week 2

完成：

```text
T04–T06
```

输出：

> PCR–Retention Figure。

---

## Week 3

完成：

```text
T07
T08
```

目标：

> Advantage State 可以稳定训练。

---

## Week 4

完成：

```text
T09
```

输出：

* Main Table；
* Gap–CHG；
* TGRR。

这是第一次真正的：

> Go / No-Go。

---

## Week 5

若 CHG 有信号：

```text
T10
```

输出：

* Pareto；
* Context Scaling；
* Scenario A/B/C。

---

## Week 6

完成：

```text
T12
Ablation
```

输出：

* Multi-turn；
* JCR；
* KL；
* Geometry；
* Ablation。

---

# 69. 论文最终可能出现的四种路径

## Path A：Runtime Capability Transfer

如果：

```text
Retention high
CHG > 0
TGRR > 0
PSR_A > 0
Behavior stable
Geometry interpretable
Second pair replicated
```

论文主线保持：

> **Runtime Capability Transfer via KV State Handoff**

---

## Path B：Efficient State Handoff

如果：

```text
Retention high
CHG <= 0
PCR low
PSR_A high
```

转：

> **Efficient Cross-Model State Handoff**

---

## Path C：Mechanism Boundary

如果：

```text
Retention high
CHG <= 0
Geometry mismatch strongly explains failure
```

转：

> **State Compatibility Does Not Imply Capability Compatibility**

这仍然可能是很有价值的机制论文。

---

## Path D：Stop

如果：

```text
Replacement unstable
```

则：

> 停止 Capability Transfer 扩展。

优先研究：

* Alignment；
* Direction Asymmetry；
* Geometry。

---

# 70. 实验前 Pre-registration

建议正式 Test 前生成：

```text
PREREGISTRATION.md
```

冻结以下内容。

## Models

* Teacher；
* Student；
* Commit。

## Dataset

* Primary Benchmark；
* Splits。

## Hyperparameters

* Rank Candidates；
* Top-k；
* α_max；
* Loss Range。

## Seeds

预先固定。

## Statistics

预先固定：

* Primary Metric；
* Statistical Test；
* CI。

## Gates

冻结：

* Retention Gate；
* CHG Criterion；
* Behavior Criterion；
* PSR_A Criterion。

## Negative Result Policy

明确：

> CHG≤0 不重新筛 Test Dataset。

---

# 71. 智能体执行顺序

后续 Coding Agent / Research Agent 必须按照：

```text
T00
 ↓
T01
 ↓
T02
 ↓
T03
 ↓
T04
 ↓
T05
 ↓
T06
 ↓
T07
 ↓
T08
 ↓
T09
 ├── T10
 ├── T12
 └── T11
 ↓
T13
```

执行。

---

# 72. Agent 强制规则

1. 一次只允许执行一个 Task；
2. 每个 Task 有 Acceptance Test；
3. 每个 Task 生成结果报告；
4. Gate FAIL 后不能跳过；
5. Test 不得用于 Auto-Hyperparameter Search；
6. Agent 不得为了产生正 CHG 自动增加无上限实验；
7. 所有 Negative Result 必须保留。

---

# 73. Agent Task Report

```text
TASK_ID:

STATUS:
PASS / FAIL / BLOCKED

OBJECTIVE:

MODEL_PAIR:

DATASET:

CONFIG:

IMPLEMENTATION:

OUTPUT_FILES:

KEY_METRICS:

STATISTICAL_CHECK:

BEHAVIOR_CHECK:

GEOMETRY_CHECK:

SYSTEM_COST:

ACCEPTANCE_CRITERIA:

RESULT:

FAILURE_ANALYSIS:

NEXT_ALLOWED_TASK:
```

---

# 74. 最终实验交付清单

## 数据

* Full Test Results；
* All Seeds；
* Split Manifest；
* Teacher Gap；
* Statistical Test。

## 模型

* Ridge Mapper；
* Low-rank Base；
* Advantage Branch；
* Final APCS。

## 图

* Fig.1 PCR–Retention；
* Fig.2 Gap–CHG/TGRR；
* Fig.3 Pareto；
* Fig.4 Context Scaling；
* Fig.5 Multi-turn；
* Fig.6 Ablation；
* Fig.7 Geometry。

## 表

* Main Results；
* Ablation；
* System Cost；
* Behavior；
* Generalization。

## 工程

* Code；
* Config；
* Environment；
* Test；
* Logs；
* Reproduction Script。

---

# 75. 最终论文判据

最终不得用以下结果替代论文核心结论：

```text
High KV R²
High KV Cosine
High CKA
High Attention Similarity
High Retention
```

上述结果最多说明：

> **State Compatibility**

只有在完整冻结 Test 上得到：

[
CHG>0
]

并同时满足：

* Behavior Stability；
* Defensible System Cost；
* Independent Pair Replication；

才有资格支持：

> **Runtime Capability Transfer**

的强论文主张。

因此整个实验工作的核心不是“想办法把 APCS 做成功”，而是用严格、可复现、可证伪的实验回答：

> **Teacher 相对 Student 多出来的能力，究竟有多少存在于可以跨模型迁移、且能够被冻结 Student 真正消费的 Runtime State 中？**

这应作为所有后续实验、代码、图表和论文分析的唯一主问题。
