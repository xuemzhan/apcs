# APCS 真实 GPU 实验与论文对照结论报告

**报告日期：** 2026-08-26  
**主运行：** `qwen3-4b-to-1.7b-real-gpu-20260826-015843`  
**模型对：** Qwen3-4B → Qwen3-1.7B  
**最终判定：** 当前实现与当前证据不足以支持论文所定义的 Runtime Capability Transfer；应按 Path D 停止当前结论链并重构验证流程。研究方向本身尚未被证伪，但 APCS 的核心机制实际上还没有被真实实验测试。

---

## 1. 执行摘要

本轮实验最可靠的正向结果有三个：

1. Qwen3-1.7B 的 Self-KV 可以经过外部保存、重建和注入后保持原生解码等价，4 个样本上 logit cosine、token agreement 都为 1.0。这说明基础 cache replay 工程路径成立。
2. Qwen3-4B 与 Qwen3-1.7B 在 matched KV 结构下，K 的跨模型线性映射很强：held-out 数据上 K cosine 为 0.9661、K R² 为 0.9282。
3. calibration/test 的样本 ID 没有重叠，64 个 calibration、16 个 evaluation 样本的划分在数据层面是独立的。

但严格论文结论必须否定：

- V 映射明显失败：V cosine 仅 0.5772，V R² 为 -0.9449；K/V 平均 0.7716 不能掩盖 V 的失真。
- 论文核心端点 CHG/TGRR 没有真实测量。T09 的分数由固定期望值和随机噪声合成，正 CHG 是构造出来的，不是模型输出。
- T08 没有使用真实 Teacher/Student KV，也没有优化论文写出的 task/self/teacher/attention 联合目标；其输入是随机数组，且最终 alpha 全为 0，因此所谓 advantage branch 没有真实注入。
- T10 不是端到端 HandoffPipeline 计时，只是 CUDA proxy；0.617 的 PSR_A 不能作为系统收益。
- T12 是随机子空间占位，不能解释真实几何。
- T13 把同一模型对的两个 run 当成两个 pair，实际上没有第二个模型对的泛化实验。
- 最新运行把 Gate 1 的最低阈值从原先预注册/论文的 0.90 降到 0.75，发生在看见 V≈0.57 之后。这种事后改阈值不能用于科学结论；按原始规则 0.7716 应为 FAIL，而不是 CONDITIONAL。

因此，本轮结果证明的是“工程通路部分可用、K 对齐有信号、V/端到端行为链未打通”，而不是“能力可以通过 KV 状态迁移”。

---

## 2. 证据等级划分

| 模块 | 是否使用真实模型/GPU | 是否直接支撑论文主张 | 本报告判定 |
|---|---:|---:|---|
| T00 架构兼容 | 是/配置与模型结构 | 间接 | 有效工程前提 |
| T01 Self-KV Replay | 是 | 间接 | 有效，但 n=4，只证明重放链 |
| T02 RoPE round-trip | 数值验证 | 间接 | 有效数学单元测试 |
| T03 Layer Alignment | 默认 proportional；其他策略未做真实选择 | 否 | 仅配置结果 |
| T04 Ridge KV Mapping | 是，HF KV + GPU | 仅支撑表示层诊断 | K 有效、V 失败 |
| T05 Replacement | 是，HF KV + GPU | 当前不能直接支撑 | 指标是 KV cosine，不是论文 task retention |
| T06 Lightweight Mapper | 是，HF KV + GPU | 间接 | 当前低秩方案失败 |
| T07 Teacher Gap | 否，合成分数 | 否 | 无科学证据 |
| T08 Advantage State | 否，随机 KV | 否 | APCS 核心机制未测试 |
| T09 CHG/TGRR | 否，硬编码分布合成 | 否 | Gate 2A 无效 |
| T10 PSR/Cost | CUDA proxy，非端到端 | 否 | 系统收益未测 |
| T11 Decision | 混合了真/假输入 | 只能作流程提示 | D 方向正确，但输入链不完整 |
| T12 Geometry | 否，随机子空间 | 否 | 无机制证据 |
| T13 Generalization | 同一 pair 的重复 run | 否 | 泛化未测 |

---

## 3. 真实 GPU 结果的详细解读

### 3.1 Gate 0：Self-KV Replay 成立，但证据范围有限

T01 在真实 Student 模型上得到：

- n=4
- mean logit cosine = 1.0
- mean max error = 0
- mean token agreement = 1.0
- Gate 0 = PASS

这说明 HF 原生 cache 与 numpy offload/rebuild 路径在当前短样本上等价，排除了“cache 格式或注入 API 本身错误”这一大类问题。

它不能证明 cross-model mapped cache 可被稳定消费，也不能证明长上下文、多 token 解码或任务正确率。下一轮仍需把 replay 扩展到多个长度、多个 decode step 和至少数十个样本。

### 3.2 Ridge 映射存在极强的 K/V 不对称

T04/T05 的 held-out 结果：

| 指标 | K | V | K/V 平均 |
|---|---:|---:|---:|
| cosine / 当前代码中的 retention | 0.9661 | 0.5772 | 0.7716 |
| R² | 0.9282 | -0.9449 | -0.0083 |
| proxy token agreement（T05） | 0.8047 | 0.2578 | 0.5313 |

解释：

- K 的线性可预测性很强，证明同系列 matched-KV 模型在 key 空间存在可利用结构。
- V 的负 R² 表明当前 Ridge 输出在平方误差意义上甚至弱于用目标均值预测，不能视为可替换状态。
- 0.7716 是 K 与 V 的算术平均。对实际注意力而言，K 和 V 是串联关系：K 决定权重，V 决定被聚合内容。一个严重失败的 V 不能被一个优秀的 K 简单“平均补偿”。
- T05 的 token agreement 是 KV 最后位置最大通道索引的一致率，不是真实 vocabulary token agreement。

### 3.3 当前 retention 与论文 retention 口径不一致

论文中的 retention 是 handoff 任务得分相对 Student Self-Prefill 得分的保持率；当前 T05 计算的是 predicted KV 与 Student KV 的 cosine，因为 self cosine 恒为 1，所以所谓 retention 实际就是 KV cosine。

这导致两个问题：

1. 0.7716 不能与论文先验的 90–96% task-score retention 直接比较。
2. Gate 1 不能仅凭该数值决定 downstream replacement 是否成功，必须通过真实 cache 注入后测任务得分、logit KL、top-k overlap 和多步生成稳定性。

因此，当前最准确的说法不是“任务 retention=77.16%”，而是“短上下文 held-out KV cosine：K=96.61%，V=57.72%”。

### 3.4 `attn_output_cosine_V=0.936` 不能作为正面证据

当前实现对 K 和 V 分支分别调用同一个 proxy：`softmax(Q·X^T)·X`。

- K 分支令 X=K，可以作为一种 K-only 形状 proxy，但仍没有使用真实 Student Q。
- V 分支令 X=V，相当于计算 `softmax(Q·V^T)·V`，不是 Transformer 的注意力输出。
- 真实联合注意力应为：`A=softmax(Q·K^T/sqrt(d))`，`O=A·V`。V 的评价必须在相同 K/attention weights 下比较 `A·V_self` 与 `A·V_mapped`，或直接比较联合 `(K_mapped,V_mapped)` 的实际 layer output/logits。

所以不能根据 0.936 得出“V 虽然 raw cosine 低但实际质量优秀”。该结论需要重新测量。

### 3.5 短上下文限制很严重

配置请求 512/1024 token，但 T04/T05/T06 为避免 padding，将所有样本裁到全体样本的公共最短长度，最终 effective sequence length 只有 70。

因此本轮只验证了约 70 token 的映射，无法回答论文的 4K/8K/16K 长上下文问题。当前 HellaSwag 样本也不是合适的长上下文载体。

### 3.6 超参数结果只说明“当前线性方案趋于饱和”，不证明数学上限

已有真实 run 表明：

| calibration | λ_V | V cosine |
|---:|---:|---:|
| 16 | 0.001 | 0.5256 |
| 32 | 0.01 | 0.5634 |
| 64 | 0.001 | 0.5772 |
| 64 | 0.1 | 0.5772 |

样本从 16 增加到 64 有改善，λ 从 0.001 到 0.1 基本不变。这支持“在当前无偏置 per-layer/per-head Ridge、proportional alignment、70-token 数据上已经接近经验平台期”，但不支持“0.57 是数学极限”。

尤其需要纠正：Teacher V 与 Student V 的原始 cosine 接近 0，并不意味着不存在高质量线性映射。两个坐标系即使正交，也可能通过一个线性旋转完美对齐。要证明线性不可行，需要报告 held-out 最优线性风险、奇异谱/CCA、样本规模曲线、带偏置/标准化基线，并给出置信区间，而不能只依据原始 cosine。

### 3.7 Lightweight Mapper 未达到论文预期

T06：

- Low-rank 8/16/32 的总 cosine 仅 0.070/0.033/0.028，且 rank 增大反而下降。
- Shared-basis-16 为 0.621，但 K=0.929、V=0.313，仍由 V 主导失败。

这与论文先验“PCR≈0.23 时 retention 只小幅下降”明显不符。当前低秩 ALS 实现、训练样本聚合方式或目标函数需要先做单元诊断；在修复前不能把 T06 当作有效的 rate–retention 曲线。

---

## 4. APCS 核心主张为何尚未被测试

### 4.1 T08 不是论文所描述的 Advantage State Training

论文定义的目标是：

`L_task + λ_self L_self + λ_teacher L_teacher + λ_att L_att + λ_reg L_reg`

但当前 T08：

- 生成随机 Teacher/Student KV；
- 训练残差去逼近 `kv_s - z`；
- 没有真实任务答案、Teacher advantage、Student logits 或 attention output；
- 没有读取 T07 的真实 teacher gap；
- bounded alpha 初始化为 0 后没有训练，最终 alpha mean/std 都为 0。

这实际上仍是一个随机数据上的 self-KV reconstruction demo，不是 advantage-preserving branch。由于 alpha=0，即使把该分支接入推理，最终状态也退化为 compatibility base。

### 4.2 T09 的正 CHG 是预设分布，不是实验发现

T09 为 Student、Teacher、Ridge、Base、Base+Advantage、Full APCS 分别硬编码了约 0.50/0.80/0.58/0.60/0.66/0.70 的期望分，再加入可复现噪声。因此：

- base+adv CHG=+0.1626；
- 95% CI 不跨 0；
- permutation p=0.0005；
- Gate 2A=PASS；

这些统计量只反映合成器被设计成 handoff 优于 student，不能说明真实模型行为。统计检验无法把合成前提变成科学证据。

### 4.3 T10/T12/T13 同样没有补齐论文证据

- T10 执行了 CUDA proxy，但不是 teacher capture → offload → map → H2D → inject → query/decode 的端到端流水线。
- T12 的 CKA/principal angle/effective rank 来自随机子空间，不能用于解释 V 失败。
- T13 扫描了两个 run_id，但二者都是 Qwen3-4B→1.7B；run 数量不等于模型 pair 数量。

---

## 5. 与论文要求逐项对照

最新版论文母稿明确要求：

- frozen-test CHG>0，且置信区间排除 0；
- TGRR 持续为正；
- PSR>0；
- Student 不重新 prefill 原始 X；
- 至少两个 Large→Small pair；
- 真实 4K/8K/16K 系统与任务测量；
- 负结果时转向 Efficient State Handoff 或 State Compatibility ≠ Capability Compatibility。

当前状态：

| 论文条件 | 当前状态 | 结论 |
|---|---|---|
| 无 X 重预填充的真实 handoff 解码 | 只验证了 Self-KV replay；cross-model 端到端未测 | 未满足 |
| Task retention ≥90% | 没有真实 task retention；KV cosine 均值 0.7716 | 未满足 |
| CHG>0 且 CI>0 | 合成分数 | 未满足 |
| TGRR>0 | 合成分数 | 未满足 |
| PSR>0 | CUDA proxy | 未满足 |
| Advantage branch 有效 | 随机数据，alpha=0 | 未满足 |
| 真实 geometry/behavior | 占位 | 未满足 |
| ≥2 模型对 | 同一 pair 重复 run | 未满足 |

按照论文自己的判定逻辑，只能进入 Path D：Stop / Redesign。

---

## 6. 最终可行性判断

### 6.1 对“当前这套实现”的判断

**不可行，不能继续沿用当前 T07→T13 结果链得出论文结论。**

原因不是 GPU 资源不足，而是核心实验仍是合成/占位，并且 Gate、指标和论文定义发生了错位。继续调低 Gate、调 Ridge λ 或增加相同类型的随机 demo，不会回答核心科学问题。

### 6.2 对“Teacher KV → Student KV 直接全量替换”的判断

**当前模型对上，朴素 per-head Ridge 全量替换风险很高。**

K 映射非常有希望，但 V 映射在真实 held-out 数据上明显失败。由于真实注意力输出和下游行为尚未测量，目前不能断言彻底不可行，也不能断言 0.57 是不可突破的上限。更准确的结论是：

> 在 Qwen3-4B→1.7B、约 70-token、64 calibration/16 eval、proportional layer alignment、无偏置 per-head Ridge 条件下，K 可线性迁移，V 不能被当前目标稳定重建；全量 KV replacement 尚未达到进入能力迁移实验的最低工程条件。

### 6.3 对“Runtime Capability Transfer 研究方向”的判断

**方向仍有研究价值，但成功概率目前未知，且必须先建立真实闭环。**

论文提出的关键问题仍然成立：Teacher 的 context-conditioned advantage 是否存在于 Student 可消费的运行时子空间。当前实验没有真正训练或注入这样的子空间，所以既没有证明 Yes，也没有证明 No。

最有价值的现有发现是“K 可迁移、V 不对称”。它可能导向三条研究路径：

1. 改善 V/联合 attention-output 对齐，继续追求严格 replacement；
2. 接受 strict zero-prefill 很难，转向 selective re-prefill 的系统方案；
3. 不重建全量 KV，转向 Causal Advantage Subspace / Capability Capsule，只传 Student 真正可消费的任务因果成分。

---

## 7. 改进路线（按优先级）

### P0：先把科学闭环接通，暂停新结论

#### P0.1 实现真实 score provider 和端到端 handoff

每个样本至少跑四条严格可比路径：

1. Student full-prefill X + q（基线）；
2. Teacher full inference（teacher gap）；
3. Teacher KV → mapper → Student，仅输入 q，不再输入 X；
4. Text/Summary handoff 基线。

输出真实 task score、next-token KL、logit cosine、top-k overlap、生成正确率和多 token 稳定性。任何 cache 注入失败都禁止 silent re-prefill。

#### P0.2 纠正 Gate 与 retention 定义

- 恢复原始预注册 Gate；不要根据本轮结果把 0.90 改为 0.75。
- 表示层指标命名为 `kv_cosine_K/V`，不要称为 task retention。
- Gate 1 以真实 downstream score retention 和行为稳定性为主。
- K/V 必须分项报告；禁止用简单平均掩盖单路失败。

#### P0.3 修复 attention-output 指标

对同一真实 Student Q 计算：

- Self：`A_s=softmax(Q_s K_s^T)`，`O_s=A_s V_s`；
- K-only：`A_k=softmax(Q_s K_hat^T)`，`O_k=A_k V_s`；
- V-only：`O_v=A_s V_hat`；
- Joint：`O_hat=softmax(Q_s K_hat^T)V_hat`。

分别报告 output cosine、MSE、next-layer hidden cosine 和最终 logits/KL。这样才能知道 V raw cosine 低是否真的影响模型消费。

#### P0.4 使用真正的长上下文数据

- 不再把全部样本裁到全局最短 70 token。
- 按长度 bucket，使用 attention mask 或逐样本变长聚合。
- 引入 RULER/LongBench/多文档 QA，至少完成 512/1K/4K，出现真实信号后再扩到 8K/16K。
- 校准规模从 64 提升到论文要求的 100–500，并做 3 个独立训练 seed；eval 至少数百样本或按任务规模做功效分析。

### P1：针对 V 的系统诊断与修复

#### P1.1 先补齐强线性基线

当前 Ridge 没有偏置项。建议依次比较：

- per-layer/head centering + affine Ridge（含 bias）；
- 标准化/whitening 后 Ridge；
- Orthogonal Procrustes；
- CCA/PLS/reduced-rank regression；
- teacher-layer mixture + head permutation/transport；
- data-driven/geometry-aware layer alignment，而不是只改配置字符串。

用 held-out 曲线判断究竟是均值/尺度问题、层错配、head 错配还是非线性问题。

#### P1.2 优化联合功能目标，而不是只重建 raw V

可以将训练目标改为：

- joint attention output loss；
- next-layer hidden-state loss；
- student logits KL；
- task loss + self-stability loss；
- 少量 raw KV 正则作为稳定项。

这与论文强调的“behavior-aligned objective”一致，也能避免把坐标等价但 raw cosine 较低的表示误判为失败。

#### P1.3 非线性 mapper 必须作为受控基线

在强线性基线之后，再比较小型 MLP、gated low-rank residual 或 token-conditioned adapter。必须限制参数量/PCR，并用独立 validation 选择容量，避免把 test 当调参集。

### P2：如果 strict replacement 仍失败，改变问题设定

#### P2.1 Selective Re-prefill

只重算失败最严重的层/head/token，复用 K 或其它兼容部分。研究最小重算比例与 task retention/PSR 的 Pareto。该方向不再满足 strict zero-X-prefill，但更可能形成可部署系统结果。

#### P2.2 Causal Advantage Subspace

不要先复制完整 Student KV，再叠加一个未定义的 advantage residual。改为通过真实 intervention 找出：

- teacher-exclusive；
- task-causal；
- student-consumable；

的 layer/head/subspace。用注入/删除实验测 `Score(S|base+z)-Score(S|base)`，而不是靠 reconstruction loss 给 residual 命名为 advantage。

#### P2.3 Pair-aware 路由

先在更对齐的 pair 上建立正例，例如同架构不同 checkpoint、蒸馏关系更强或层/head 可自然对应的模型；同时把当前 Qwen3-4B→1.7B 作为方向不对称/几何失配的负例。

---

## 8. 下一轮最小实验矩阵与停止条件

### 8.1 最小实验矩阵

| 阶段 | 必做实验 | 通过条件 |
|---|---|---|
| E0 | 真实 cross-model cache inject + 多步 decode | 无 X re-prefill；无 fallback；流程可重复 |
| E1 | 512/1K task retention，Student/Teacher/Ridge | task retention≥0.90，KL/生成不崩 |
| E2 | K-only/V-only/Joint attention ablation | 明确 V 是否为实际瓶颈 |
| E3 | affine/whitened Ridge、Procrustes/CCA、MLP | held-out 与 3 seeds 有稳定改善 |
| E4 | 真实 APCS Base vs Base+Adv | CHG CI 下界>0，TGRR>0 |
| E5 | 4K/8K E2E timing | Scenario A PSR>0；Scenario B 成本完整 |
| E6 | 第二模型 pair | 相同方向复现或形成可解释边界 |

### 8.2 建议停止条件

满足任一条件时停止 strict Runtime Capability Transfer 主线：

1. 强线性+受控非线性 mapper 后，真实 task retention 仍低于 0.80；
2. task retention 达标，但 Base+Adv 在 3 seeds 的 frozen test 上 CHG CI 仍覆盖 0 或为负；
3. CHG>0，但 KL/JCR/多轮稳定性明显恶化；
4. 4K/8K 端到端 PSR≤0，且没有多 Student 复用场景可摊销；
5. 第二 pair 无法复现，且没有几何指标能预测成功/失败。

此时论文应转向：

- Efficient Cross-Model State Handoff；或
- State Compatibility Does Not Imply Capability Compatibility；或
- Selective Re-prefill / Geometry-aware Partial Handoff。

---

## 9. 可直接用于论文更新的最终结论

> 在真实 Qwen3-4B→Qwen3-1.7B GPU 实验中，我们验证了 Student Self-KV 外部重放的工程等价性，并观察到显著的 K/V 映射不对称：per-head Ridge 在 held-out 短上下文上可高质量恢复 K（cosine 0.966，R² 0.928），但不能稳定恢复 V（cosine 0.577，R² -0.945）。这一结果说明 matched KV 几何并不足以保证完整状态可替换。不过，当前实验尚未对 Runtime Capability Transfer 作出正面或负面的最终科学判定，因为 advantage branch、CHG/TGRR、端到端 PSR、真实几何和第二模型对仍未被真实测量。特别是，现有正 CHG 来自合成评分，不能作为证据。按预注册标准，本轮应判定为 Stop/Redesign：先恢复真实 task-level replacement 与端到端成本闭环，再评估是否存在可被 Student 消费的 teacher-advantage subspace。若强基线下真实 CHG 仍不为正，最诚实且有价值的结论将是 State Compatibility Does Not Imply Capability Compatibility，而不是继续降低 Gate 或把表示相似度解释成能力迁移。

---

## 10. 核心证据文件

- 主运行汇总：`reports/runs/2026-08-25/qwen3-4b-to-1.7b-real-gpu-20260826-015843/ANALYSIS_REPORT.md`
- T04 真实 KV 映射：`.../t04/metrics.json`、`.../t04/kv_split_manifest.json`
- T05 当前 replacement 指标：`.../t05/metrics.json`
- T06 低秩结果：`.../t06/metrics.json`
- T08 合成 advantage demo：`.../t08/metrics.json`
- T09 合成能力分数：`.../t09/metrics.json`
- T10 CUDA proxy：`.../t10/metrics.json`
- T12 placeholder geometry：`.../t12/metrics.json`
- T13 泛化扫描：`.../t13/metrics.json`
- 原始与修改后的预注册：`qwen3-4b-to-1.7b-real-gpu-20260825-215241/PREREGISTRATION.md`、`qwen3-4b-to-1.7b-real-gpu-20260826-015843/PREREGISTRATION.md`
- 最新论文母稿：`paper/KV_Runtime_Capability_Transfer_Prior_Projection_v2.docx`

