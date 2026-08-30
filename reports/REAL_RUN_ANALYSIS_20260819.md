# APCS 真实轨验证报告（Qwen3-4B → Qwen3-1.7B，RTX 5090）

> 实验配置：`configs/pair_qwen3_real.yaml`（provider.kv=hf, score=synthetic, timing=hf）
> Run：`reports/runs/qwen3-4b-to-1.7b-real-20260819-000742/`
> 执行日期：2026-08-19（全部 13 个核心 Task + compliance + multiturn 跑通；ablation 因 ALS 计算量无法在时限内完成）
> 硬件：RTX 5090 32GB / 503GB RAM / CPU-only numpy 数值内核

---

## 0. 结论摘要（TL;DR）

**APCS 跨模型 KV 迁移在当前实验中没有产生有效果。**

| 证据链层 | 对应 Task | 结果 |
|---|---|---|
| Engineering Correctness | T00/T01/T02 | ✅ 全 PASS（Gate 0 通过） |
| Replacement | T04/T05/T06 | ❌ **T05 Gate 1 FAIL**（retention=0.714 < 0.80） |
| Capability Transfer | T07/T08/T09 | ⚠️ T09 Gate 2A "PASS" 但**官方标注 [SIMULATED]**，得分源为合成数据，不可作证据 |
| Behavior Stability | multiturn | ⚠️ CHG/KL 全程恒定为合成构造值，无真实模型 forward |
| Geometry Mechanism | T12 | ⚠️ `placeholder: true`，合成子空间 |
| System Cost | T10 | ⚠️ CUDA 相对基准计时，非端到端管线计时 |
| MVP Decision | T11 | **D_STOP_REPLACEABILITY_UNSTABLE** |
| Generality | T13 | Gate 2A 通过率 = 0.0（0/3 对） |

**核心失败点：跨模型 Ridge KV 映射的替换保真度只有 0.714（目标 ≥ 0.90），
低秩映射更低（0.32–0.38）。Replaceability 不成立 → 后续所有
Runtime Capability Transfer 主张在法律上（§69/§75）不可达。**

---

## 1. 运行数据（中间数据 + 结果数据，已全部落盘）

### 1.1 任务状态总表

| Task | STATUS | 关键数值 |
|---|---|---|
| T00 | PASS | verdict=G1_MATCHED_KV |
| T01 | PASS | logit_cosine=1.0, max_err=0, token_agr=1.0（真实 GPU） |
| T02 | PASS | RoPE round-trip max_err≈8.9e-16, cosine=1.0 |
| T03 | OK | 4 策略，selected=proportional |
| T04 | PASS | kv_cosine=0.729, attn_output_cos=0.810, **R²=0.0034** |
| **T05** | **FAIL** | **mean_retention=0.714 < 0.80**, token_agr=0.209 |
| T06 | OK | lowrank-8/16/32 retention=0.318/0.377/0.365 |
| T07 | OK | gap 均值=0.165，Low/Med/High=8/16/8（val） |
| T08 | OK | **α=0.0**，loss 1.866→1.285 |
| T09 | [SIMULATED] PASS | CHG=+0.163（base_plus_adv），**offline_demo=true** |
| T10 | OK | PSR_A=0.617, N_BE=3, Cost_B=117/234 ms |
| T11 | OK | **D_STOP_REPLACEABILITY_UNSTABLE** |
| T12 | [SIMULATED] OK | **placeholder=true**，CKA=0.943, attn-cos=-0.009 |
| T13 | OK | **gate2a_rate=0.0** |

### 1.2 关键指标明细

**T04 Ridge Baseline（合成 KV，512 ctx，84 calib 样本）**
- mapper 参数：7,340,032；R²=0.0034；KV cosine=0.729；attn-output cosine=0.810
- R²≈0.003 → Ridge 对 Student KV 的方差解释能力≈0（真实 KV 空间不相容）

**T05 Replacement（合成 KV，512/1024 ctx）**
- retention：512→0.728，1024→0.700，均值 **0.714**
- token agreement：**0.209**（argmax 维度一致性 ≈ 随机水平）
- 首测被 15min 超时终止，二次后台运行约 20min 完成（numpy per-head Ridge 极慢）

**T06 Lightweight Mapper（PCR vs Retention）**
- lowrank-8：PCR=0.125, retention=0.318
- lowrank-16：PCR=0.25, retention=0.377
- lowrank-32：PCR=0.50, retention=0.365
- 低秩比全量 Ridge 更差；retention 随 rank 不单调，说明映射没有学出有效结构

**T08 Advantage State Training**
- α_layer_mean = **0.0**（advantage residual 实际上没有被注入）
- base_rms=0.567，residual_rms=0.287，ratio≈0.51
- loss 1.866→1.285 只是低秩残差自身拟合，未传导到 Student 解码行为

**T09 Main Capability**
- 5 方法（text/ridge/base_only/base_plus_adv/full_apcs）CHG 均 > 0（+0.06~+0.21）
- bootstrap 95% CI=[0.149, 0.177]（>0），permutation p=0.0005（<0.05）
- **但 `offline_demo: true`，得分来自 `provider.score: synthetic`（design.md §75 明令禁止用作证据）**

**T11 MVP Decision（§69）**
- retention=0.714 < 0.80 → **D_STOP_REPLACEABILITY_UNSTABLE**

**T13 Generalization**
- n_pairs=3，pass=0 → gate2a_rate=0.0

---

## 2. 分析方法与解读

### 2.1 证据链视角：哪一层是根因？

按 §5 六层证据链，失败发生在第 2 层（Replacement）：

```
Engineering Correctness ✅ → Replacement ❌ → Capability Transfer ⚠️ → ...
```

- **第 1 层（T01/T02）通过**：Student 自产 KV 注入 + 重放 = 原生，数学正确性（RoPE round-trip）也通过。
- **第 2 层（T05）失败**：Teacher KV → Student KV 的线性（Ridge per-head）映射 retention 仅 0.714。
  token agreement 0.209 ≈ 随机（256 维 argmax 随机一致率 ≈ 0.004，0.209 略高但仍远低于可用）。
- **第 3 层起全部无效**：T09/T12 的 PASS/OK 均为合成标注，T11/T13 的负面结果由第 2 层失败直接推导。

### 2.2 为什么替换不成立？（机制假设）

1. **Teacher/Student 的 KV 状态空间线性不可相容**：R²=0.003 说明 Ridge 的线性闭式解几乎无法解释 Student KV 的方差；KV cosine=0.73 只是"方向相似"的弱信号，不是可逆映射的证据。
2. **低秩更差**：低秩映射（rank 8–32）retention 0.32–0.38，进一步说明"低维共享结构"假设不成立 —— Teacher 与 Student 的 KV 不存在一个共享的低秩子空间。
3. **α=0**：T08 的 advantage 残差从未被注入（bounded α 全部收敛到 0），说明"Teacher advantage"无法被编码成 Student 可加状态。
4. **合成 vs 真实**：T04/T05/T06 使用的 `_synth_calibration_set` 生成共享 W_t/W_s 的线性数据，本身就是一个"理想可映射"的构造；在这种理想构造下 retention 仍只有 0.71，真实模型上只会更差。

### 2.3 系统成本（T10）解读

- PSR_A=0.617、N_BE=3、Cost_B=117/234ms（8K/16K）是**真实 CUDA 相对基准**（warmup+10 repeats），但 map/load/query 是轻量算子替代，不是 HandoffPipeline 端到端计时。
- 即便系统收益成立，也只对 Scenario A（沉没成本）有意义；由于替换保真度不成立，**系统收益无法落地**（T11 已排除 Path A/B/C）。

---

## 3. 结论

1. **没有效果，未达预期。** 核心科学端点 CHG>0 / TGRR>0 / Retention≥0.90 均无法在可验证的真实语义下成立：
   - T05 Gate 1 FAIL（retention 0.714），Replaceability 不成立；
   - T11 verdict = `D_STOP_REPLACEABILITY_UNSTABLE`；
   - T13 泛化 Gate 2A 通过率 = 0。
2. **T09 的 "PASS" 不构成证据**：官方标注 `[SIMULATED]`，得分来自合成 provider，§75 禁止作为 Runtime Capability Transfer 的声明依据。
3. **结论**：在 Qwen3-4B→Qwen3-1.7B 上，**线性/低秩 APCS 映射无法把 Teacher KV 替换为 Student KV 的高保真状态**。若继续，需要：
   - 接入真实 Student 自产 KV（`provider.kv=hf` 的 T04–T06 路径当前仍是合成），先量化真实替换保真度；
   - 或放弃线性映射假设，探索非线性/逐层学习映射，但需重新论证 Replaceability。

---

## 4. 附：合规性 & 未完成任务

- **compliance**：0 violations，但 **8/8 规则 UNKNOWN、fully_verified=False**（运行时埋点缺失，只证明"静态无违规"）。
- **multiturn**：跑通，但 CHG/KL 为合成恒定值（0.170/0.684），task_score 0.67→0.57 线性衰减为模拟。
- **ablation**：未能完成。ALS LowRank 对 64 样本×1024 token concat 拟合在 numpy 上计算量约等于 t06 的 8 倍（t06 本身耗时 ~49min），4 个变体预计 >6h，超出执行时限。这不是实验失败，而是当前 numpy 实现的工程瓶颈。
- **生产级差异**：本环境 Python 3.10（README 建议 3.11+），部分行为未按 3.14 验证。

所有中间数据与结果数据保存在：
`reports/runs/qwen3-4b-to-1.7b-real-20260819-000742/<task>/{metrics.json, summary.md, task_report.md, metadata.json, compliance.json}`。