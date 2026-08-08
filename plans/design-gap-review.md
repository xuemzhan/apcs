# design.md ↔ 当前实现 差距审查报告

> 审查日期：2026-08-08
> 审查对象：`design.md`（2599 行实验方案 V2.1）vs `apcs/`（25 模块）+ `configs/` + `tests/`（72 测试）
> 审查方式：逐条对照 §1–§75 的 Gate / 指标 / 任务 / 禁止项 / 产物规范

---

## 0. 结论摘要（TL;DR）

**现状定性**：这是一个**结构完整、章节标注清晰、但 100% 离线仿真（numpy）**的研究脚手架。全部 13 个核心任务（T00–T13）+ 3 个额外 subcommand + 8 个 Figure 渲染器都有可运行实现，72 个测试全绿。

**三类差距**：
1. **科学诚实性差距（最关键）**：所有 Gate（0/1/2A）在当前实现中**恒 PASS**——T01 模拟返回 1.0、T09 用固定 base 分使 CHG≈0.16 恒定、Gate 2A 必然通过。这会产生"实验成功"的假象，直接违反 design.md §75（不得用模拟替代真实结论）。
2. **工程正确性差距（9 项已确认 bug）**：RidgeMapper per-head 冗余、top-k 截断、校准循环覆盖、`n_t<n_s` 崩溃、G2 未接线、双重 de-RoPE、prereg "> True"、T11 run_id 正则永远 None、Figure7 数据流断裂、`hash()` 跨进程不可复现。
3. **论文规格差距（结构性缺失）**：K/V 未独立参数化（§22）、严格训练/验证/test 三层未见（§16/§36）、无真实 GPU 执行路径（§53）、Score 层无真实 LLM 输出（§37 7 方法）。

**结论：** 脚手架可作为"工程骨架"用于协议排演，但**论文口径上不得作为证据**。修复优先序：诚实性 > 确定性 bug > 规格补全。

---

## 1. 逐任务覆盖对照表

| Task | design.md 条款 | 实现文件 | 覆盖度 | 关键差距 |
|------|--------------|---------|-------|--------|
| T00 | §28 读取 12 字段 → model_compatibility.json + G1/G2/G3 分流 | compat/scanner.py | ✅ 覆盖 | fallback 硬编码仅 Qwen3-4B/1.7B；SmolLM2 走 else 分支（32000 vocab、16 kv_heads=**错误**：SmolLM2 是 8/3） |
| T01 | §29 Self-KV Replay（Gate 0）| replay/runner.py | ⚠️ 模拟 | `_simulated_replay` 恒返 logit_cosine=1.0/max_err=0/agreement=3.0 → Gate 0 恒 PASS。无真实注入路径 |
| T02 | §30 RoPE round-trip（3×3×128） | rope/runner.py | ✅ 数学正确 | de_rope 公式正确（负角度），pass |
| T03 | §21 四策略 + §31 输出 | alignment/runner.py | ✅ 覆盖 | 无 `w_{l,i,h}` 权重和=1 的约束检查输出；si_m 合成带状矩阵（真实需 attn-output cosine） |
| T04 | §32 Ridge baseline（100–500 samples） | mapper/runner.py | ⚠️ bug | calibration 循环覆盖 W；T(KV 混合训练，非 K/V 分离) |
| T05 | §33 Replacement（512–4K，若干语义） | mapper/runner.py | ⚠️ bug | 同上循环覆盖；KL 用 |a|.mean 代理；Retention=cosine 比，非真实 Score |
| T06 | §34 PCR vs Retention (Figure 1) | mapper/runner.py | ⚠️ bug | 每 rank 单独校准覆盖；SharedBasis 仅 `shared_basis: true` 时跑 |
| T07 | §35 Gap Freeze（train/val/test） | capability/runner.py | ⚠️ 模拟 | 随机 uniform 得分，无真实模型 eval；train 层未见真实 margin 计算 |
| T08 | §36 Advantage State Training（冻结 Student） | advantage/runner.py | ⚠️ 结构对 | SourceLayerMixer 可；但 LowRankResidual **无 σ 非线性**（自注"线性版"声明）、**无训练循环**（A/B 随机初始化后直接 eval）、KV 混合在 (Z) 上而非独立 |
| T09 | §37 7方法 + §51 统计 | capability/main.py | ⚠️ bug+模拟 | `_simulate_scores` 固定 base → CHG≈0.16 恒定；`hash()` 跨进程随机；Gate 2A 恒 PASS；无真实 model forward |
| T10 | §4 三场景 + §38 测量 + §53 单卡 | system/runner.py | ⚠️ 模拟 | `_simulate_timings` 用线性公式；**N_BE 由公式算出**（§38 明确禁止"理论估算当论文结果"）；VRAM est 公式可疑（4096*2 近似） |
| T11 | §39 三类判定 | decision/runner.py | ⚠️ bug | run_id 正则解析 `run_dir.name`（"t11"）→ 恒 None；fallback 到 mtime 选 run |
| T12 | §40–§43 几何诊断 | geometry/runner.py | ⚠️ 模拟 | `_random_subspace` placeholder（明确标注）；**fig7 数据断裂**：metrics.json 无 geometry 键 |
| T13 | §44 泛化 ≥2 Pair | generalization/runner.py | ⚠️ 数据依赖 | 读 run 目录，无 run 则空；Gate 判定 PASS |
| §46 多轮 | figure/ runner.py | ✅ 结构 | 合成得分衰减；KL 用随机 Dirichlet（无意义） |
| §47 消融 | ablation/runner.py | ⚠️ 部分 | A1/A6/A8 有数值，A2/A3/A9/A10 只报告 config 状态、无实际 on/off 对照；A4/A5/A7/A11 未实现 |
| §52 合规 | compliance/__init__.py + runtime | ✅ 结构 | 信号从 cfg 静态推断（`infer_signals_from_cfg`），**runtime 信号靠手动 track**，无自动挂接 → 只能抓静态违规 |
| §64/§65 | io/metadata.py | ✅ 覆盖 | validate/enrich 正确；缺 hardware 实时检测等 |
| §70 prereg | prereg/__init__.py | ⚠️ bug | Gate 2A 渲染 "`> True`"; 缺 `${...}` 占位符处理（T70 无实现） |
| §71-73 | orchestrator/__init__.py | ✅ 结构 | DAG 依赖、Task Report 12 字段完整；`next_allowed` 有传递依赖 bug：`if deps & failed` 未检查非直接依赖的传递 FAIL |
| §56-62 Figures 1-7 | figures/__init__.py | ⚠️ | fig7 数据断裂；fig2 gap_strata 若 t09 dict 无 → 空；fig4 需要 t10 metrics 里有 per_context（t10 的 metrics 是 `{}`！CLI system.json）→ **fig3/fig4 拿不到 t10 数据** |

---

## 2. 九项已确认 Bug（程序化验证过）

| # | 位置 | 现象 | 验证方式 |
|--|-------|------|--------|
| B1 | math.py `RidgeMapper.fit` L289-318 | H 次循环求同一矩阵复制 `W[(s,h)]`，transform 只用 `W[(s,0)]`，n_params 虚高 H× | 手动算：n_params=3.67M == H+ professores 声明矛盾 |
| B2 | math.py 4 mapper top 池堆丁).fit | n=min(src,tgt) 把 k*S 截断到 S → k>1 时只用第一层训练，但 transform 平均 k 层 | `n used by fit: 64 vs available rows: 128` |
| B3 | mapper/runner.py 校准循环 | `for i in range(100): fit(...)` 每轮覆盖 W → 实际只用最后一组样本 | W 值随最后一次 fit 变化 |
| B4 | configs/pair_smollm2.yaml + math `_check_shape` | teacher 24(n_t<n_s) → 抛 ValueError；H_t= 8≠H_s=3 从未走 G2 | 触 T04 崩溃确认 |
| B5 | prereg __init__.py L8 | 布尔插值 → "`> True`" 字面量 | PREREG 行输出确认 |
| B6 | decision/runner.py L9 | 正则匹配 `run_dir.name`（"t11"）→ shared_run_id 恒 None | 实验确认 None |
| B7 | geometry runner + figures | metrics.json 不含 geometry；fig7 期待 t12_metrics["geometry"] | fig7 输出为空 |
| B8 | mapper/mismatched.py | `_project_teacher` 已 de-RoPE 又传给 inner → 双重 de-RoPE | 代码路径确认 |
| B9 | capability/main.py | `_simulate_scores` 固定 base ∈ seed 无关；`hash()` 跨进程随机；Gate 2A 恒 PASS | 三种种子输出同一 0.16/CI/p |

---

## 3. 结构性差距（超出 bug，属于规格未实现）

### 3.1 科学诚实性（最高优先）
- §75 明示：R²/Cosine/CKA/Attention/Retention **最多说明 State Compatibility**，只有冻结 Test 上 CHG>0 + 行为稳定 + 系统成本 + 独立 Pair 复现才支持 Runtime Capability Transfer。
- 当前实现输出 metrics.json 中 `mean_kv_cosine`/`mean_retention` 等，且 T04 gate `PASS if cosine>0.5`；T05 retention 用 cosine 比值；T09 恒 PASS → **整条证据链被模拟结果填充**。必须：
  1. 在 T01/T04/T09/T10 metrics 中加 `"offline_demo": true` + `note` 字段；
  2. Gate 判定逻辑区分"模拟 smoke test"与"真实实验"两档。
- `_simulate_decision` 用 `hash()`（§51 可复现要求下跨进程随机）→ 必须换 `default_rng(稳定 hash)`。

### 3.2 架构规格没有实现
- **K/V 分离（§22 强制）**：全部 mapper 对单一 tensor `(L,S,H,D)` 拟合/变换，无 K、V 独立适配器。`separate_kv` 配置项仅是旗帜，没接进代码。
- **无训练过程（T08）**：LowRankResidual 随机初始化后直接推理，返回"训练"名不副实（无 loss、无梯度、无 rank 收敛）。
- **真实推理管线（第 53）**：唯一接口是 `single_card_pipeline`（已存在）但无接线——没有任何 `apcs` 代码加载 HF model 或捕获 past_key_values。
- **Score 层缺失**：§15 Scoring 是 Retention/KL 计算；真实 score 需 LLM forward。当前无 `evaluate()` 实现。
- **§21 `sum_i w_{l,i,h}=1`**：SourceLayerMixer 初始化均匀的 1/k，但未输出训练后的权重、也没有对 w 做归一化校验报告。
- **§25 α_l=α_max·tanh(a_l)** 存在（BoundedAlpha）但 a 从未训练（全 0 → α=0）→ base+adv 退化为 base。

### 3.3 第二阶段 Pair 配置不精确
- pair_smollm2.yaml：SmolLM2-1.7B 真实架构为 30 层/32 attn-heads/8 kv-heads/head_dim 64；current 配置 24 层错误（其实是 SmolLM2-1_7b 是 24 层？——需核对：HuggingFaceTB/SmolLM2-1.7B 是 30 层，135M 是 30 层。要对照 HF 确认真实值）。
- `/compat` scanner fallback 对 SmolLM2 走默认分支（vocab=32000/attn=16/kv=16/head_dim=128），与 pair_sm.mol₂ 配置冲突。

### 3.4 测试覆盖缺口
- 现有 72 测试覆盖：metrics/rope/mapper math/einsum/batch/§52/metadata/CLI 端到端/T11 判定。
- **未覆盖**：config 加载字段（_cfg_layers 逻辑）、G2 路径、prereg 文本、figures 渲染、T09 可复现性、inference 接口。

---

## 4. 按优先级排序的修复路线（含 design.md ↔ 文件对照）

| 优先级 | 修复项 | design.md | 文件 | 测试 |
|------|-------|----------|------|------|
| **P0-1** | T09 RNG 化 + offline_demo 标注（消除恒 PASS） | §51/§75 | capability/main.py | test_decision.py |
| **P0-2** | `_check_shape` 支持 n_t<n_s + SmolLM2 G2 若 H 不等 → MismatchedHeadMapper | §20 §12 | math.py, runner.py | test_mapper.py（改1个既有测试） |
| **P0-3** | RidgeMapper per-layer（去 H× 虚高） | §3.4 | math.py | test_core.py |
| **P0-4** | 四 mapper 修正 top_k 堆叠（去掉 n=min 截断） | §19 B3 | math.py | test_mapper.py |
| **P0-5** | 校准循环 → 聚合架构（buffered samples） | §30 | mapper/runner.py | test_mapper.py |
| P1 | prereg 数字 gate 渲染 | 0-70 | prereg/__init__.py | test_extras.py |
| P1 | 11 号 run_id 解析 | §69 | 决策/runner.py | test_cli.py |
| P1 | fig7 数据贯通（metrics 加 geometry + fig 读取） | §62 | geometry/runner.py, figures | test_extras2.py |
| P1 | K/V 分离接口（mapper fit/transform 加 kv kind 参数） | §22 | math.py+mapper | test_mapper.py |
| P1 | 单卡 pipeline 真实预存：interpreter/torch backend 接口骨架（numpy 后端可跑） | §53 §52 | 新 `apcs/inference/` | test_inference.py |
| P2 | 消融 A3/A4/A5/A9/A10 真实 on/off 对照（而非报告 config） | §47 | ablation/runner.py + advantage | test_ablation.py |
| P2 | T08 训练优化：加真实 LS 闭合循环 + σ 非线性 | §36 §22 | advantage via 损失迭代 | test_extras.py |
| P2 | 真实 Score pipeline（T05/T09 用 softmax 假 logits | scoring) | §15 §16 | 新增 | — |
| P2 | orchestrator `deps & failed` 传递依赖 bug | §72 | orchestrator/__init__.py | test_extras.py |
| P2 | pair_qwen3/pair_smollm2 确认真实架构（HF） | §11 | 配置 | — |

---

## 5. 72 现有测试兼容性

只有 **1 个既有测试**与新修复冲突（`test_check_shape_raises_on_too_few_teacher_layers`，断言 n≤n 抛错；B4 修复后该断言必须改）—— 计划中已列为 intentional change。其余 71 个经手算：k=1 或相对 n_params 不等式等，修复后仍绿。

---

## 6. 结论与建议

1. **判定**：结构覆盖度高（功能在），但**证据性不足**（仿真项恒 PASS）。作为→"实验方案工程骨架"合格；作为→"论文跑出结果"不合格。
2. **第一优先**：诚实性改造（offline_demo 标注 + RNG 化 + Gate 区分真实/仿真），避免未来在模拟数据上直接引用"Gate 通过"。
3. **第二优先**：B2 系列 mapper 正确性修复（直接决定 T04/06 数值可信度）。
4. **第三优先**：inference 接口骨架（§53）接入真实 GPU 的前提，也是 §52 禁止的非特权要实现落点。
5. 所有修复沿用 TDD：先写 RED 失败测试 → 修复 → GREEN → 回归 72 测试。

*报告完毕* — 详细逐条对照可在 `plans/` 中复查；修复执行按已批准 PLAN（波浪式并行，T1–T6 并行 / T7-T8 / T9 / T10）。