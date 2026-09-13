# APCS 实验需求拆解

> 来源：`design.md`（2599 行 ICLR 2027 实验方案 V2.1）
>
> 按 §71 Agent 强制执行顺序逐 Task 拆解。每节 = 一个 Task，包含：
>   - **目的**（对应章节号）
>   - **输入**（cfg / 前序产物）
>   - **输出**（reports/runs/<run_id>/<task>/ 下的标准文件）
>   - **Gate 条件**（§5/§6/§7 三层）
>   - **依赖**（前序 task）
>   - **实现要点**（关键算法 / 数值方法）

## 总览

| Task | 章节 | 目的 | 主要输出 | Gate |
|------|------|------|----------|------|
| T00 | §28 | 模型兼容性扫描 | `model_compatibility.json` + G1/G2/G3 判定 | - |
| T01 | §29 | Self-KV Replay 验证注入等价 | `metrics.json` (logit_cosine/max_error/token_agreement) | **Gate 0** PASS |
| T02 | §30 | RoPE round-trip 验证 | 3L×3H×128T 抽样误差 | cosine > 0.9999 |
| T03 | §21, §31 | 4 种层映射策略 | `layer_mapping.json` | - |
| T04 | §32 | Full Ridge 校准 | `metrics.json` (R²/KV cos/attn-output cos) | - |
| T05 | §33 | 跨模型 KV 替换 | Retention/KL/Token Agree/Latency | **Gate 1** ≥ 0.90 |
| T06 | §34 | 低秩 + Shared Basis | `rows[]` (rank × PCR × Retention) | - |
| T07 | §16, §36 | Teacher Gap 分桶 + 触发 PREREGISTRATION | `teacher_gap.json` | - |
| T08 | §22, §36 | Advantage State 训练 | RMS/α/Component 报告 | - |
| T09 | §37, §51 | 主能力实验 | CHG/TGRR/JCR + bootstrap CI + permutation | **Gate 2A** |
| T10 | §4, §38 | 系统成本 | Scenario A/B/C + PSR_A + N_BE + 单卡流程 | - |
| T11 | §39, §69 | MVP 决策 | verdict ∈ {A, B, C, D} | - |
| T12 | §40, §62 | 几何诊断 | CKA/PA/attn-cos/ER/head-corr | - |
| T13 | §11, §44 | 第二 Pair 泛化 | Gate 2A 在 ≥ 2 Pair 上的可复现 | - |

## T00 Compatibility Scanner (§28)

**目的**：自动读取 teacher / student 的架构元数据，判定属于 G1/G2/G3 哪一类实验路径。

**输入**：
- `cfg["teacher"]`：`{model_id, revision, dtype, attention_implementation}`
- `cfg["student"]`：同上

**输出**：
- `model_compatibility.json` — 两份 `ModelSpec` + `compatibility.verdict/note`
- `metrics.json` — `{verdict, matched_kv}`
- `summary.md` — 两边架构对比表

**判定规则**：
- `matched_kv AND same_tokenizer` → `G1_MATCHED_KV`
- `matched_kv AND diff_tokenizer` → `G1_MATCHED_KV_DIFF_TOKENIZER`
- `diff_kv AND same_tokenizer` → `G2_MISMATCHED_HEAD_DIM`
- `diff_kv AND diff_tokenizer` → `G3_CROSS_FAMILY`

**依赖**：无

**实现要点**：
- 优先 `transformers.AutoConfig.from_pretrained`；ImportError 时 fallback 到 Qwen3 已知架构硬编码
- 字段对齐 GPT-2 (`n_embd/n_layer/n_head`) 与 LLaMA/Qwen (`hidden_size/num_*`) 两套命名

## T01 Self-KV Replay (§29, H0, Gate 0)

**目的**：证明 Student 自产 KV 保存-重注入与原生推理**完全等价**，作为后续 Cross-Model Mapper 实验的前置条件。

**输入**：
- 任意 `cfg`（仅作输出位置）
- Student 模型 + 输入 `X`

**输出**：
- `metrics.json` — `{mean_logit_cosine, mean_max_error, mean_token_agreement, gate0}`

**Gate 0**：
- 若任何 sample 的 `max_error` 非零或 `token_agreement < 1.0` → **FAIL**
- FAIL → 禁止进行 Cross-Model Mapper 实验

**依赖**：T00

**实现要点**：
- 双轨：CI 用 `_simulated_replay` 返回 1.0（保证可跑）；真实 GPU 用 `model.generate(past_key_values=...)`
- §52 禁止 8：Cache 注入失败后禁止 Silent Re-prefill（mapper 通过 `track_runtime("silent_re_prefill_on_failure", False)` 标注）

## T02 RoPE Round-trip (§30, §23)

**目的**：证明 `K_rope → de-RoPE → re-RoPE ≈ K_rope` 数值精度足够，让 §23 强制 de-RoPE 路径成立。

**输入**：
- Teacher 的 `theta`（Qwen3 用 1_000_000.0）
- 抽样：3 Layers × 3 Heads × 128 Tokens

**输出**：
- `metrics.json` — `{mean_max_err, mean_cosine, gate}`

**Gate**：cosine > 0.9999

**依赖**：T00

**实现要点**：
- 纯 NumPy 实现 `apply_rope` / `de_rope`（配对旋转）
- 形状约定：支持 `(S, D)` / `(S, H, D)` / `(..., D)`，通过 `positions.reshape(-1, *([1] * (x.ndim - 1)))` 广播

## T03 Layer Alignment (§21, §31, A7)

**目的**：构造 Teacher ↔ Student 的 4 种层映射策略，供后续 mapper 选用。

**输入**：
- Teacher/Student 层数（默认 36 / 28）
- 相似度矩阵 `sim[S, T]`（合成对角带状；真实用 hidden state cos）

**输出**：
- `layer_mapping.json` — `{proportional, last_layer, data_driven_topk, geometry_aware_topk}`，每项是长度 `L_s` 的 list
- `metrics.json` — `{strategy, mapping_size_avg, selected}`

**4 种策略**：
1. **Proportional** — 按比例均摊 Teacher 层（默认）
2. **Last-layer** — 每 Student 层取最末 k 个 Teacher 层
3. **Data-driven top-k** — 基于相似度矩阵 top-k
4. **Geometry-aware top-k** — data-driven + 邻近层平滑

**约束**：§21 `sum_i w_{l,i,h} = 1`

**依赖**：T00

## T04 Ridge Baseline (§32)

**目的**：Full Ridge 校准 100-500 samples @ 512/1K context，输出 mapper size / R² / KV cos / attn-output cos。

**输入**：
- 100-500 calibration 样本（KV 对应关系由 `_synth_calibration_kv` 构造共享 latent）
- context 长度 512/1K

**输出**：
- `metrics.json` — `{n_calib_samples, context_length, mapper_n_params, mean_r2, mean_kv_cosine, mean_attn_output_cosine, latency_map_ms_p50, latency_map_ms_p95, gate}`

**依赖**：T01, T02, T03

**实现要点**：
- §23 强制启用 `de_rope_fn`：`K_teacher → de-RoPE(unrotated) → Mapper → Student RoPE`
- `_check_shape(kv_t, kv_s)` 显式断言，不静默截断
- `_ridge_closed_form_batch`：H 个 head 一次性 batch solve
- `attn_output_cosine`：用随机 Q + 学到的 K 计算 attn logits 分布相似度

## T05 Replacement (§33, Gate 1)

**目的**：跨模型 KV 替换，验证 Replacement Fidelity。

**输入**：
- 训练：100 calibration 样本（§32）
- 测试：20 samples × 4 contexts (512/1K/2K/4K)

**输出**：
- `metrics.json` — `{contexts, retention_per_context, mean_retention, mean_token_agreement, latency_p50_ms, latency_p95_ms, gap_strata, gate1}`

**Gate 1**：
- `mean_retention ≥ 0.90` → **PASS**
- `0.80 ≤ mean_retention < 0.90` → **CONDITIONAL**
- `mean_retention < 0.80` → **FAIL**

**§33 报告项**：Retention / KL / Token Agreement / Latency（§49 P50/P95）

**依赖**：T04

## T06 Lightweight Mapper (§34, Figure 1)

**目的**：Low-rank (8/16/32) + Shared Basis → PCR vs Retention 图（论文 Figure 1）。

**输入**：
- 64 calibration 样本 @ 1K context
- 4 个变体：lowrank-8 / lowrank-16 / lowrank-32 / shared-basis-16（cfg 开关）

**输出**：
- `metrics.json` — `{p_ref_n_params, rows[]}`，每 row `{variant, rank, params, pcr, retention, r2, cosine}`

**§20 Stage 3**：Shared Basis 用共享降维矩阵 `A_shared ∈ R^{D×r}` + 每 (s, h) 独立升维 `B`

**依赖**：T04

## T07 Teacher Gap Freeze (§16, §36, §48)

**目的**：在 Validation split 上计算 Teacher-Student 得分差，按 Low/Med/High 分桶，**冻结** Teacher-Student gap strata，触发 PREREGISTRATION.md 生成（§70）。

**输入**：
- 64 samples：32 train + 32 validation（test 不可用于此阶段，§36）
- Teacher / Student 评估函数

**输出**：
- `teacher_gap.json` — `{rows[], summary}`，summary 含 `{n_total, n_validation, n_train, gap_distribution, mean_gap_validation}`
- 根目录 `PREREGISTRATION.md`（t07 后自动生成，§70）

**桶阈值**：< 0.10 low / < 0.25 medium / ≥ 0.25 high

**依赖**：T00

## T08 Advantage State Training (§22, §36, §7)

**目的**：构造 Advantage Residual R_K / R_V（K/V 独立低秩），校准 RMS，bounded α，Student 全冻结。

**输入**：
- `cfg["advantage"]`：`{rank, key, value, source_mixer, rms_calibration, bounded_alpha}`
- `cfg["mapper"].alpha_max`

**输出**：
- `metrics.json` — `{rank, alpha_max, alpha_layer_mean/std, base_rms, residual_K/V_rms, ratio_K/V, ablation_switches}`

**§36 第一版固定**（不允许改）：
- Source-Layer Mixer + Separate K/V Low-rank + Rank 16 + RMS Calibration + Bounded α + No Query Gate

**§22 公式**：
- R_K = A_K · σ(B_K · Z_K)；R_V 同理
- 当前实现线性化（省 σ）；严格按论文应用 σ=tanh 或 ReLU

**依赖**：T04, T07

## T09 Main Capability (§37, §45, §48, §51, §7)

**目的**：跑 7 种方法对比（Student / Teacher / Text / Ridge / Base Only / Base+Adv / Full APCS），主指标 CHG/TGRR/JCR + bootstrap CI + permutation + Gap strata。

**输入**：
- 32 samples × 3 seeds = 96 组得分
- Teacher / Student 评估函数

**输出**：
- `metrics.json` — `{student_score, teacher_score, teacher_gap, per_method[], chg_bootstrap, chg_permutation, gap_strata, gate2a}`
- `summary.md` — 表 + bootstrap CI + p-value + Gap strata 表

**Gate 2A 完整判定**（§7）：
- CHG > 0
- bootstrap CI 下界 > 0
- TGRR > 0
- permutation p-value < 0.05
- (隐含) Behavior stable + PSR_A > 0

**依赖**：T05, T06, T07, T08

## T10 System Cost (§4, §38, §49, §53, §54)

**目的**：严格区分三种 Scenario + 7 项测量 + §53 单卡执行流程。

**输入**：
- `cfg["context_lengths_extended"]`：[1K, 4K, 8K, 16K]
- `cfg["timing"]`：`{warmup, repeats, sync_cuda}`

**输出**：
- `system.json` — `{contexts, per_context[], vram_teacher_mb_est, ram_student_mb_est, scenario_definitions, single_card_pipeline_order}`

**三种 Scenario 公式**：
- A (Natural): PSR_A = 1 - (T_map + T_load + T_query) / T_prefill_S
- B (For-Transfer): Cost_B = T_prefill_T + T_map + T_load + T_query
- C (One-Teacher-Many): N_BE = ⌈(T_prefill_T + T_map) / (T_prefill_S - T_load - T_query)⌉

**§38 测量项**：teacher_prefill, map, H2D/load, query_prefill, decode, student_full_prefill, VRAM, RAM, cache_bytes

**依赖**：T05, T08

## T11 MVP Decision (§39, §69)

**目的**：读 T05/T09/T10 产物，按 §69 严格顺序输出 verdict ∈ {A, B, C, D}。

**判定顺序**（先 D 再 A 再 B 再 C）：
- `retention < 0.80` → `D_STOP_REPLACEABILITY_UNSTABLE`
- `retention ≥ 0.90 AND chg > 0 AND tgrr > 0 AND psr_a > 0` → `A_RUNTIME_CAPABILITY_TRANSFER`
- `retention ≥ 0.90 AND chg ≤ 0 AND psr_a > 0` → `B_EFFICIENT_STATE_HANDOFF`
- `retention ≥ 0.90 AND chg ≤ 0` → `C_MECHANISM_BOUNDARY`（State Compatibility ≠ Capability Compatibility）
- else → `C_INCONCLUSIVE`

**依赖**：T05, T09, T10

**实现要点**：按 `run_id` 前缀在 base_dir 下找到共享的 T05/T09/T10，**不允许每个 task 单独 run_id**。

## T12 Geometry Diagnostics (§40, §62)

**目的**：每 (layer, head) 报告 5 个几何指标。

**输入**：
- 真实 hidden states（cfg["hidden_states_path"] 可选）→ PCA rank-r 子空间
- 或占位：用合成随机子空间（标注 `placeholder=True`）

**输出**：
- `geometry.json` — `{placeholder, per_layer[], mean_cka, mean_principal_angle, mean_attn_output_cosine, mean_head_correlation, fig7_axes}`
- `metrics.json` — 上述 mean 值
- `summary.md` — 表格 + Figure 7 关联键

**5 指标**：CKA / principal angle / effective rank / attn-output cosine / head correlation

**依赖**：T04

## T13 Generalization (§11, §44, §69)

**目的**：扫描 `reports/runs/` 下所有 run_id 的 T05/T09，列出 (pair, run) 的 Gate 2A 通过情况；论文 Path A 需在 ≥ 2 个 Large→Small Pair 上同时成立。

**输入**：base_dir 下所有 run 的 T05/T09 metrics

**输出**：
- `metrics.json` — `{n_pairs_evaluated, n_pairs_pass_gate2a, gate2a_rate}`
- `summary.md` — 表格 + 若 < 2 Pair 则警告（§11 要求）

**依赖**：T09

**§44 顺序**：Second Pair → Mismatched Head Pilot → Cross-family Pilot（禁止在第一 Pair 未通过 Gate 2 时大量投入 G2/G3）。

---

## 额外 subcommand

| Subcommand | 章节 | 目的 |
|---|---|---|
| `ablation` | §47 | 11 项消融（A1 Rank / A3 Adv / A6 de-RoPE / A8 K/V / A9 RMS / A10 Bounded） |
| `multiturn` | §46, §60 | 1/5/10/20 轮稳定性（CHG/KL/JCR/Latency） |
| `compliance` | §52 | 8 条禁止自动检查器 |

## 全局硬约束（§52 八条禁止）

1. Student 在主实验中重新读取 X → 禁止
2. 微调 Student 主体后仍称 Runtime State Transfer → 禁止
3. Test 调参 → 禁止
4. 只挑 Teacher-win Test Sample → 禁止
5. 隐藏 Teacher Prefill 成本 → 禁止
6. 隐藏 H2D / Cache Load → 禁止
7. 用 R² / Cosine / CKA 代替 CHG → 禁止
8. Cache 注入失败后 Silent Re-prefill → 禁止