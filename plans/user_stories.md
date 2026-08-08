# APCS 用户故事（User Stories）

> 每个用户故事采用 **As a / I want / So that** 格式，便于产品化讨论。
>
> 来源：design.md 全部章节 + plans/requirements.md。
>
> 共 13 个核心故事 + 3 个 subcommand 故事。

---

## US-01 Compatibility Scanner

**As a** 实验工程师
**I want** 自动读取 teacher / student 的模型架构（层数 / heads / head_dim / vocab / RoPE config / dtype）
**So that** 在跑任何实验前能立即知道当前 pair 属于 G1 / G2 / G3 哪条路径，并决定后续需要哪些 mapper（matched、mismatched、cross-family）

**CLI**：`python -m apcs.cli t00 --config configs/pair_qwen3.yaml`
**输出**：`model_compatibility.json` + `summary.md`
**章节**：§28
**模块**：`apcs.compat.scanner`

**验收**：
- GIVEN 任意 Qwen3-4B / Qwen3-1.7B 配置
- WHEN 跑 `t00`
- THEN verdict = `G1_MATCHED_KV`（同 family + matched heads/head_dim）

---

## US-02 Self-KV Replay（Gate 0）

**As a** 论文作者
**I want** 证明 Student 自产 KV 保存-注入与原生推理完全等价
**So that** 后续 cross-model mapper 实验才有意义（§52 禁止 1：Student 不能在主实验中重新读 X）

**CLI**：`python -m apcs.cli t01 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（logit cosine / max error / token agreement）
**章节**：§29 / §5 / H0 / Gate 0
**模块**：`apcs.replay.runner`

**验收**：
- Gate 0 PASS → 继续 T02-T13
- Gate 0 FAIL → **强制阻断**后续 Cross-Model Mapper 实验

---

## US-03 RoPE Round-trip

**As a** 论文作者
**I want** 验证 `K_rope → de-RoPE → re-RoPE ≈ K_rope` 的数值精度
**So that** §23 强制 de-RoPE 路径可以成立（Mapper 在 unrotated 空间工作）

**CLI**：`python -m apcs.cli t02 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（mean max err < 1e-10、cosine > 0.9999）
**章节**：§30
**模块**：`apcs.rope.runner`

**验收**：3 Layers × 3 Heads × 128 Tokens 抽样 round-trip cosine > 0.9999

---

## US-04 Layer Alignment（4 种策略）

**As a** 实验工程师
**I want** 在 4 种层映射策略（Proportional / Last-layer / Data-driven top-k / Geometry-aware top-k）间切换
**So that** 完成 §21 / A7 消融并选定主实验默认策略

**CLI**：`python -m apcs.cli t03 --config configs/pair_qwen3.yaml`
**输出**：`layer_mapping.json`（4 策略并存）+ `metrics.json`
**章节**：§21 / §31 / A7
**模块**：`apcs.alignment.runner`

**约束**：`sum_i w_{l,i,h} = 1`（§21）

---

## US-05 Ridge Baseline

**As a** 实验工程师
**I want** Full Ridge 校准在 100-500 samples @ 512/1K context
**So that** 得到 baseline mapper size / R² / KV cos / attn-output cos，作为 §19 B3 的 baseline

**CLI**：`python -m apcs.cli t04 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（mapper_n_params, mean_r2, mean_kv_cosine, mean_attn_output_cosine, latency_map_ms_p50/p95）
**章节**：§32
**模块**：`apcs.mapper.runner` (`run_ridge_baseline`)

**关键**：必须启用 §23 de-RoPE 路径；attn-output cosine 必须报告（§32）

---

## US-06 Replacement（Gate 1）

**As a** 论文作者
**I want** 跨模型 KV 替换后在 4 个 context 长度上报告 Retention / KL / Token Agreement / Latency
**So that** 验证 Replacement Fidelity（§33 Gate 1）

**CLI**：`python -m apcs.cli t05 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（`gate1: PASS/CONDITIONAL/FAIL` + `mean_retention` + `gap_strata`）
**章节**：§33 / §6 / §48
**模块**：`apcs.mapper.runner` (`run_replacement`)

**Gate 1 阈值**：
- `≥ 0.90` → PASS
- `0.80 ≤ x < 0.90` → CONDITIONAL
- `< 0.80` → FAIL

---

## US-07 Lightweight Mapper（PCR vs Retention，Figure 1）

**As a** 论文作者
**I want** Low-rank (8/16/32) + Shared Basis（Stage 3）的 PCR vs Retention 曲线
**So that** 画 Figure 1：mapper 可以压缩多少而不破坏 Replacement？

**CLI**：`python -m apcs.cli t06 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json.rows[]`（variant, rank, params, pcr, retention, r2, cosine）
**章节**：§34 / §20 Stage 2/3 / A1/A2
**模块**：`apcs.mapper.runner` (`run_lightweight_mapper`)

---

## US-08 Teacher Gap Freeze（PREREGISTRATION 触发）

**As a** 论文作者
**I want** 在 Validation split 上把 Teacher-Student gap 分到 Low/Med/High 三桶并冻结
**So that** §36 / §70 pre-registration 之后不被 test 数据污染超参搜索；t07 后自动生成 PREREGISTRATION.md

**CLI**：`python -m apcs.cli t07 --config configs/pair_qwen3.yaml`
**输出**：`teacher_gap.json` + `run_root/PREREGISTRATION.md`（自动）
**章节**：§16 / §36 / §48 / §70
**模块**：`apcs.capability.runner` + `apcs.prereg`

**Acceptance**：
- Train/Validation/Test split 严格分层
- Test 数据集**不可**用于调整 α_max / rank / top-k / loss weights

---

## US-09 Advantage State Training

**As a** 论文作者
**I want** 训练 Source-Layer Mixer + K/V 独立低秩残差 + RMS Calibration + Bounded α
**So that** Advantage State `C_S* = C_base + α_l · R_adv` 可注入冻结 Student

**CLI**：`python -m apcs.cli t08 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（alpha_max, α_layer mean/std, base_rms, residual_K/V_rms, ratio_K/V）
**章节**：§22 / §36 / §7
**模块**：`apcs.advantage.runner`

**§36 第一版固定**：
- Source-Layer Mixer + Separate K/V Low-rank + Rank 16 + RMS Calibration + Bounded α + No Query Gate
- Student 主体：全冻结（§52 禁止 2 强制）

---

## US-10 Main Capability（Gate 2A）

**As a** 论文作者
**I want** 跑 7 种方法（Student / Teacher / Text / Ridge / Base Only / Base+Adv / Full APCS）3 seeds + bootstrap CI + permutation
**So that** 主结果支持 Path A（Runtime Capability Transfer）

**CLI**：`python -m apcs.cli t09 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（per_method[], chg_bootstrap, chg_permutation, gap_strata, gate2a）
**章节**：§37 / §45 / §48 / §51 / §7
**模块**：`apcs.capability.main`

**Gate 2A 完整判定**：
- CHG > 0
- bootstrap 95% CI 下界 > 0
- TGRR > 0
- permutation p-value < 0.05

---

## US-11 System Cost（Scenario A/B/C）

**As a** 论文作者 / 系统工程师
**I want** 严格区分 Scenario A（Natural Handoff）/ B（Teacher-for-Transfer）/ C（One-Teacher-Many-Student）
**So that** 不犯 §4 关键错误：用 PSR_A 声称 Scenario B 有端到端收益

**CLI**：`python -m apcs.cli t10 --config configs/pair_qwen3.yaml`
**输出**：`system.json`（per_context[], PSR_A, Cost_B, N_BE, cache_bytes, single_card_pipeline_order）
**章节**：§4 / §38 / §49 / §53 / §54
**模块**：`apcs.system.runner`

**关键约束**：
- §53 单卡执行顺序：Teacher Load → Forward → Capture → CPU Offload → Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode
- §54 KV 存储：禁止 Dataset × All Layers × All Heads × All Tokens 全保存；streaming；debug KV ≤ 10 samples

---

## US-12 MVP Decision

**As a** 论文作者
**I want** T11 根据 T05/T09/T10 自动判定 Path A/B/C/D
**So that** 在写论文前就明确知道：主路径成立 / 退化为 Efficient Handoff / 走 Mechanism Boundary / Stop

**CLI**：`python -m apcs.cli t11 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json.verdict` ∈ {A_RUNTIME_CAPABILITY_TRANSFER, B_EFFICIENT_STATE_HANDOFF, C_MECHANISM_BOUNDARY, D_STOP_REPLACEABILITY_UNSTABLE}
**章节**：§39 / §69
**模块**：`apcs.decision.runner`

**关键**：按 run_id 前缀共享数据；不允许每个 task 单独 run_id（避免读取路径找不到）。

---

## US-13 Geometry Diagnostics（Figure 7）

**As a** 论文作者
**I want** 每 (layer, head) 报告 CKA / principal angle / attn-output cosine / effective rank / head correlation
**So that** 画 Figure 7：Geometry ↔ Capability 的关联图（机制分析核心图）

**CLI**：`python -m apcs.cli t12 --config configs/pair_qwen3.yaml`
**输出**：`geometry.json` + `summary.md`（含 Fig.7 关联键）
**章节**：§40-§43 / §62
**模块**：`apcs.geometry.runner`

**Figure 7 子图**：
- 7a: Layer × Layer CKA heatmap
- 7b: Principal Angle × CHG
- 7c: Attention-output Cosine × Retention
- 7d: Effective Rank × CHG

---

## US-14 Generalization（§11 第二 Pair 必做）

**As a** 论文作者
**I want** T13 扫描所有 run_id 的 T05/T09，列出每个 (model pair, run) 的 Gate 2A 通过情况
**So that** 验证 Path A 在 ≥ 2 个 Large→Small Pair 上同时成立（论文 §69 强主张之一）

**CLI**：`python -m apcs.cli t13 --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（n_pairs_evaluated, n_pairs_pass_gate2a, gate2a_rate）+ Markdown 表格
**章节**：§11 / §44 / §69
**模块**：`apcs.generalization.runner`

**Acceptance**：
- `n_pairs_evaluated ≥ 2` 且全部 `gate2a_pass = True` → 论文 Path A 强主张成立
- `n_pairs_evaluated < 2` → 警告：需运行 `configs/pair_smollm2.yaml`

---

## US-15 Ablation（§47）

**As a** 论文审稿人
**I want** 跑 11 项消融：A1 Rank / A3 Adv / A6 de-RoPE / A8 K/V / A9 RMS / A10 Bounded
**So that** 每个核心组件都有单独的消融数据支撑（论文 §47 必做）

**CLI**：`python -m apcs.cli ablation --config configs/pair_qwen3.yaml`
**输出**：`metrics.json.rows[]`（ablation, setting, mean_retention, note）
**章节**：§47
**模块**：`apcs.ablation.runner`

---

## US-16 Multi-turn Stability（§46 / Figure 5）

**As a** 论文作者
**I want** 跑 1/5/10/20 轮对话，记录 CHG / KL / JCR / task score / latency
**So that** 画 Figure 5：多轮稳定性曲线（CHG / KL / JCR 三条）

**CLI**：`python -m apcs.cli multiturn --config configs/pair_qwen3.yaml`
**输出**：`metrics.json.per_turn[]`（turn, chg_mean, kl_mean, jcr_mean, task_score_mean, latency_p50_ms）
**章节**：§46 / §60
**模块**：`apcs.multiturn.runner`

---

## US-17 Compliance Check（§52 八条禁止）

**As a** 实验伦理审查者
**I want** 自动检查实验是否违反 §52 八条禁止
**So that** 在论文投稿前能自动审计所有违规项

**CLI**：`python -m apcs.cli compliance --config configs/pair_qwen3.yaml`
**输出**：`metrics.json`（n_violations, violations[], passed）+ `compliance.json`（含运行时信号）
**章节**：§52
**模块**：`apcs.compliance` + `apcs.compliance.runtime`

**8 条规则**：
1. Student 不重读 X
2. cfg.student.freeze=True 时 Student 参数不被更新
3. Test 阶段不调参
4. 不筛选 Teacher-win Test Sample
5. PSR_A 报告必须含 teacher_prefill
6. 不隐藏 H2D/Cache Load
7. 不能仅用相似度指标主张 Path A
8. Cache 注入失败后禁止 Silent Re-prefill