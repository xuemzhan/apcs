# KVCache-APCS

> **跨模型 KV Cache 运行时能力迁移实验工程**
>
> 对应论文：**Beyond Cache Emulation: Advantage-Preserving Large-to-Small KV State Handoff without Re-Prefill**
>
> 核心方法：**Advantage-Preserving Cache Synthesis (APCS)**
>
> 当前主模型对：**Qwen3-4B → Qwen3-1.7B**
>
> 实验方案原文：[`design.md`](./design.md)（2599 行 ICLR 2027 实验方案 V2.1）
>
> 拆解与追踪：[`plans/`](./plans/README.md)（requirements · user_stories · traceability · roadmap）

---

## ⚠️ 项目状态声明（务必先读）

**本仓库当前是一个 100% 离线仿真的「协议脚手架」，不是已完成的实验系统。**

- **不加载任何模型权重，不下载任何数据集**：`apcs/` 内没有一处 `from_pretrained` /
  `load_dataset`；`TorchBackend` 显式 `raise NotImplementedError`（§75 诚实性）。
- **所有 Score / 计时 / 几何指标均为合成数据**：
  - T09 的 CHG 来自 `capability/main.py` 的硬编码期望分表（`teacher=0.80 > student=0.50`），
    因此 **CHG > 0 与 Gate 2A 通过是构造出来的必然结果，不是测量结果**；
  - T01 `_simulated_replay` 无条件返回理想值 → **Gate 0 恒 PASS**；
  - T10 计时来自线性公式 → PSR_A / Cost_B / N_BE 均为推导值（§38 禁止当论文结果）；
  - T12 几何指标由随机子空间生成（`placeholder: true`）。
- **仿真结果已被自动标注**：带 `offline_demo` / `placeholder` 的任务，
  其 `task_report.md` 的 STATUS 行会强制显示 `[SIMULATED]`，
  `summary.md` 顶部有醒目警告。
- **`compliance` 的 `passed=True` 不构成合规证据**：八条禁止大多缺少运行时埋点，
  报告中以 `UNKNOWN` 标出检测盲区；只有 `fully_verified=True` 才代表真正验证过。

**因此：本仓库的任何输出都不得作为论文证据引用。** 真实化路径见
[`plans/design-gap-review.md`](./plans/design-gap-review.md)。真正经过验证的部分是
**数学内核**（RoPE round-trip、Ridge 闭式解与聚合等价性、CKA / bootstrap / permutation 等），
它们由 120 个测试覆盖。

---

## 目录

- [1. 项目定位与研究逻辑](#1-项目定位与研究逻辑)
- [2. 核心概念与符号](#2-核心概念与符号)
- [3. 主要特性](#3-主要特性)
- [4. 13 个核心 Task 与 3 个 subcommand](#4-13-个核心-task-与-3-个-subcommand)
- [5. 证据链与 Gate 条件](#5-证据链与-gate-条件)
- [6. §52 八条禁止（合规性）](#6-52-八条禁止合规性)
- [7. 快速开始](#7-快速开始)
- [8. 目录结构](#8-目录结构)
- [9. 任务执行顺序与产物规范](#9-任务执行顺序与产物规范)
- [10. 配置与扩展](#10-配置与扩展)
- [11. §53 单卡推理管线（apcs.inference）](#11-53-单卡推理管线apcsinference)
- [12. 论文 Figure 渲染（§56–§62）](#12-论文-figure-渲染56–62)
- [13. 测试](#13-测试)
- [14. 依赖](#14-依赖)
- [15. 引用](#15-引用)
- [16. 许可](#16-许可)

---

## 1. 项目定位与研究逻辑

本项目**不**以“成功实现跨模型 KV 映射”为最终目标。给定长上下文 `X`、Teacher `T`、Student `S`，Teacher 已经 Prefill `X` 得到 `C_T(X)`，本研究要求 Student **不再重新 Prefill** `X`，而是通过

```
C_T(X)  →  C_S*(X)
```

得到合成状态 `C_S*`，再以 `ŷ = S(q; C_S*(X))` 解码 Query `q`。

研究逻辑必须**严格**按六层证据链推进，禁止在前一层证据没有成立时直接讨论后一层结论：

```
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

每一层在 design.md 与本仓库的对应任务/章节为：

| # | 证据层 | 对应 Task / 章节 |
|---|--------|-----------------|
| 1 | Engineering Correctness | T01（Gate 0）/ T02 / §29 / §30 |
| 2 | Replacement | T05（Gate 1）/ T04 / T06 / §32 / §33 / §34 |
| 3 | Capability Transfer | T09（Gate 2A）/ T07 / T08 / §16 / §22 / §37 |
| 4 | Behavior Stability | T09 内 JCR / §45 / §48 |
| 5 | Geometry Mechanism | T12 / §40–§43 / §62 |
| 6 | System Cost | T10 / §4 / §38 / §49 / §53 / §54 |
| 7 | Generality | T13 / §11 / §44 |

T11 给出 MVP 决策 verdict ∈ {A, B, C, D}，对应 §69 的论文路径选择。

---

## 2. 核心概念与符号

### 2.1 四个核心指标（design.md §3）

| 指标 | 定义 | 含义 |
|------|------|------|
| **Retention** | `Score_handoff / Score_student_self` | Handoff 能否近似替代 Student 自 Prefill |
| **CHG**（Capability Handoff Gain） | `Score_handoff − Score_student_self` | **首要科学端点**；CHG > 0 才算 Runtime Capability Transfer |
| **TGRR**（Teacher Gap Recovery Rate） | `(S_h − S_s) / (S_t − S_s)` | Teacher–Student 原始 gap 被恢复的比例 |
| **PCR**（Parameter Compression Ratio） | `|Mapper_ours| / |Mapper_ridge|` | Mapper 相对 Full Ridge 的参数压缩比 |

### 2.2 系统收益（design.md §4）

```
PSR_A (Scenario A) = 1 − (T_map + T_load + T_query) / T_prefill_S
```

PSR_A **仅**在 Scenario A（Natural Handoff）下定义，禁止用 PSR_A 主张 Scenario B（Teacher-for-Transfer）的端到端收益。

### 2.3 三个部署场景（design.md §4 / §38）

| Scenario | 含义 | 报告指标 |
|----------|------|----------|
| **A – Natural Handoff** | Teacher 早已为别的用途 Prefill；State 是“沉没成本” | PSR_A |
| **B – Teacher-for-Transfer** | 显式让 Teacher Prefill 以向 Student 移交 | Cost_B = T_prefill_T + T_map + T_load + T_query |
| **C – One-Teacher-Many-Student** | 单 Teacher Prefill 摊销给多个 Student | N_BE = ⌈(T_prefill_T + T_map) / (T_prefill_S − T_load − T_query)⌉ |

### 2.4 三个 G 路径（design.md §10 / §11）

| 路径 | 含义 | 对应配置 |
|------|------|----------|
| **G1 – Matched KV** | Teacher/Student 维度一致（matched heads & head_dim） | `configs/pair_qwen3.yaml`（Same Family） |
| **G2 – Mismatched Head Dim** | head_dim 不一致，需 `P_d`（truncate/pad/linear） | `configs/pair_smollm2.yaml`（SmolLM2-1.7B→135M + GQA 比例不同） |
| **G3 – Cross Family** | 不同 Model Family | 第三 Pair 候选（§11 顺序：Second Pair → Mismatched Head Pilot → Cross-family Pilot） |

判定由 T00 Compatibility Scanner 输出 `verdict ∈ {G1_MATCHED_KV, G1_MATCHED_KV_DIFF_TOKENIZER, G2_MISMATCHED_HEAD_DIM, G3_CROSS_FAMILY}`。

---

## 3. 主要特性

- **13 个核心 Task**（T00–T13）+ **3 个 subcommand**（`ablation`、`multiturn`、`compliance`），严格按 §71 顺序执行
- **K/V 独立参数化**（§22）：`mapper.separate_kv: true` 时 K 与 V 各自独立校准/评估，输出 `retention_K/V`、`r2_K/V` 等分项指标（A8 消融）
- **T08 Advantage State Training**：Source-Layer Mixer + K/V 独立低秩残差 + RMS Calibration + Bounded α，Student 全冻结
- **校准样本聚合（§32 bug-3 修复）**：100–500 个 calibration sample 通过 Gram 聚合一次 fit，内存 `O(L_s × H × D²)` 与样本数无关；逐位等价于“拼接后一次 fit”
- **§53 单卡推理管线骨架**：`apcs.inference.HandoffPipeline` + `InferenceBackend`（numpy 可跑 / torch 显式 raise）
- **§72 Orchestrator + §73 Task Report**：DAG 依赖校验、Gate FAIL 阻断（含**传递闭包**）、12 字段 Task Report
- **§70 PREREGISTRATION.md**：T07 后自动生成，冻结 Models / Dataset / Hyperparams / Seeds / Statistics / Gates / Negative Result Policy
- **§52 八条禁止自动化检查器**：静态 cfg 推断 + 运行时信号采集（`apcs.compliance.runtime`），跑每个 task 时同步落 `compliance.json`
- **§64/§65 metadata.json 标准字段自动补全**：model / dataset / commit / run_id 等
- **8 个论文 Figure 渲染脚本**（§56–§62，matplotlib）
- **完整中文注释**：每个模块 docstring + 函数注释 + 行内关键说明，标注 design.md 章节

---

## 4. 13 个核心 Task 与 3 个 subcommand

| Task | 章节 | 标题 | 关键产物 | Gate |
|------|------|------|----------|------|
| **T00** | §28 | Compatibility Scanner | `model_compatibility.json` + G1/G2/G3 判定 | — |
| **T01** | §29 | Self-KV Replay | `metrics.json`（logit cosine / max error / token agreement） | **Gate 0** PASS |
| **T02** | §30 | RoPE Round-trip | `metrics.json`（max_err / cosine） | cosine > 0.9999 |
| **T03** | §21 / §31 | Layer Alignment（4 策略） | `layer_mapping.json` | — |
| **T04** | §32 | Ridge Baseline | `metrics.json`（R² / KV cos / attn-output cos） | — |
| **T05** | §33 | Replacement | Retention / KL / Token Agree / Latency | **Gate 1** ≥ 0.90 |
| **T06** | §34 | Lightweight Mapper（Figure 1） | `metrics.json.rows[]`（rank × PCR × Retention） | — |
| **T07** | §16 / §36 / §70 | Teacher Gap Freeze + PREREG | `teacher_gap.json` + `run_root/PREREGISTRATION.md` | — |
| **T08** | §22 / §36 | Advantage State Training | RMS / α / K、V 分项报告 | — |
| **T09** | §37 / §45 / §48 / §51 | Main Capability（Gate 2A） | CHG / TGRR / JCR + bootstrap CI + permutation + Gap strata | **Gate 2A** |
| **T10** | §4 / §38 / §49 / §53 | System Cost | Scenario A/B/C + PSR_A + N_BE + 单卡流程 | — |
| **T11** | §39 / §69 | MVP Decision | `verdict ∈ {A, B, C, D}` | — |
| **T12** | §40 / §62 | Geometry Diagnostics（Figure 7） | CKA / PA / attn-cos / ER / head-corr | — |
| **T13** | §11 / §44 | Generalization（≥ 2 Pair） | `n_pairs_pass_gate2a` / `gate2a_rate` | ≥ 2 Pair |
| `ablation` | §47 | A1/A3/A6/A8/A9/A10 消融 | `metrics.json.rows[]` | — |
| `multiturn` | §46 / §60 | 多轮稳定性（Figure 5） | `metrics.json.per_turn[]` | — |
| `compliance` | §52 | 8 条禁止检查器 | `metrics.json` + `compliance.json` | violations == 0 |

每个 Task 的入口函数、常量阈值、依赖关系、验收条件详见：
- [`plans/requirements.md`](./plans/requirements.md)（按 §71 顺序的硬约束）
- [`plans/user_stories.md`](./plans/user_stories.md)（As-a/I-want/So-that 端到端 CLI 用法）
- [`plans/design-gap-review.md`](./plans/design-gap-review.md)（design ↔ code 缺口审计）

---

## 5. 证据链与 Gate 条件

### 5.1 Gate 0（Engineering Correctness，§29 / H0）

```
T01 Self-KV Replay：
  - max_error == 0
  - token_agreement == 1.0
```

任意 sample 失败 → **FAIL** → 强制阻断后续 Cross-Model Mapper 实验。

### 5.2 Gate 1（Replacement Fidelity，§33 / §6）

```
T05 Replacement：
  mean_retention ≥ 0.90  → PASS
  0.80 ≤ x < 0.90        → CONDITIONAL
  x < 0.80               → FAIL
```

### 5.3 Gate 2A（Capability Transfer，§7）

```
T09 Main Capability：必须同时满足
  - CHG > 0
  - bootstrap 95% CI 下界 > 0
  - TGRR > 0
  - permutation p-value < 0.05
  - （隐含）Behavior stable + PSR_A > 0
```

### 5.4 MVP Decision（§69，按顺序判定）

```
retention < 0.80                                   → D_STOP_REPLACEABILITY_UNSTABLE
retention ≥ 0.90 AND chg>0 AND tgrr>0 AND psr_a>0  → A_RUNTIME_CAPABILITY_TRANSFER
retention ≥ 0.90 AND chg≤0 AND psr_a>0             → B_EFFICIENT_STATE_HANDOFF
retention ≥ 0.90 AND chg≤0                         → C_MECHANISM_BOUNDARY
else                                               → C_INCONCLUSIVE
```

注意：T11 必须按 `run_id` 前缀在 base_dir 下找到共享的 T05/T09/T10，**不允许**每个 task 单独 run_id。
这一点由 `apcs.io.runs.resolve_run_id` 的**粘性指针**保证（见 §7.2）：
首个 task 把 run_id 写入 `reports/runs/<experiment_name>.current`，后续 task 自动复用。

---

## 6. §52 八条禁止（合规性）

| # | 规则 | 自动化检查信号 |
|---|------|----------------|
| 1 | Student 在主实验中重新读取 X | `student_input_has_context` |
| 2 | 微调 Student 主体后仍称 Runtime State Transfer | `student_params_updated` ∧ `cfg.student.freeze` |
| 3 | Test 调参 | `test_hp_search` |
| 4 | 只挑 Teacher-win Test Sample | `test_filtered_to_teacher_win` |
| 5 | 隐藏 Teacher Prefill 成本 | `reports_teacher_prefill` |
| 6 | 隐藏 H2D / Cache Load | `hides_h2d_load` |
| 7 | 用 R² / Cosine / CKA 代替 CHG | `claim_path_a_on_similarity_only` |
| 8 | Cache 注入失败后 Silent Re-prefill | `silent_re_prefill_on_failure` |

通过 `python -m apcs.cli compliance --config ...` 跑全套；每个 task 通过 `apcs.orchestrator.run_with_compliance` 包装时同步落 `compliance.json`。详见 [`apcs/compliance/`](./apcs/compliance/__init__.py)。

---

## 7. 快速开始

### 7.1 安装

```bash
# Python 3.11+（开发环境实测 3.14）
pip install -e .            # 核心：numpy / scipy / pyyaml，CI 与全部测试可跑
pip install -e ".[dev]"     # + pytest / matplotlib
pip install -e ".[figures]" # 仅论文 Figure 渲染
# TODO(real-gpu): 真实 GPU 实验（当前代码不加载模型，TorchBackend 显式 raise）
pip install -e ".[gpu]"     # + torch / transformers / datasets / psutil
```

安装后可直接使用 `apcs` 命令（等价于 `python -m apcs.cli`）。

### 7.2 主实验：Qwen3-4B → Qwen3-1.7B

§71 强制顺序执行（一次一个 task）。**run_id 自动共享**：首个 task 建立
`reports/runs/<experiment_name>.current` 指针，后续 task 自动复用同一 run 目录，
无需手动传 `--run-id`。

```bash
python -m apcs.cli t00 --config configs/pair_qwen3.yaml   # 建立本次实验的 run_id
python -m apcs.cli t01 --config configs/pair_qwen3.yaml   # Gate 0
python -m apcs.cli t02 --config configs/pair_qwen3.yaml
python -m apcs.cli t03 --config configs/pair_qwen3.yaml
python -m apcs.cli t04 --config configs/pair_qwen3.yaml
python -m apcs.cli t05 --config configs/pair_qwen3.yaml   # Gate 1
python -m apcs.cli t06 --config configs/pair_qwen3.yaml
python -m apcs.cli t07 --config configs/pair_qwen3.yaml   # t07 后自动生成 PREREGISTRATION.md
python -m apcs.cli t08 --config configs/pair_qwen3.yaml
python -m apcs.cli t09 --config configs/pair_qwen3.yaml   # Gate 2A
python -m apcs.cli t10 --config configs/pair_qwen3.yaml
python -m apcs.cli t12 --config configs/pair_qwen3.yaml
python -m apcs.cli t11 --config configs/pair_qwen3.yaml   # MVP verdict
python -m apcs.cli t13 --config configs/pair_qwen3.yaml   # §11 Generalization
```

**§72 准入检查（自动强制）**：CLI 在执行前校验依赖闭包，
若前置 task 未跑或未通过 Gate，则拒绝执行并返回**退出码 2**：

```
[apcs] BLOCKED: 缺少前置 task（尚未执行）：t04
[apcs] §72 要求按依赖顺序执行；如确需跳过请显式加 --force。
```

| 退出码 | 含义 |
|---|---|
| 0 | task 执行且 status ∈ {PASS, OK} |
| 1 | task 执行了但 Gate 未通过（FAIL / CONDITIONAL） |
| 2 | **§72 准入阻断**：依赖缺失或前置 Gate FAIL，task 未执行 |

常用参数：

| 参数 | 作用 |
|---|---|
| `--run-id <id>` | 显式指定 run_id（并把指针对齐到它） |
| `--new-run` | 强制开启新实验（生成新 run_id，重置指针） |
| `--force` | 跳过 §72 准入检查（仅调试用） |
| `--no-prereg` | t07 时跳过 PREREGISTRATION.md 生成 |

### 7.3 第二 Pair（§11，不同 Model Family）

```bash
python -m apcs.cli t05 --config configs/pair_smollm2.yaml
python -m apcs.cli t09 --config configs/pair_smollm2.yaml
```

### 7.4 三个 subcommand

```bash
python -m apcs.cli ablation   --config configs/pair_qwen3.yaml   # §47
python -m apcs.cli multiturn  --config configs/pair_qwen3.yaml   # §46 / Figure 5
python -m apcs.cli compliance --config configs/pair_qwen3.yaml   # §52
```

### 7.5 渲染论文 Figure

```python
from apcs.figures import render_all
render_all({"t05": ..., "t06": ..., "t09": ...}, "reports/figures")
```

详见 [`apcs/figures/__init__.py`](./apcs/figures/__init__.py)，共 8 张图：

| Figure | 内容 | 依赖 Task |
|--------|------|-----------|
| Fig. 1 | PCR vs Retention | T04–T06 |
| Fig. 2 | Teacher Gap ↔ CHG / TGRR | T07 / T09 |
| Fig. 3 | CHG–PSR_A Pareto（仅 Scenario A） | T09 / T10 |
| Fig. 4 | Context Length Scaling | T10 |
| Fig. 5 | Multi-turn Stability | multiturn |
| Fig. 6 | Ablation | ablation |
| Fig. 7 | Geometry（7a CKA / 7b PA-CHG / 7c attn-cos-Retention / 7d ER-CHG） | T12 |

### 7.6 跑测试

```bash
python -m pytest tests/
```

---

## 8. 目录结构

```
KVCache/
├── design.md                         # 实验方案原始论文版本（2599 行）
├── README.md                         # 本文件
├── PREREGISTRATION.md                # §70 T07 后自动生成（提交到 repo）
├── plans/                            # 需求 / 用户故事 / 追踪 / 路线图
│   ├── README.md
│   ├── requirements.md               # 13 个 Task 的硬约束
│   ├── user_stories.md               # 13+3 个端到端用户故事
│   ├── design-gap-review.md          # design ↔ code 缺口审计
│   └── traceability.md / roadmap.md  # 三向追踪 + 6 周节奏
├── configs/
│   ├── pair_qwen3.yaml               # Qwen3-4B → 1.7B 主实验
│   └── pair_smollm2.yaml             # SmolLM2-1.7B → 135M 第二 Pair
├── apcs/
│   ├── cli.py                        # CLI 入口（§63 stdout.log + §73 Task Report）
│   ├── compat/        T00            # §28 Compatibility Scanner
│   ├── replay/        T01            # §29 Self-KV Replay (Gate 0)
│   ├── rope/          T02            # §30 RoPE + de-RoPE 实现
│   ├── alignment/     T03            # §21 Layer Alignment 4 策略
│   ├── mapper/        T04–T06        # §20 Ridge / LowRank / SharedBasis + §32 bug-3 聚合
│   │   ├── runner.py                  # run_ridge_baseline / run_replacement / run_lightweight_mapper
│   │   ├── math.py                    # RidgeMapper / RidgePerHeadMapper / LowRank / SharedBasis
│   │   ├── aggregate.py               # §32 bug-3：fit_ridge_aggregate / concat_kv_samples
│   │   └── mismatched.py              # §12 G2 P_H / P_D head_dim 维度不对齐
│   ├── inference/                   # §53 单卡推理管线骨架
│   │   ├── pipeline.py               # HandoffPipeline：8 阶段计时 + Student zero-prefill 断言
│   │   └── backends.py               # InferenceBackend / NumpyBackend / NumpyFakeModel / TorchBackend
│   ├── advantage/     T08            # §22 Advantage State
│   ├── capability/    T07 / T09      # §16 Teacher Gap + §37 Main Capability
│   ├── system/        T10            # §4 / §38 Scenario A/B/C
│   ├── geometry/      T12            # §40 Geometry Diagnostics
│   ├── decision/      T11            # §69 MVP Decision
│   ├── generalization/  T13          # §11 / §44 多 Pair Gate 2A 通过率
│   ├── ablation/                     # §47 A1/A3/A6/A8/A9/A10
│   ├── multiturn/                    # §46 Multi-turn Stability
│   ├── compliance/                   # §52 8 条禁止检查器 + 运行时信号
│   │   ├── __init__.py               # check_1..check_8 + check_all + compliance_report
│   │   └── runtime.py                # 运行时信号 reset / track / collect / 静态推断
│   ├── prereg/                       # §70 PREREGISTRATION.md 生成器
│   ├── orchestrator/                 # §71–§73 DAG 依赖（含传递闭包）+ 12 字段 Task Report
│   ├── figures/                      # §56–§62 8 张 Figure 渲染
│   ├── data/                         # §14–§18 数据集
│   ├── metrics/                      # §3 / §4 / §45 / §48 / §51 全部指标
│   ├── io/                           # 配置加载 + Run 输出 + metadata (§64/§65)
│   └── utils/                        # seed / 计时 / percentile
├── reports/
│   ├── figures/                      # 论文 Figure 输出
│   └── runs/                         # 每个 Run 的产物（§63）
│       └── <run_id>/<task>/
│           ├── config.json
│           ├── metrics.json
│           ├── system.json           # 仅 T10
│           ├── geometry.json         # 仅 T12
│           ├── compliance.json       # 通过 run_with_compliance 包装时生成
│           ├── metadata.json         # §64/§65 标准字段补全
│           ├── summary.md
│           ├── stdout.log            # §63 CLI 捕获的 stdout/stderr
│           └── task_report.md        # §73 12 字段
└── tests/                            # 120 个单元 + 集成测试
    └── test_arch_fixes.py            # 架构审查修复的回归测试（P0-1/2/3, P1-2/4）
```

---

## 9. 任务执行顺序与产物规范

### 9.1 §71 强制顺序（DAG）

```
T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08 → T09 → {T10, T12, T11} → T13
                                            ↓
                                  PREREGISTRATION.md (自动)
```

详细依赖：

```
T00  : -
T01  : T00
T02  : T00
T03  : T00
T04  : T01, T02, T03
T05  : T04
T06  : T04
T07  : T00
T08  : T04, T07
T09  : T05, T06, T07, T08
T10  : T05, T08
T12  : T04
T11  : T05, T09, T10
T13  : T09
ablation  : T04, T08
multiturn : T09
compliance: -
```

**§72 强制规则**：一次只允许执行一个 Task；Gate FAIL 后**不能跳过**（含传递闭包）；Test 不得用于 Auto-Hyperparameter Search；所有 Negative Result 必须保留。

### 9.2 §63 Run 产物规范

每个 Run 写到：

```
reports/runs/<run_id>/<task>/
├── config.json      # 完整 cfg（合并默认 + 用户）
├── metrics.json     # 任务主指标
├── system.json      # T10 系统计时/显存/带宽
├── geometry.json    # T12 几何指标
├── compliance.json  # §52 检查结果（通过 run_with_compliance 包装）
├── metadata.json    # §64/§65 标准字段（自动补全）
├── summary.md       # 任务摘要（Markdown 表）
├── stdout.log       # §63 CLI 捕获的 stdout/stderr
└── task_report.md   # §73 12 字段报告
```

`run_id` 默认取自 `cfg.experiment.run_id`（如 `qwen3-4b-to-1.7b-20260808-195506`），可用 `--run-id` 覆盖。**T11 / T13 必须按 run_id 前缀共享数据。**

### 9.3 §73 Task Report 12 字段

```
TASK_ID, STATUS, OBJECTIVE, MODEL_PAIR, DATASET, CONFIG,
IMPLEMENTATION, OUTPUT_FILES, KEY_METRICS, STATISTICAL_CHECK,
BEHAVIOR_CHECK, GEOMETRY_CHECK, SYSTEM_COST, ACCEPTANCE_CRITERIA,
RESULT, FAILURE_ANALYSIS, NEXT_ALLOWED_TASK
```

---

## 10. 配置与扩展

### 10.1 顶层 cfg 节

| 节 | 含义 | 示例 |
|----|------|------|
| `experiment` | run 名 / id / 描述 | `name: qwen3-4b-to-1.7b` |
| `teacher` / `student` | 模型 id / revision / dtype / attention / device_map / freeze / 显式架构字段 | `freeze: true`（T08 强制） |
| `datasets` | fidelity / teacher_advantage / long_context / behavior_sensitive | `teacher_advantage.primary: mmlu` |
| `context_lengths` | 主实验上下文长度 | `[512, 1024, 2048, 4096]` |
| `context_lengths_extended` | P1 扩展 | `[8192, 16384]` |
| `mapper` | ridge / lowrank / shared_basis；rank / separate_kv / de_rope / source_top_k / layer_selection / alpha_max / rms_calibration | `separate_kv: true` |
| `advantage` | rank / key / value / source_mixer / rms_calibration / bounded_alpha | K、V 均 `lowrank` |
| `query_gate` | A11 默认关闭 | `enabled: false` |
| `loss` | lambda_task / lambda_self / lambda_teacher / lambda_att / lambda_reg | `lambda_att: 1.0`（优先于 raw KV MSE） |
| `seeds` / `statistics` | 多种子 + bootstrap_n / ci / paired_test | `seeds: [0,1,2]`、`ci: 0.95` |
| `timing` | warmup / repeats / sync_cuda / report | `report: [p50, p95]` |
| `cache_residency` | R0 gpu / R1 cpu / R2 nvme | `cpu` |
| `kv_save_full_debug` | §54 全 KV 保存上限 | `max_samples: 10` |
| `output` | base_dir / files 清单 | `base_dir: reports/runs` |
| `gates` | retention_min / retention_strong / chg_positive / psr_a_positive | `retention_min: 0.90` |

### 10.2 添加新 Model Pair

1. 拷贝 `configs/pair_smollm2.yaml` → `configs/pair_<name>.yaml`
2. 改 `teacher` / `student` 的 `model_id`、`num_layers`、`num_kv_heads`、`head_dim`、`num_attention_heads`、`vocab_size`
3. 若 head_dim 不一致：开启 `mapper.mismatched` 维度调整（§12）
4. T00 跑出 `verdict`（G1/G2/G3），再按 §71 顺序执行 T01–T11
5. T13 自动汇总所有 `reports/runs/*/t05 / t09` 的 Gate 2A 通过率

---

## 11. §53 单卡推理管线（`apcs.inference`）

`apcs.inference` 提供可测试接口骨架，覆盖 §53 单卡执行策略：

```
Teacher Load → Forward → Capture → CPU Offload →
Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode
```

### 11.1 端到端用法

```python
from apcs.inference import HandoffPipeline, NumpyBackend
from apcs.mapper.math import RidgeMapper
from apcs.alignment.runner import build_proportional_mapping

mapper = RidgeMapper(num_layers=28, num_heads=8, head_dim=128, lam=1e-2)
layer_map = build_proportional_mapping(L_t=36, L_s=28)

pipe = HandoffPipeline(NumpyBackend(), mapper, layer_map, cfg, out_dir)
metrics = pipe.run(teacher_tokens, student_prefix, n_gen=64)
```

### 11.2 后端契约

```python
class InferenceBackend:
    name: str
    def load_model(self, run): ...
    def forward_prefill(self, model, tokens): ...   # 返回 past_key_values
    def decode(self, model, tokens, past_key_values): ...  # 只消费注入 KV + 当前 token
    def inject(self, model, kv): ...               # 把映射后 (L_s, S, H, D) 注入 Student
    def unload(self, model): ...
    def sync(self): ...                            # §49 计时边界同步
```

- `NumpyBackend` / `NumpyFakeModel`：`offline_demo` 标注，可跑、可测
- `TorchBackend`：骨架接口显式 `raise NotImplementedError`（§75 诚实性），不可静默 mock
- `HandoffPipeline` 严格按 8 阶段顺序；Student 在 `decode` 阶段**不重新读 X**（zero prefill），用计数器做统计断言

详见 [`apcs/inference/pipeline.py`](./apcs/inference/pipeline.py) 与 [`backends.py`](./apcs/inference/backends.py)。

---

## 12. 论文 Figure 渲染（§56–§62）

| Figure | 子图 | 数据来源 | 渲染函数 |
|--------|------|----------|----------|
| **Fig. 1** PCR vs Retention | — | `t06_metrics.rows[]` | `fig1_pcr_retention` |
| **Fig. 2** Teacher Gap ↔ CHG / TGRR | — | `t07_metrics` + `t09_metrics` | `fig2_teacher_gap` |
| **Fig. 3** CHG–PSR_A Pareto | — | `t09_metrics` + `t10_metrics` | `fig3_chg_psra_pareto` |
| **Fig. 4** Context Length Scaling | — | `t10_metrics.per_context[]` | `fig4_context_scaling` |
| **Fig. 5** Multi-turn Stability | — | `multiturn_metrics.per_turn[]` | `fig5_multiturn` |
| **Fig. 6** Ablation | — | `ablation_metrics.rows[]` | `fig6_ablation` |
| **Fig. 7** Geometry | 7a CKA heatmap / 7b PA×CHG / 7c attn-cos×Retention / 7d ER×CHG | `t12_geometry.per_layer[]` | `fig7_geometry` |

调用：

```python
from apcs.figures import render_all
render_all(
    {"t06": t06_metrics, "t09": t09_metrics, "t10": t10_metrics, "t12": t12_geometry},
    Path("reports/figures"),
)
```

---

## 13. 测试

```bash
python -m pytest tests/   # 120 个测试，全部通过
```

测试覆盖：

- 指标模块（Retention / CHG / TGRR / PCR / PSR_A / JCR / KL / CKA / principal angle / effective rank 等）
- RoPE 圆环精度（`max_err < 1e-10`、`cosine > 0.999999`）
- Mapper 数学正确性（einsum 加速 vs 朴素循环，`atol=1e-5`）
- Batch ridge 数学等价性（`atol=1e-9`）
- K/V 独立参数化（`separate_kv`：K/V 各自 fit/transform，键空间隔离）
- **§32 bug-3 校准聚合等价性**：`fit_ridge_aggregate` 与“concat 后一次 fit”逐位等价（`maxdiff < 1e-4`）
- §52 八条禁止检查器
- §64/§65 metadata schema
- CLI 端到端 + T11 verdict + Orchestrator **传递闭包阻断**（`next_allowed` 闭包检查）
- Inference pipeline 8 阶段计时 + Student zero-prefill 断言
- Compliance runtime 信号采集（`reset` / `track` / `collect`）

最近一次基线：`120 passed in 71.52s`。

---

## 14. 依赖

- **Python 3.11+**（开发环境实测 3.14）
- **核心**（CI / CPU 可跑）：`numpy`, `scipy`, `pyyaml` → `pip install -e .`
- **真实 GPU 实验**：`torch`, `transformers`, `datasets`, `psutil` → `pip install -e ".[gpu]"`
  （TODO(real-gpu)：当前代码不加载模型，`TorchBackend` 显式 raise）
- **论文 Figure 渲染**：`matplotlib` → `pip install -e ".[figures]"`
- 依赖声明见 [`pyproject.toml`](./pyproject.toml)；CI 见 [`.github/workflows/tests.yml`](./.github/workflows/tests.yml)
- **最低推荐硬件**（design.md 头部）：1×24GB NVIDIA GPU + 64–128GB RAM + 1–2TB NVMe

---

## 15. 引用

论文原文 [`design.md`](./design.md)（2599 行 ICLR 2027 实验方案 V2.1）描述了完整的实验方案、gate、metric、figure 与论文路径。

需求/用户故事/追踪/路线图：[`plans/`](./plans/README.md)。

---

## 16. 许可

本仓库采用 **MIT License**（详见 [`LICENSE`](./LICENSE)）。
你可以在保留版权声明的前提下自由使用、修改、分发本代码。

> ⚠️ **研究代码诚信声明**：本仓库当前为 100% 离线仿真脚手架
> （详见顶部项目状态声明）；
> 若用于论文级实验，必须由作者自行接入真实 Teacher/Student 模型权重
> 与评测数据集，并重新校准所有指标。