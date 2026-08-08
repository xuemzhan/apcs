# KVCache-APCS

> **跨模型 KV Cache 运行时能力迁移实验工程**
>
> 对应论文：**Beyond Cache Emulation: Advantage-Preserving Large-to-Small KV State Handoff without Re-Prefill**
>
> 核心方法：**Advantage-Preserving Cache Synthesis (APCS)**
>
> 当前主模型对：**Qwen3-4B → Qwen3-1.7B**
>
> 实验方案原文：[`design.md`](./design.md)

---

## 简介

本项目实现并验证 APCS（优势保持缓存合成）：在不重新 Prefill 长上下文的前提下，把 Teacher 模型的 KV 状态迁移到 Student 模型，并证明 Teacher 相对 Student 的能力优势能被运行时状态被冻结的 Student 消费。

研究逻辑严格按六层证据链：**工程正确 → Replacement → Capability Transfer → Behavior Stability → Geometry Mechanism → System Cost**。

## 主要特性

- **13 个核心 Task**（T00–T13）：从 Compatibility Scanner → Mapper → Advantage State → Main Capability → MVP Decision → Generalization
- **3 个额外 subcommand**：`ablation`（§47）、`multiturn`（§46）、`compliance`（§52）
- **K/V 独立参数化**（§22）：`mapper.separate_kv: true` 时 K 与 V 各自独立校准/评估，输出 `retention_K/V`、`r2_K/V` 等分项指标（A8 消融）
- **T08 Advantage State Training**：Source-Layer Mixer + K/V 独立低秩残差（解析梯度） + RMS Calibration + Bounded α
- **8 个论文 Figure** 渲染脚本（§56-§62）
- **§64/§65 标准字段自动补全**（metadata.json schema 校验）
- **§52 八条禁止** 自动化检查器 + 运行时信号采集
- **§70 PREREGISTRATION.md** 在 T07 后自动生成
- **§71-§73 Orchestrator**：DAG 依赖校验、Gate FAIL 阻断、12 字段 Task Report
- **完整中文注释**：每个模块 docstring + 函数注释 + 行内关键说明，标注 design.md 章节

## 快速开始

```bash
# 强制顺序：T00 → T01 → ... → T13
python -m apcs.cli t00 --config configs/pair_qwen3.yaml
python -m apcs.cli t01 --config configs/pair_qwen3.yaml
python -m apcs.cli t02 --config configs/pair_qwen3.yaml
python -m apcs.cli t03 --config configs/pair_qwen3.yaml
python -m apcs.cli t04 --config configs/pair_qwen3.yaml
python -m apcs.cli t05 --config configs/pair_qwen3.yaml
python -m apcs.cli t06 --config configs/pair_qwen3.yaml
python -m apcs.cli t07 --config configs/pair_qwen3.yaml   # t07 后自动生成 PREREGISTRATION.md
python -m apcs.cli t08 --config configs/pair_qwen3.yaml
python -m apcs.cli t09 --config configs/pair_qwen3.yaml
python -m apcs.cli t10 --config configs/pair_qwen3.yaml
python -m apcs.cli t12 --config configs/pair_qwen3.yaml
python -m apcs.cli t11 --config configs/pair_qwen3.yaml
python -m apcs.cli t13 --config configs/pair_qwen3.yaml   # §11 Generalization

# 第二 Pair：不同 Model Family（§11）
python -m apcs.cli t05 --config configs/pair_smollm2.yaml
python -m apcs.cli t09 --config configs/pair_smollm2.yaml

# 额外 subcommand
python -m apcs.cli ablation   --config configs/pair_qwen3.yaml   # §47 消融
python -m apcs.cli multiturn  --config configs/pair_qwen3.yaml   # §46 多轮
python -m apcs.cli compliance --config configs/pair_qwen3.yaml   # §52 八条禁止检查

# 渲染 8 个 Figure
python -c "from apcs.figures import render_all; render_all({...}, 'reports/figures')"

# 跑全部测试
python -m pytest tests/
```

## 目录结构

```
KVCache/
├── design.md                         # 实验方案原始论文版本
├── README.md                         # 本文件
├── configs/
│   ├── pair_qwen3.yaml               # Qwen3-4B → 1.7B 主实验
│   └── pair_smollm2.yaml             # SmolLM2-1.7B → 135M 第二 Pair
├── apcs/
│   ├── cli.py                        # CLI 入口（§63 stdout.log + §73 Task Report）
│   ├── compat/        T00            # §28 Compatibility Scanner
│   ├── replay/        T01            # §29 Self-KV Replay (Gate 0)
│   ├── rope/          T02            # §30 RoPE + de-RoPE 实现
│   ├── alignment/     T03            # §21 Layer Alignment 4 策略
│   ├── mapper/        T04-T06        # §20 Ridge / LowRank / SharedBasis + §12 G2 Mismatched
│   ├── inference/                   # KV 注入推理后端（T05 Replacement 骨架）
│   ├── advantage/     T08            # §22 Advantage State
│   ├── capability/    T07/T09        # §16 Teacher Gap + §37 Main Capability
│   ├── system/        T10            # §4/§38 Scenario A/B/C
│   ├── geometry/      T12            # §40 Geometry Diagnostics
│   ├── decision/      T11            # §69 MVP Decision
│   ├── ablation/                     # §47 A1/A3/A6/A8/A9/A10
│   ├── multiturn/                    # §46 Multi-turn Stability
│   ├── compliance/                   # §52 8 条禁止检查器
│   ├── prereg/                       # §70 PREREGISTRATION.md
│   ├── orchestrator/                 # §71-§73 DAG + Task Report
│   ├── figures/                      # §56-§62 Figure 渲染
│   ├── data/                         # §14-§18 数据集
│   ├── metrics/                      # §3/§4/§45/§48/§51 全部指标
│   ├── io/                           # 配置 + Run 输出 + metadata
│   └── utils/                        # seed/计时/percentile
├── reports/runs/                     # 每个 Run 的产物（§63）
└── tests/                            # 100 个单元 + 集成测试
```

## 任务执行顺序（§71 Agent 强制）

```
T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08 → T09 → {T10, T12, T11} → T13
                                            ↓
                                  PREREGISTRATION.md (自动)
```

- **§72 Gate FAIL 不得跳过**（orchestrator 校验）
- **一次只跑一个任务**
- **每个任务产出标准产物**：`config.json / metadata.json / metrics.json / system.json / geometry.json / summary.md / stdout.log / task_report.md`（§63 + §73）

## 测试

```bash
python -m pytest tests/   # 100 个测试，全部通过
```

测试覆盖：
- 指标模块（retention / CHG / TGRR / PCR / PSR_A / JCR / KL / CKA / 等）
- RoPE 圆环精度（max_err < 1e-10，cosine > 0.999999）
- Mapper 数学正确性（einsum 加速 vs 朴素循环，atol=1e-5）
- Batch ridge 数学等价性（atol=1e-9）
- K/V 独立参数化（separate_kv：K/V 各自 fit/transform，键空间隔离）
- §52 八条禁止检查器
- §64/§65 metadata schema
- CLI 端到端 + T11 verdict + Orchestrator 传递依赖阻断

## 依赖

- Python 3.14+
- numpy, scipy, pyyaml
- pytorch, transformers, datasets（**仅真实 GPU 实验需要**；CI/CPU 环境可跑工程骨架）

## 引用

论文原文 `design.md`（2599 行）描述了完整的实验方案、gate、metric、figure 与论文路径。

## 许可

研究代码，按论文作者的论文指南使用。