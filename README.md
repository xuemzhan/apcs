# APCS: Cross-Model KV Cache Runtime Capability Transfer

**Status**: ✅ 实验闭环（协议 v1.5）＋ 论文已按 audit-7 定稿（Accept 7/10）

**论文只有两个在库版本**（数字与图表同源，2026-09-13 起）：

| 版本 | 位置 | 形态 |
|---|---|---|
| **arXiv 完整稿** | `paper/arxiv/` | 13 页（正文 9 页 + 参考文献 + 附录），署名（Xuemin Zhang、Liangbin Hu、Kun Yi、Liheng Zhong、Junpeng Yu），图 4、表 4，含 `arxiv_submission.zip` 提交包 |
| **ICLR 2026 投稿稿** | `paper/iclr2026/` | 正文 **9 页**（参考文献自第 10 页起），双盲匿名，ICLR 2026 样式，证据不足项已下沉到附录 |

其余历史版本（`cache_audit` 全量长文、匿名版、旧 ICLR 全量稿、官方模板、更早的
prior-projection 稿与预览）统一放入 `paper/archive/`，**不入库**（本地保留，git 历史可追）。
图表与结论的对应关系、逐轮审稿意见与修改记录见 `docs/`。

---

## Overview

核心目标：大尺寸模型在**用户特定数据集**上推理，其 KV Cache **持久化**后经
**mapper** 迁移到小尺寸模型，使小模型**冷启动**（零 re-prefill）即在用户数据
上具备大模型的能力。

**结论：该能力迁移在冻结学生 + 零 re-prefill 下不成立。** 论文以"审计 + 负结果 +
方法论"定位，用恒等对照（H1）与 oracle 探针（H3）把结论钉死。

---

## 最终裁决（三假说框架，171 个记录在案的评测 run；计数规则见 `scripts/aggregate_claims.py`）

| 假说 | 判定 | 关键证据 |
|---|---|---|
| **H1 机制无损** | ✅ 通过 | `self_kv` 恒等注入 ≡ 学生自 prefill，逐样本 logit 余弦 1.000 |
| **H2 替换级** | ⚠️ 弱学生近平局但未过 ε=0.02 非劣；强学生全族失败 | 7 族 mapper，CHG −0.14 ~ −0.39；校准预算惰性（c30/c200 差 0.005）；非线性不帮助（per-head MLP 5 次训练区间 −0.39~−0.26、joint MLP 4 次训练 −0.30~−0.26）；5 个跨架构学生中 1 个（Llama-3.2-1B，近随机）过替换 gate 但无增益，其余 4 个未过 |
| **H3 能力迁移** | ❌ 原理性否定 | oracle 探针（fraction/三分位/八分位）单调劣化 + 原生内容对照 + PPL/acc 解耦（PPL 23.7 但 acc 0.267）；长上下文（~1k token）同样失败；朴素 target-side replay 不能恢复 |

**核心机理**（论文口径：与"教师优势由权重中介"一致，非不可能性证明）：V 状态经
W_V·h 投影，h 在 W_V 行空间外的分量对教师缓存不可见 → 单头 student value 不由对应
单头 teacher value 决定（held-out R²：K +0.81 / V +0.32），翻译器只能追条件均值；
而条件均值本身不携带学生可读的答案相关增益。

## ⚠️ 历史数据声明

8/28 的历史数字（"76.8% / +0.23 / p=0.0004" 等）来自互相矛盾且 UNVERIFIED 的
结果文件，含伪造的 multi-seed（std=0）。这些数字**不得引用**。本仓库自 v1.1
起的所有数字均来自 `reports/runs/*/inject-eval/metrics.json` 的逐样本记录，
每 run 携带 git hash + protocol version。

## 协议演进（v1.1 → v1.5）

- **v1.1**：gold/accuracy 主口径 + CI + permutation p；互斥切分；KV 持久化 +
  在线冷启动；cache 污染修复（transformers 4.52 在 use_cache=False 下仍改 cache）；
  einsum→matmul（PSR −13→−3.4）。
- **v1.2**：self_kv 恒等对照；统一 prompt；KV 范数诊断（K 1.58× / V 7.44×）；
  Affine/中心化 mapper；task-aware 混合目标；teacher-summary 基线。
- **v1.3**：规模阶梯（受控评估集）；AffineLayerMapper；λ 惰性。
- **v1.4**：RAT（残差锚定翻译器：embedding 对齐核 + 头匹配 + 闭式修正）；
  Oracle 探针（mix_α / win_*）。
- **v1.5**：MLP mapper（非线性测试）；多 seed（42/43/44）；受控规模阶梯
  （`eval_from_tail_n`）；H2 gate 拆分（replacement vs gain）。

## 关键机制结论（可复用）

- `mapper.type: affine`（per-head 中心化）—— 单一最大收益（CHG −0.296→−0.138）；
- `mapper.type: affine_layer` —— 参数少 8 倍，小数据首选；
- `inject_eval.ablation_modes` 必含 `self_kv`（H1 对照）、`mix_a*`/`win_*`（H3 探针，`probe_mode` 门控）；
- `native_rescale: true` + KV 范数诊断（分离尺度失配 vs 语义失配）；
- `mapper.type: rat` / `mlp` —— 架构锚定 / 非线性，均未超越 affine（负结果，留作审计证据）。

## Project Structure

```
KVCache/
├── apcs/                     # Python 包
│   ├── mapper/               # math(ridge/affine/lowrank/CCA) + rat + mlp + joint + task_aware
│   ├── inference/            # evaluator（协议 v1.5）+ kv_store（持久化/在线）+ backends
│   ├── metrics/              # CHG/TGRR/CI/permutation_p/...
│   └── providers/            # hf KV/score/timing
├── tests/                    # 单测（133 passed, 1 skip；test_protocol_v11.py 覆盖 v1.1→v1.5）
├── configs/                  # v1{1..5}_*.yaml 运行配置 + paper_audit_config.yaml
├── data/                     # 小体量数据集/清单
├── docs/
│   ├── design/               # design.md（设计规格，代码注释里的 "design.md §NN" 指此文件）、BRIEF.md、INNOVATION.md
│   ├── protocol/             # PROTOCOL.md（实验协议）、EXPERIMENT_LOG.md（历轮实验终审表）
│   ├── audits/               # 逐轮审稿意见 audit1..audit6 + AUDIT6_GPU_CORRESPONDENCE.md（实验↔审稿逐条对应）
│   └── plans/                # REVISION_PLAN*.md / SUPPLEMENT_EXPERIMENTS_PLAN.md / GPU_PLAN_AUDIT6.md + 需求与用户故事
├── paper/
│   ├── arxiv/                # arXiv 完整稿：main.tex/.bbl/.pdf + figures/ + references.bib + arxiv_submission.zip
│   ├── iclr2026/             # ICLR 2026 投稿稿（正文 9 页、匿名）+ 样式文件 + figures/
│   └── archive/              # 历史版本（全量长文、匿名版、旧 ICLR 稿、官方模板、prior-projection 稿）——不入库
├── reports/runs/             # 记录在案的评测 run（metrics.json，git hash 落盘）；聚合口径见 scripts/aggregate_claims.py
├── review_results/           # 审稿报告生成物（scripts/run_paper_audit.sh 的输出目录）
└── scripts/                  # 聚合/复现/审稿工具 + 运行脚本（run_all_tasks.sh、run_optimization.sh、run_dag*.ps1、run_real_gpu.*）
```
根目录只保留 `README.md`、`LICENSE`、`pyproject.toml`、`requirements.txt`、`.gitignore`、
包目录 `apcs/`、配置/数据/文档/论文/报告/脚本目录。

## Quick Start

```bash
pip install -e .
pytest tests/ -v                      # 单元测试（133 passed, 1 skip）
# 单次审计评测（协议 v1.5）：
python -m apcs.cli inject-eval --config configs/v15_4b_1.7b_affine_c30_tail.yaml --new-run
# 编译论文（两版都跑到无 "Warning: Reference" 为止）：
cd paper/arxiv    && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper/iclr2026 && pdflatex main && bibtex main && pdflatex main && pdflatex main
# arXiv 提交包（含 main.bbl 与 figures/，可由 zip 直接上传）：
#   paper/arxiv/arxiv_submission.zip
# GPU 机整轮实验（Linux）：
bash scripts/run_real_gpu.sh
```

## Documentation

- `docs/audits/`：逐轮审稿意见（`audit1.md` … `audit7_1.md`）、`AUDIT6_GPU_CORRESPONDENCE.md`
  （第六轮意见↔GPU 实验逐条对应）、`AUDIT6_REWRITE_COMPLIANCE.md` / `AUDIT7_REWRITE_COMPLIANCE.md`
  （重写与两轮意见的逐条完成情况）
- `docs/protocol/PROTOCOL.md`：实验协议（§11 v1.1 / §12 v1.2-1.3 / §13 RAT 与探针）
- `docs/protocol/EXPERIMENT_LOG.md`：历轮实验终审表（含 audit-6 勘误）
- `docs/design/design.md`：设计规格；代码注释中的 "design.md §NN" 均指该文件
- `docs/plans/`：各轮修改方案、补充实验计划、GPU 计划
- `docs/paper/CLAIM_MAP.md`：写作前的 claim→证据→置信级别对照表
- `paper/arxiv/README_SUBMIT.md`：arXiv 元数据表单值与纯文本摘要

---

## License

See `LICENSE` file.
