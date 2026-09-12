# APCS: Cross-Model KV Cache Runtime Capability Transfer

**Status**: ✅ 闭环完成（协议 v1.5 + 论文定稿）
**Paper**: `paper/cache_audit/`（`main.pdf`，11 页，arXiv 打包于 `paper/cache_audit/arxiv/`）

---

## Overview

核心目标：大尺寸模型在**用户特定数据集**上推理，其 KV Cache **持久化**后经
**mapper** 迁移到小尺寸模型，使小模型**冷启动**（零 re-prefill）即在用户数据
上具备大模型的能力。

**结论：该能力迁移在冻结学生 + 零 re-prefill 下不成立。** 论文以"审计 + 负结果 +
方法论"定位，用恒等对照（H1）与 oracle 探针（H3）把结论钉死。

---

## 最终裁决（三假说框架，30 次真实 GPU 运行）

| 假说 | 判定 | 关键证据 |
|---|---|---|
| **H1 机制无损** | ✅ 通过 | `self_kv` 恒等注入 ≡ 学生自 prefill，逐样本 logit 余弦 1.000 |
| **H2 替换级** | ⚠️ 弱学生平局；强学生全族失败 | 6 族 mapper，CHG −0.14 ~ −0.47（3 seed）；校准预算惰性（c30/c200 差 0.005）；非线性不帮助（MLP −0.303） |
| **H3 能力迁移** | ❌ 原理性否定 | oracle 探针单调劣化（对翻译器质量鲁棒）+ PPL/acc 解耦（PPL 23.7 但 acc 0.267） |

**核心机理**：教师优势在其**参数**（FFN 电路），不在缓存。V 状态经 W_V·h 投影，
h 在 W_V 行空间外的分量对教师缓存不可见 → V_S 不是 V_T 的函数，任何翻译器都只能
追条件均值；而条件均值本身不携带学生可读的答案相关增益。

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
apcs/
├── apcs/
│   ├── mapper/              # math(ridge/affine/lowrank/CCA) + rat + mlp + joint + task_aware
│   ├── inference/           # evaluator（协议 v1.5）+ kv_store（持久化/在线）+ backends
│   ├── metrics/             # CHG/TGRR/CI/permutation_p/...
│   └── providers/           # hf KV/score/timing
├── tests/                   # 133 单测（test_protocol_v11.py 覆盖 v1.1→v1.5）
├── configs/v1{1,2,3,4,5}_*.yaml
├── paper/cache_audit/       # main.tex + figures + references.bib + arxiv/ 打包 + review_report.md
├── reports/runs/            # 92 个运行产物（metrics.json，git hash 落盘）；其中 v1.2+ 审计协议 run 58 个
└── PROTOCOL.md / EXPERIMENT_LOG.md / CRITICAL_REVIEW_RESPONSE.md / DATA_PROVENANCE.md
```

## Quick Start

```bash
pip install -e .
pytest tests/ -v                      # 单元测试（133 passed, 1 skip）
# 单次审计评测（协议 v1.5）：
python -m apcs.cli inject-eval --config configs/v15_4b_1.7b_affine_c30_tail.yaml --new-run
# 编译论文：
cd paper/cache_audit && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Documentation

- `paper/cache_audit/review_report.md`：两轮独立审稿记录（含修复闭环）
- `PROTOCOL.md`：实验协议（§11 v1.1 / §12 v1.2-1.3 / §13 RAT 与探针）
- `EXPERIMENT_LOG.md`：历轮实验终审表
- `CRITICAL_REVIEW_RESPONSE.md` / `DATA_PROVENANCE.md`：历史数据问题与处置

---

## License

See `LICENSE` file.
