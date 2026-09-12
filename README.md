# APCS: Cross-Model KV Cache Runtime Capability Transfer

**Status**: ✅ 闭环完成（协议 v1.5 + 论文定稿）
**Paper**: 三份同步稿，内容与数字一致（2026-09-12 起）：
`paper/cache_audit/main.pdf`（正文稿，16 页）、`paper/cache_audit/arxiv/main.pdf`
（arXiv 打包）、`paper/cache_audit_iclr2026/main.pdf`（ICLR 2026 模板）。图 1–4 与
表 1–3 均按最新数据重绘/重排；图表与结论的对应关系见
`paper/cache_audit/EVIDENCE_MAP.md`。
作者：Xuemin Zhang（zmx0813@gmail.com）、Liangbin Hu、Kun Yi、Liheng Zhong、
Junpeng Yu。署名版为 `cache_audit` 与 `arxiv` 两份；`cache_audit_iclr2026`
保持匿名投稿态，真实作者块以注释形式留在 tex 中，camera-ready 时取消注释并
打开 `\iclrfinalcopy`。
匿名审稿版：`paper/cache_audit_anon/`（`main.pdf` + 源码 + 图，已去除作者块、
PDF 元数据作者为 Anonymous、仓库名中性化），并打包为
`paper/cache_audit_anon/anonymous_review_package.zip`。审稿包内**不含**代码仓库
（git 历史含作者信息），如需给代码请另做无 `.git` 的快照。

**投稿版（正文 9 页）**：`paper/cache_audit_iclr2026/main.pdf` 已按 ICLR 正文 9 页
限制压缩——正文严格 9 页（参考文献自第 10 页起），细节整块下沉到附录：相关工作四条主线、
审计方法学细节、RAT 组件消融、通道不对称、校准阶梯/重拟合、九分位（octant）探针、
跨架构细节、Survey 对照表、Table 2 辅助行、部署解读。压缩只做搬迁与合并，未删除任何证据；
完整长文版本仍保留在 `paper/cache_audit/main.pdf`（19 页）与 `arxiv/`。

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
apcs/
├── apcs/
│   ├── mapper/              # math(ridge/affine/lowrank/CCA) + rat + mlp + joint + task_aware
│   ├── inference/           # evaluator（协议 v1.5）+ kv_store（持久化/在线）+ backends
│   ├── metrics/             # CHG/TGRR/CI/permutation_p/...
│   └── providers/           # hf KV/score/timing
├── tests/                   # 133 单测（test_protocol_v11.py 覆盖 v1.1→v1.5）
├── configs/v1{1,2,3,4,5}_*.yaml
├── paper/cache_audit/       # main.tex + figures + references.bib + arxiv/ 打包 + review_report.md
├── reports/runs/            # 171 个记录在案的评测 run（metrics.json，git hash 落盘）；聚合口径见 scripts/aggregate_claims.py
└── PROTOCOL.md / EXPERIMENT_LOG.md / CRITICAL_REVIEW_RESPONSE.md / DATA_PROVENANCE.md
```

## Quick Start

```bash
pip install -e .
pytest tests/ -v                      # 单元测试（133 passed, 1 skip）
# 单次审计评测（协议 v1.5）：
python -m apcs.cli inject-eval --config configs/v15_4b_1.7b_affine_c30_tail.yaml --new-run
# 编译论文：
cd paper/cache_audit && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
# 交叉引用/浮动体需要跑到无 "Warning: Reference" 为止（ICLR 版尤其如此）
```

## Documentation

- `paper/cache_audit/review_report.md`：两轮独立审稿记录（含修复闭环）
- `PROTOCOL.md`：实验协议（§11 v1.1 / §12 v1.2-1.3 / §13 RAT 与探针）
- `EXPERIMENT_LOG.md`：历轮实验终审表
- `CRITICAL_REVIEW_RESPONSE.md` / `DATA_PROVENANCE.md`：历史数据问题与处置

---

## License

See `LICENSE` file.
