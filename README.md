# APCS: Cross-Model KV Cache Runtime Capability Transfer

**Status**: 🔬 协议 v1.1 重构后验证中
**Paper**: `paper/arxiv/main.tex`（cautionary 版本）

---

## Overview

核心目标：大尺寸模型在**用户特定数据集**上推理，其 KV Cache **持久化**后经
**mapper** 迁移到小尺寸模型，使小模型**冷启动**（零 re-prefill）即在用户数据
上具备大模型的能力。

---

## ⚠️ 结果可信度声明（2026-08-29）

下表中的历史数字（76.8% / +0.23 / p=0.0004 等）来自 8/28 的两份**互相矛盾且
UNVERIFIED** 的结果文件（见 `CRITICAL_REVIEW_RESPONSE.md` / `DATA_PROVENANCE.md`），
**不得作为科学结论引用**。

| Method | CHG | TGRR | p-value | 状态 |
|--------|-----|------|---------|------|
| Native average | +0.23 | 0.768 | 0.0004 | ⚠️ UNVERIFIED（历史，勿引用） |
| V only / Ridge KV / ... | — | — | — | ⚠️ UNVERIFIED（历史，勿引用） |

### 当前真实证据（8/29 sweep，真实推理，但受协议限制）

- 0.6B 学生：CHG（**top-1 置信度口径**）+0.15 ~ +0.26 稳定为正；
- 1.7B 学生：CHG 随 seed 在 -0.09 ~ +0.10 之间变号；
- 局限：置信度≠正确率；校准=评估同样本（记忆效应）；无显著性检验；
  KV 未持久化。**这些结论不能支撑"能力迁移"声明。**

---

## 协议 v1.2/v1.3（当前）

**三假说框架**（详见 `PROTOCOL.md` §12）：

- **H1 机制无损：✅ 已判定通过** —— `self_kv` 恒等注入与学生自 prefill
  逐样本 logit 余弦 = 1.000（zero-prefill 注入通路数学无损）；
- **H2 映射质量：进行中** —— Affine（中心化+截距）把 mapper 损失砍半
  （4B→1.7B: acc 0.20→0.367，CHG −0.296→−0.138）；KV 范数诊断显示
  教师 V 幅值是学生 7.4 倍；正在执行校准规模阶梯 30→200（→500）；
- **H3 可利用性：以 H2 达标为前提** —— task-aware 混合目标在 30 样本下
  过拟合（预期内），规模阶梯完成后判定。

**关键机制结论**（可复用）：
- `mapper.type: affine_layer` —— 逐层中心化岭回归，参数少 8 倍，小数据首选；
- `inject_eval.ablation_modes` 必含 `self_kv`（H1 对照）；
- `native_rescale: true` + `summary_baseline: true`（尺度诊断 + 实用性基线）；
- mapper 变换已用批量 matmul 替代 naive einsum（~67×，PSR −13→−3.4）。

### 判定 Gate（能力结论的唯一依据）

- **Gate B（能力）**：同分布 held-out 上 gold/accuracy 口径 CHG>0 且
  CI 下限>0 且 p<0.05，PPL 不显著劣化 —— 输出在 `metrics.capability_gate`。
- Gate A（工程 zero-prefill 真验证）/ C（计入校准成本后 PSR>1）/
  D（跨数据集/跨域保持增益）依次叠加。

---

## Project Structure

```
apcs/
├── apcs/
│   ├── mapper/              # Ridge/LowRank/Affine/CCA + JointKVMapper + task_aware
│   ├── inference/           # evaluator（协议 v1.1）+ kv_store（持久化）+ backends
│   ├── metrics/             # CHG/TGRR/CI/permutation_p/...
│   └── providers/           # hf KV/score/timing
├── tests/                   # 单元测试（协议 v1.1 见 tests/test_protocol_v11.py）
├── configs/v11_*.yaml       # v1.1 验证配置（切分+native+持久化+RoPE 对照）
├── reports/runs/            # 原始运行产物（JSON，含 per-prompt 分数）
└── CRITICAL_REVIEW_RESPONSE.md / DATA_PROVENANCE.md / PROTOCOL.md
```

---

## Quick Start

```bash
pip install -e .
pytest tests/ -v                      # 单元测试
# 协议 v1.1 离线阶段（Teacher+Student，KV 落盘）：
python -m apcs.cli inject-eval --config configs/v11_4b_1.7b_rotated.yaml --new-run
# 在线阶段（仅 Student，从 kv_store 冷启动）：
#   在 config 的 inject_eval 节加 online_kv_dir: <上一次运行的 kv_store 路径>
```

---

## Documentation

- `PROTOCOL.md`：实验协议（v1.0；v1.1 变更见本 README 与代码注释）
- `CRITICAL_REVIEW_RESPONSE.md` / `DATA_PROVENANCE.md`：历史数据问题与处置
- `EXPERIMENT_LOG.md` / `EXPERIMENT_ANALYSIS.md`：历史实验记录（UNVERIFIED 标注）

---

## License

See `LICENSE` file.
