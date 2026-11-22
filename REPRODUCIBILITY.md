# 可复现性与数据可获得性

本文件回答三个问题：**材料在哪、怎么重算、哪些必须要 GPU**。
论文（`paper/arxiv/main.tex`，附录 B）中的 Availability 段指向本文件。

## 1. 材料在哪

| 材料 | 位置 | 说明 |
|---|---|---|
| 逐样本评分记录 | `reports/runs/<run-id>/inject-eval/metrics.json` + `inject_eval/*_artifact.json` | 每个 run 一个目录，含 `git_hash`、`protocol_version`、`zero_prefill_verified` |
| run 清单与计数口径 | `scripts/aggregate_claims.py` | 目录名前缀 `v12-/v13-/v14-/v15-` 视为审计协议 run |
| 区间重算 | `scripts/bootstrap_ci_report.py` | 10⁴ 次配对 percentile bootstrap |
| 论文数字总校验 | `scripts/paper_consistency_check.py` | 48 项检查，见第 2 节 |
| 提交前总校验 | `scripts/verify_submission.py` | 页数、引用、打包、分发件、数字门，见第 6 节 |
| 重建 R² 诊断 | `scripts/measure_reconstruction_r2.py` → `reports/reconstruction_r2/*.json` | 见第 2b 节 |
| 图表数据 | `paper/iclr2026/figures/data/all_results.json` | 供 `figures/gen_*.py` 重绘 |
| 协议与实验日志 | `docs/protocol/PROTOCOL.md`、`docs/protocol/EXPERIMENT_LOG.md` | 版本演进 v1.1→v1.5 |
| 逐条 claim→证据映射 | `docs/paper/CLAIM_MAP.md` | 写作期的对照表 |

当前库存（`python scripts/aggregate_claims.py` 实测）：

```
recorded evaluation runs (metrics file present): 177
audit-protocol configuration rows: 181 (translations 133, probes 48)
max translation point estimate: +0.0101 [-0.0806, +0.1018] n=63
translations clearing the epsilon=0.02 margin: 4 (3 distinct configurations)
max probe point estimate: +0.0196 [-0.0113, +0.0529]
probe rows whose interval excludes zero on the positive side: 0
```

## 2. 不用 GPU 就能重算的部分

论文里的**任务指标**全部可从上述 JSON 复算，不加载任何模型：

```bash
python scripts/aggregate_claims.py         # 计数与极值口径
python scripts/paper_consistency_check.py  # 48 项：Table 1 全行、probe 表、家族 CI 表、
                                           # 附录区间、派生算术、摘要数字、陈旧端点、交叉引用
python scripts/verify_submission.py        # 上面两项 + 页数/引用/打包/分发件，提交前跑这个
```

`paper_consistency_check.py` 会重新跑 10⁴ 次 bootstrap，纯 CPU，耗时以分钟计；退出码非零
即表示某个印在纸上的数字与记录不符。改稿后应把它当作提交前的硬门。

这一条只覆盖**任务指标**（gold-letter probability、准确率、PPL 及其区间）。重建指标不在
其中，原因见下一节。

## 2b. 重建 R² 与任务指标不是同一个 checkpoint

论文表 4 的 $R^2$ 来自 `scripts/measure_reconstruction_r2.py`：它在固定的
20 个校准上下文 / 10 个留出上下文上**重新拟合**每个映射器家族，工件写入
`reports/reconstruction_r2/*.json`（`reconstruction_r2.json` 为逐头仿射，
`joint_mlp_reconstruction.json` 为联合 MLP，另有 6 个 Heo 式 top-k 变体）。

因此表 4 的 $R^2$ 与表 1 中 CHG 那一列来自**同一架构的两次不同拟合**，而不是同一个
训练好的 checkpoint。论文的配对是"家族对家族"而非"同一实例对自己的任务得分"，表 4
的注脚已明确写出这一点。要把两者压到同一实例上，需要在同一个 checkpoint 上同时导出
KV 预测与逐样本作答分数——属于需要 GPU 的改动，本轮未做，也不影响"重建更好、任务
不变"这一家族层面的结论。

## 3. 需要 GPU 的部分

| 任务 | 命令 | 说明 |
|---|---|---|
| 单次审计评测 | `python -m apcs.cli inject-eval --config configs/v15_4b_1.7b_affine_c30_tail.yaml --new-run` | 需要被审计的两个模型 |
| 整轮实验 | `bash scripts/run_real_gpu.sh` | 写入 `reports/runs/` |
| 重跑对照 | `python scripts/reproduce_paper.py [--only SUBSTR]` | 22 个核心配置的端到端复现，硬编码 `/workspace` 路径 |
| 重建 R² 诊断 | `python scripts/measure_reconstruction_r2.py --config <run>/inject-eval/config.json --mapper affine|joint_mlp` | 需要 KV 前向 |
| 重新拟合翻译器 | 同上，mapper 训练在 run 内完成 | 唯一无法只靠已存产物复现的步骤 |

## 4. 已知的不可复现项（已在论文 §8 披露）

- **梯度训练家族**：逐头 MLP 在**同一种子**下的两次运行 CHG 相差 0.046（−0.303 vs −0.257）。
  run 设置了 Python/NumPy/PyTorch 种子（含 CUDA），但未开启确定性 kernel，残余非确定性未隔离。
  因此该家族在论文中以**区间**而非点估计报告；联合 MLP 的种内复现差异为 0.001。
- **4K / 8K 长上下文行**：`n=10`、float16 缓存、关闭固定尾部协议，与 1K 行属不同评测
  regime，论文 §6.4 已按此限定。
- **确定性家族**：23 个核心配置的重跑审计在 CHG 上复现到 |Δ| ≤ 0.002。

## 5. 版本标识

每个 run 目录内的 `metrics.json` 都带有生成时的 `git_hash`，因此论文数字可以追溯到代码
状态，而不只是"某个版本的仓库"。论文自身引用的是 `docs/audits/` 下逐轮审查后的定稿状态。

## 6. 提交前的打包与匿名化

```bash
python scripts/verify_submission.py
```

一条命令检查：三份稿的引用/交叉引用闭合、ICLR 正文是否仍在 9 页内（读 `main.aux`）、
`arxiv_submission.zip` 是否与源目录逐字节一致且保留 `figures/`、根目录两份分发 PDF 是否
与源 PDF 同哈希、以及 48 项数字门。逐项清单见 `docs/SUBMISSION_CHECKLIST.md`。

双盲投稿（`paper/iclr2026/`）不公开作者与仓库地址；向审稿人提供材料时使用去标识化归档：
移除 `README.md` 中的姓名与邮箱、`git` 历史的作者信息、以及 `paper/` 下的署名 PDF，
保留 `reports/runs/`、`scripts/`、`docs/protocol/`。`paper/iclr2026/main.tex` 的作者块
默认已注释，`\iclrfinalcopy` 关闭。

## 7. 本文件未覆盖的内容

- `paper/archive/` 下的历史版本不复现，也不声称与当前数字一致。
- `paper/humanized/` 是正文改写副本，自 audit-10 起未再同步，若要启用需从
  `paper/arxiv/main.tex` 重新改写，而不是手工拼接。
- 部署成本（PSR）不在本轮声明范围内：论文 §8 明确不报告部署开销，记录中的 PSR 包含
  一次仅用于测量的前向，因此偏保守。
