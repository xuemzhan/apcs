# APCS / Cache Translation Across Model Scale

《Cache Translation Across Model Scale: An Audit with Identity Controls and
Teacher-Content Probes》的代码、逐样本数据与论文。**产出是一套审计方法，不是一个更快的翻译器。**

## 当前状态（2026-11-22，audit-12 之后）

| 项 | 现状 |
|---|---|
| 论文定位 | 与具体映射算法无关的三层审计：注入机制（H1）、接收端替换（H2）、教师内容增益（H3） |
| 评审 | audit-8 **7/10** → audit-9 **7/10** → audit-10 **6.5** → audit-11 **7/10** → audit-12 **7/10 Accept** |
| 数值门 | `scripts/paper_consistency_check.py` **48/48**（10⁴ 次 bootstrap） |
| 提交门 | `scripts/verify_submission.py` **7/7**（页数、引用、打包、分发件、数值门） |
| 在库稿 | 三份：`paper/arxiv`（15 页，署名）、`paper/iclr2026`（正文 9 页，双盲）、`paper/zh`（15 页） |

**投稿窗口**：稿件的格式合规目标是「正文 9 页 + 双盲」，当前满足；但 ICLR 2027 的
正式投稿截止（2026-09-25）与讨论期修改窗口（2026-11-18）均已过，投递目标需另定。

## 论文现在主张什么

原始工程目标是：大模型在用户数据上推理，把 KV Cache 持久化后经 mapper 搬到小模型，
让小模型零 re-prefill 冷启动即具备大模型能力。

**当前论文不主张"这件事不可能"。** 它主张：

1. 公开结果把三个可以各自失败的问题混在了一起——注入管线对不对（H1）、mapper 能不能
   在替换级取代学生自己的 prefill（H2）、教师缓存里有没有冻结学生可用的内容（H3）；
2. 每个问题配一个仪器与一道门，测量结果可以独立打分，且不依赖 mapper 怎么训练出来的；
3. 在我们测过的**重建导向 + 架构先验 + 严格零 re-prefill**这一族里，H2 与 H3 都没有通过。

审计本身说，结论的边界也在说：行为/蒸馏导向的翻译器（CacheLink、KV-Lingo）、稀疏迁移与
选择性重算（ACCT、RaReCache）、混合循环状态搬运（LatentPort）都不在受测范围内，论文
Appendix E 逐个点名并说明差别。

## 三个假说的当前裁决

| 假说 | 门 | 结果 | 关键证据 |
|---|---|---|---|
| **H1 机制** | 逐样本 logit cos ≥ 0.9998 | ✅ 通过 | `self_kv` 恒等注入 ≡ 学生自 prefill；实测最小 0.99986、均值 0.99998 |
| **H2 替换级** | CHG 区间下界 > −ε（ε=0.02） | ❌ 强学生全族未过；弱学生既不建立退化也不建立非劣 | 四选一设置下点估计 −0.138 ~ −0.392，长上下文（4K）到 −0.596；7 族 mapper、校准预算 10→500、37M 参数联合 MLP 都不改变结论 |
| **H3 教师内容增益** | 某探针区间正侧排除零 | ❌ 无一行通过 | fraction 探针随教师占比单调劣化；顶部三分位层中性；八分层扫描最好点估计 +0.020 [−0.012,+0.052] |

两条解耦作为**支持性证据**报告，不作为本文首次发现：更好的重建不改善答案
（$R^2_V$ 0.323→0.481 仍同 CHG 带）、接近原生的困惑度不改善答案（PPL 23.7 而 acc 0.267）。

关于重建 $R^2$ 的口径：表 4 的 $R^2$ 是**独立诊断重拟合**（固定 20 校准 / 10 留出上下文），
不是产生 CHG 那一列的那个 checkpoint；论文表注与 `REPRODUCIBILITY.md` §2b 都写明了。

## 数字从哪来

| 材料 | 位置 |
|---|---|
| 逐样本评分 | `reports/runs/<run-id>/inject-eval/metrics.json` + `inject_eval/*_artifact.json`（每 run 带 `git_hash`、`protocol_version`、`zero_prefill_verified`） |
| 计数口径 | `scripts/aggregate_claims.py` → **177** 个评测 run、**181** 个协议行（133 translations + 48 probes） |
| 任务指标与区间 | `scripts/bootstrap_ci_report.py`（10⁴ 次配对 bootstrap）、`scripts/paper_consistency_check.py`（48 项） |
| 重建 $R^2$ | `scripts/measure_reconstruction_r2.py` → `reports/reconstruction_r2/*.json`（需 KV 前向） |
| 图表数据 | `paper/iclr2026/figures/data/all_results.json` |
| 协议与实验日志 | `docs/protocol/PROTOCOL.md`、`docs/protocol/EXPERIMENT_LOG.md` |
| 总说明 | `REPRODUCIBILITY.md`：材料位置、不用 GPU 的重算命令、必须 GPU 的步骤、已知不可复现项 |

## 在库版本与分发件

| 版本 | 位置 | 形态 |
|---|---|---|
| **arXiv 完整稿** | `paper/arxiv/` | 15 页（正文至第 10 页 + 参考文献 + 附录），署名；图 4、表 6；含 `arxiv_submission.zip`/`.tar.gz` |
| **ICLR 样式双盲稿** | `paper/iclr2026/` | 正文 **9 页**（结论在第 9 页，参考文献自第 10 页，附录自第 11 页），作者块默认注释、`\iclrfinalcopy` 关闭 |
| **中文对照稿** | `paper/zh/` | 15 页，逐段对照；图 1 为中文重绘，图 2–4 沿用英文图像 |

根目录两份可直接分发的 PDF 与上面两份 `main.pdf` **同哈希**（`verify_submission.py` 会查）：

- `Cache Translation Across Model Scale_An Audit with Identity Controls and Teacher-Content Probes.pdf`
- `跨模型规模的缓存翻译：一项带恒等对照与教师内容探针的审计.pdf`

作者署名与联系方式（三份稿一致）：
Xuemin Zhang（`zxm0813@gmail.com`）、Liangbin Hu（`huliangbing2000@163.com`）、
Kun Yi（`yikun@bit.edu.cn`）、Liheng Zhong（`lihengzhong@berkeley.edu`）、
Junpeng Yu（`yujunpeng@cetc.com.cn`）；`\affil` 统一写 CETC，与后两位的学校邮箱不一致，
待作者确认（清单 A5）。

`paper/humanized/` 是去 AI 味改写副本，**自 audit-10 起未再同步**（缺附录 D/E 与 Availability），
不要直接覆盖回 `paper/arxiv`；若要启用需从 `paper/arxiv/main.tex` 重新改写。

## Quick Start

```bash
pip install -e .

# 提交前检查（纯 CPU，约 3 分钟；退出码非零即不可投）：
python scripts/verify_submission.py

# 数值门单独跑（10⁴ 次 bootstrap）：
python scripts/paper_consistency_check.py

# 单次审计评测（需要模型）：
python -m apcs.cli inject-eval --config configs/v15_4b_1.7b_affine_c30_tail.yaml --new-run

# 编译论文（三份都跑到无 "Warning: Reference" 为止）：
cd paper/arxiv    && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper/iclr2026 && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper/zh       && xelatex  main && bibtex main && xelatex  main && xelatex  main
```

`verify_submission.py` 查五件事：三份稿的引用与交叉引用闭合；ICLR 正文是否仍在 9 页内
（读 `main.aux`，不看 PDF 文本层）；`arxiv_submission.zip` 是否与源目录逐字节一致且保留
`figures/` 层级；根目录分发 PDF 是否与源 PDF 同哈希；48 项数值门是否通过。

## 已知问题

- **单测有 1 项失败**：`tests/test_small_tasks.py::TestT00Provenance::test_rope_theta_in_output`
  ——它会向 modelscope 拉取 Qwen3-4B 的 config 做 provenance 断言，在本机失败；其余测试通过。
  这项不属于数值门，也不影响论文数字。
- **`docs/paper/CLAIM_MAP.md` 未登记** audit-10 之后新增的 6 条外部引用与两个附录小节。
- **根目录有编译器遗留的 `texput.log`**，可删。

## ⚠️ 历史数据声明

2026-08-28 的历史数字（"76.8% / +0.23 / p=0.0004" 等）来自互相矛盾且未经验证的结果文件，
含伪造的 multi-seed（std=0）。这些数字**不得引用**。本仓库自 v1.1 起的所有数字均来自
`reports/runs/*/inject-eval/metrics.json` 的逐样本记录。

## 协议演进（v1.1 → v1.5）

- **v1.1**：gold/accuracy 主口径 + CI + permutation p；互斥切分；KV 持久化与在线冷启动；
  cache 污染修复（transformers 4.52 在 `use_cache=False` 下仍改 cache）；einsum→matmul。
- **v1.2**：`self_kv` 恒等对照；统一 prompt；KV 范数诊断（K 1.58× / V 7.44×）；
  Affine/中心化 mapper；task-aware 混合目标；teacher-summary 基线。
- **v1.3**：规模阶梯（受控评估集）；`AffineLayerMapper`；λ 惰性。
- **v1.4**：RAT（残差锚定翻译器）；Oracle 探针（`mix_a*` / `win_*`）。
- **v1.5**：MLP mapper；多 seed（42/43/44）；受控规模阶梯（`eval_from_tail_n`）；
  H2 门拆成 replacement 与 direction 两个判断。

## 仓库结构

```
KVCache/
├── apcs/                 # Python 包：mapper / inference / metrics / providers
├── configs/              # v1{1..5}_*.yaml 运行配置 + paper_audit_config.yaml
├── data/                 # 小体量数据集与清单
├── docs/                 # design / protocol / audits(audit1..audit12 + 合规记录) / plans
│                         #   + SUBMISSION_CHECKLIST.md（投稿前逐项清单）
├── paper/                # arxiv（署名完整稿）/ iclr2026（双盲，正文 9 页）/ zh（中文对照）
│                         #   humanized（未同步的改写副本）/ archive（历史版本，不入库）
├── reports/runs/         # 177 个记录在案的评测 run（逐样本 metrics.json）
│   └── reconstruction_r2/  # 表 4 的独立诊断重拟合产物
├── review_results/       # 审稿流程输出目录（scripts/run_paper_audit.sh）
├── scripts/              # 见下表
├── tests/                # 单测（协议 v1.1→v1.5；1 项 provenance 用例本机失败，见上）
├── tmp/                  # 本地临时目录（不入库）
├── README.md             # 本文件
├── REPRODUCIBILITY.md    # 材料位置 / 重算命令 / GPU 步骤 / 不可复现项
├── LICENSE / pyproject.toml / requirements.txt
└── *.pdf                 # 两份可直接分发的成品（与 paper/*/main.pdf 同哈希）
```

`scripts/` 里与投稿直接相关的：

| 脚本 | 用途 | 需要 GPU |
|---|---|---|
| `verify_submission.py` | 提交前总校验（7 项） | 否 |
| `paper_consistency_check.py` | 48 项数值门 | 否 |
| `aggregate_claims.py` | 计数口径与极值 | 否 |
| `bootstrap_ci_report.py` | 10⁴ 次配对 bootstrap | 否 |
| `measure_reconstruction_r2.py` | 表 4 的重建诊断 | 是 |
| `reproduce_paper.py` | 22 个核心配置的端到端重跑 | 是 |

其余为运行与流程脚本：`run_real_gpu.{sh,ps1}`、`run_all_tasks.sh`、`run_optimization.sh`、
`run_dag.ps1`、`run_dag_bg.ps1`、`run_paper_audit.sh`、`generate_review_reports.py`、
`paper_audit_workflow.py`、`build_figure_data.py`。

## Documentation

- `docs/SUBMISSION_CHECKLIST.md`：投稿前逐项清单（稿件 / 分发件 / 数字归档 / 人工确认）
- `docs/README.md`：文档索引、阅读顺序、审稿轮次与状态表
- `docs/audits/`：`audit1.md` … `audit12.md`，以及 `AUDIT6/7/8/10/11/12_REWRITE_COMPLIANCE.md`
- `docs/protocol/PROTOCOL.md`、`docs/protocol/EXPERIMENT_LOG.md`：协议与历轮实验终审表
- `docs/design/design.md`：设计规格（代码注释里的 "design.md §NN" 指此文件）
- `docs/paper/CLAIM_MAP.md`：写作期的 claim→证据→置信级别对照表
- `paper/arxiv/README_SUBMIT.md`：arXiv 元数据表单值与纯文本摘要
- `REPRODUCIBILITY.md`：见上

## License

见 `LICENSE`。
