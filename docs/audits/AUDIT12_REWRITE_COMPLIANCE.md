# AUDIT-12 修订完成度记录

日期：2026-11-22
依据：`docs/audits/audit12.md`（7/10 Accept，信心 4/5）
范围：论文三份稿 + 项目代码/工具/文档整理。**不做 GPU 实验**；P1-1 以"核实真伪 + 如实披露"
关闭，而不是重算。

## 一、最重要的一项：R² 与 CHG 是否同一 checkpoint（P1-1）

**核实结论：不是。** 证据：

- `scripts/measure_reconstruction_r2.py` 每次运行都 `_make_mapper()` 重新构造并 `_fit()`
  一次，切分由 `_real_kv_splits(..., n_calib=20, n_eval=10)` 固定；
- `reports/reconstruction_r2/reconstruction_r2.json` 记录 `mapper=affine, n_calib=20,
  n_eval=10`，R²_K=0.8103、R²_V=0.3230（论文表 4 的 0.810 / 0.323）；
- `reports/reconstruction_r2/joint_mlp_reconstruction.json` 记录 `mapper=joint_mlp`，
  同样的 20/10 切分，R²_K=0.8539、R²_V=0.4815，训练集 0.9485 / 0.6472（论文表 4 括号内值）；
- 两个 JSON 的 `config` 指向对应下游 run 的 `config.json`，即"同架构、同超参"，但**不是**
  该 run 训练出来的那个 mapper 实例。

因此审计的判断正确：表 4 的 R² 与表 1 的 CHG 来自两次不同拟合。审计给出的两条路，我选第一条
（把流程交代清楚），因为第二条需要在同一 checkpoint 上同时导出 KV 预测与逐样本作答分数，
属 GPU 改动。

改动：

| 位置 | 改法 |
|---|---|
| 表 4 注脚 | 明确写 "Each $R^2$ here is a *separate diagnostic refit* … on a fixed split of 20 calibration and 10 held-out contexts (`scripts/measure_reconstruction_r2.py`); it is not the instance that produced the CHG column"，并把结论句改为 "the pairing is between families rather than between one checkpoint and its own task score" |
| §6.1 开头 | 补一句"The diagnostic refits each family on its own split, so these $R^2$ values and the CHG column … come from different fits of the same architecture; we compare families, not one checkpoint against itself." |
| §6.1 对比句 | "The joint MLP reaches …" → "Under the diagnostic fit a joint MLP reaches … and the downstream runs of that same family at $c{=}30$ produce a CHG between …" |

`REPRODUCIBILITY.md` 新增 §2b 同样记载了这条区分，供审稿人核对。

## 二、其余 P1

| 编号 | 意见 | 处理 |
|---|---|---|
| P1-2 | "largest point estimate anywhere in the audit is +0.010" 与 octant 探针的 +0.020 冲突 | 改为 "among translated-cache configurations is $+0.010$ … the exploratory octant probes of Appendix~\ref{app:extra} contain a larger positive point estimate, but none establishes a significant gain"；三份稿同步 |
| P1-3 | 复现说明内部矛盾（"Tables 5–4" 表号顺序、表 4 无法由答题分数重算） | Availability 段重写为两类：任务指标与区间由 per-sample 分数 + `paper_consistency_check.py` 重算；重建 R² 需 KV 状态/拟合统计 + `measure_reconstruction_r2.py`。不再用交叉引用区间写法，逐个点名表 |
| P2 ×5 | §3.1 0.25 表述、§3.2 非劣条件、§3.2 H3 "does not train"、表 1 教师行标签、§8 "Four families" | 依次改为 "uniform four-choice predictions score 0.25"、"…non-inferiority **if the true effect stays below $-\epsilon$**"、"The probe adds no translator training"、"Teacher Qwen3-4B full prefill"、"Several alternative families" |

## 三、项目级整理（投稿准备）

| 新增/改写 | 作用 |
|---|---|
| `scripts/verify_submission.py`（新） | 纯 CPU 的提交前校验器，7 项：三份稿的引用/交叉引用闭合、ICLR 正文 ≤9 页（读 `main.aux`）、zip 条目与 `figures/` 层级、zip 与源目录逐字节一致、根目录分发 PDF 同哈希、48 项数字门 |
| `docs/SUBMISSION_CHECKLIST.md`（新） | A 稿件 / B 分发件 / C 数字与归档 / D 人工确认 / E 改动后固定动作，每项带现状与 ✅⚠️ |
| `REPRODUCIBILITY.md`（重写） | 材料位置表增加重建 R² 与校验器；新增 §2b checkpoint 区分；GPU 任务表补 `measure_reconstruction_r2.py` 与 `reproduce_paper.py` |
| `README.md` | Quick Start 改为"先跑 `verify_submission.py`"，三份稿的编译命令补齐 `zh`；Documentation 索引更新至 audit-12 |
| `docs/README.md`（重写） | 目录索引、阅读顺序、**审稿轮次与状态表**（audit-8→12）、未完成事项 |

## 四、验证（本机实测）

```
python scripts/verify_submission.py
  PASS  arxiv:    15 pages,              31 citations, 14 refs all resolve
  PASS  iclr2026: 15 pages, body ends p9/9, 31 citations, 14 refs all resolve
  PASS  zh:       15 pages,              31 citations, 14 refs all resolve
  PASS  zip has 8 entries, figures/ layout intact, main.tex/main.bbl/references.bib match source
  PASS  both distributed PDFs match their sources by hash
  PASS  paper_consistency_check.py: PASS 48 checks, FAIL 0 checks
  → PASS 7 checks, FAIL 0 checks
```

- 三份稿重编，0 未定义引用；ICLR 正文仍在第 9 页（附录 A 自第 11 页起，总 15 页不计限）。
- `arxiv_submission.zip` 110,143 B / `.tar.gz` 98,006 B 重打；校验器已确认与源目录一致。
- 表 1 教师行改名后，`paper_consistency_check.py` 仍 48/48（该行不参与 CHG 校验）。

## 五、未做

1. **GPU 项**：在同一 checkpoint 上同时导出 R² 与 CHG（P1-1 的"第二条路"）、V-permutation
   正对照、把 CacheLink/KV-Lingo 的功能导向 mapper 放进同一审计。
2. **中文版**：缺 Availability 段；§8 的 "Several alternative families" 一处措辞未同步。
   （中文版 §3.1/§3.2/§5.2/§6.1/表 1/表 4 已全部同步。）
3. **`docs/paper/CLAIM_MAP.md`**：未登记 audit-10 之后新增的 6 条外部引用与两个附录小节。
4. **单位署名**：Kun Yi（bit.edu.cn）、Liheng Zhong（berkeley.edu）与统一的 `CETC` 单位
   仍不一致，需作者决定是否拆 `\affil`。
5. **投稿窗口**：ICLR 2027 的截止（2026-09-25）与讨论期（2026-11-18）均已过，本轮只保证
   格式合规（正文 9 页 + 双盲），投递目标另定。
