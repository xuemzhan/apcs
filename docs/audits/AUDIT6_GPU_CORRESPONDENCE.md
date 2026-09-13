# audit-6：GPU 机新实验 ↔ 最新审稿结论 逐条对应

**实验来源**：GitHub `origin/main` 提交 `b6c37c3`
`exp(audit-6): strict Heo-design test (concat + FineWeb-Edu calibration) + real 4K/8K long-context results`
（已 fast-forward 到本地工作区，原 `080022f` → `b6c37c3`）

**审稿结论来源**：`paper_audit6_1.md`（第六轮，2026-09-13，结论 Weak Accept 6/10；补掉 P0 后 7/10 Accept）

**对应关系一句话**：本轮 GPU 交付把审稿唯一真正卡分的**实验问题（P0-1 Heo baseline）闭合了**，
并额外修正了一个内部有效性 bug；其余 P0/P1/P2 条目都属于**文稿与文献工作**，目前**尚未改动**。

---

## 一、本轮从 GPU 机取回的内容

### 1. G1：Heo 设计下的严格检验（concatenation + FineWeb-Edu 校准，3 连跑）

实现改动：

- `apcs/mapper/concat.py::ConcatRidgeMapper`：选中源层在特征维**拼接**后拟合 ridge
  （$W:\mathbb{R}^{kD_t}\to\mathbb{R}^{D_s}$，正则矩阵尺寸 $kD_t$）。
- 校准语料改为 `calib_corpus=fineweb_edu`、`calib_corpus_tokens=1024`（确定性生成的长文，
  来源差异已在附录注明）。
- 其余协议不变：Qwen3-4B→1.7B、固定 tail-100（与 E1 c200 同一条评估集、逐 id 校验）、
  `self_kv` 恒等对照、`de_rope_k: true`、$\lambda=10^{-3}$、$k\in\{1,3,5\}$。
- 每个 run 新增落盘 `layer_mapping.json`（上一轮缺失的选层表）。

结果（`reports/runs/v15-4b-to-1.7b-heo-concat-fwe-k{1,3,5}-c100-2026-09-13-*`）：

| run | acc | gold | PPL | Gold CHG [95% CI] | self_kv | gate |
|---|---|---|---|---|---|---|
| concat+FWE k=1, c100×1024 | 0.230 | 0.233 | 159446.7 | −0.279 [−0.410, −0.145] | ≈0 | FAIL |
| concat+FWE k=3, c100×1024 | 0.290 | 0.289 | 45921.5 | −0.223 [−0.368, −0.084] | ≈0 | FAIL |
| concat+FWE k=5, c100×1024 | 0.270 | 0.257 | 3677.7 | −0.254 [−0.382, −0.122] | ≈0 | FAIL |

教师 0.770/0.6907，学生 0.520/0.5116；`ridge_self_kv` 与学生逐位一致（CHG≈0）。
三个 run 的 95% CI 上界均 < 0，且 CI 互相重叠。

**与旧 average 实现对照（k 的关系）**：CHG 无实质差别（−0.28/−0.22/−0.25 vs −0.27/−0.31/−0.29），
但重建质量与 PPL 的走向**反转**：concat 下 held-out $R^2$ 随 $k$ **上升**（K 0.826→0.858→0.864，
V 0.326→0.419→0.428，来自 concat mapper 的重建诊断，即论文 §6 引用的 B1/V2 口径），
PPL 随 $k$ **下降**（1.6e5→4.6e4→3.7e3，即本轮三个 G1 run 的实测 PPL），
而 average 下两者随 $k$ 同步恶化。⇒ **组合方式影响的是流畅度/稳定性，不改变"能否产生能力增益"的结论。**

**环境限制（已如实记录）**：容器 cgroup `memory.max≈63GB`，参考口径 500×1024 约需 300GB，无法执行；
本轮用 **100×1024**（校准总 token 数与 c200×512 相同）并改用 float16 存校准 KV。

### 2. 真实 4K / 8K 长上下文（修正一个内部有效性 bug）

发现并修复：`inject-eval` 的 `_load_sample_rows` 未把配置里的 `needle_target_tokens` 传给 loader，
旧 "4K" 配置实际回退到 1,024 token；`needle_mcqa` 的 `@_register` 装饰器在 `de02b86` 中被误删，也已恢复。

| 目标 | 实际长度 | n | acc / gold / PPL | Gold CHG [95% CI] | gate |
|---|---|---|---|---|---|
| 4K | ~3.8K | 10 | 0.200 / 0.200 / 15855.1 | −0.596 [−0.929, −0.170] | FAIL |
| 8K | ~7.6K | 10 | 0.200 / 0.200 / 24247.4 | −0.577 [−0.932, −0.118] | FAIL |

（旧 "4K" 行实为 1K，CHG −0.035、CI 含 0，已在 `EXPERIMENT_LOG.md` 加勘误。）
真实长上下文下负幅**变大**，方向与 1K 一致。

### 3. 同提交内的文稿改动

- 摘要：Heo 段升级为 "in both its averaged and its concatenated FineWeb-Edu-calibrated form"（CHG −0.22 至 −0.31）。
- Table 2：新增 3 行 concat 结果，脚注 $^{\sharp}$ 说明对齐了组合方式与序列长度（序列数 100 vs 参考 500）。
- §2.2：改为 "the negative result therefore persists under the reference's own combination and sequence-length regime"。
- §6：新增"两种源层组合下判定一致"，并写明 concat 使重建/PPL 诊断反转而 CHG 仍显著为负。
- 附录实现差异表：`Source-layer combination` 一列已从 "ours: average" 改为 concatenation（并对齐语料与长度）。
- "What remains unmeasured"：选层表已落盘，相关缺口删除。
- 长上下文段落：删除旧 "4K 下 CI 含 0" 的表述，改为真实 4K/8K 的进一步退化，并主动披露 plumbing bug。

---

## 二、与审稿结论逐条对应

### P0 级

| # | 审稿条目 | 本轮实验/文稿证据 | 状态 |
|---|---|---|---|
| P0-1 | §二：Heo baseline 实为 average + 200×512，函数空间不同，不能称为该方法的检验；要求做 top-$k$ concatenation + FineWeb-Edu 1,024-token 校准，或降级措辞 | **G1 三连跑全部完成**：concatenation mapper、FWE-1024 校准、k=1/3/5，Gold CHG 全部显著为负、gate FAIL、identity 对照通过；§2.2/§6/附录表同步改写 | **已闭合（技术层面）** |
| P0-2 | §三：H3 仍依赖 translator，"a null result cannot be blamed on the mapper" 不成立；建议收窄 H3 定义并把 "cannot be blamed on the mapper" 改为 "is not specific to the end-to-end replacement score of a single mapper" | 本轮为**文稿条目，未改动**。Introduction 现存原文 "so a null result cannot be blamed on the mapper"；H3 定义仍是 "the teacher's cache contains student-readable advantage"，"oracle" 措辞仍遍布标题/摘要/§4.4/H3 陈述 | **未闭合** |
| P0-3 | §四：必须补 2026-09-01 CacheBridge 与 2026-08-31 A Universal Context-Reuse Layer 两篇 | `main.bbl` 渲染仍是 **21 条**参考文献，`references.bib` 23 条，**无** CacheBridge、无 context-reuse；本轮提交未触碰 bib | **未闭合** |

### P1 级

| # | 审稿条目 | 本轮证据 | 状态 |
|---|---|---|---|
| P1-1 | §五：family-wise bound 措辞弱化为 "bootstrap 95% upper estimate"，或改 centered max-bootstrap | 已弱化为 "The bootstrap 95% upper estimate for the maximum gain across the confirmatory probe family is $+0.009$." ✅；**但该句与前一句重复**（"...its 95% upper bound is $+0.009$. The bootstrap 95% upper estimate ... is $+0.009$."），是半途编辑留下的残句；centered max-bootstrap 未做（审稿也说是可选项） | **部分闭合** |
| P1-2 | §七：Figure 1 caption 的 "capability upper bound" 改为 "teacher full prefill reference" | 仍为 "...with the teacher's full prefill as the capability upper bound" | **未闭合** |
| P1-3 | §七：§4.4 "keeps the student's own cache dominant or localized" 改为 "retains a controlled fraction ... or localizes the intervention" | 原文未改 | **未闭合** |
| P1-4 | §七：§5.6 "steer attention to plausible but task-irrelevant positions" 改为 "The observed asymmetry is consistent with key-side routing errors" | 原文未改（仍把 mechanism 写成事实，随后才说 leave the mechanism open） | **未闭合** |
| P1-5 | §七：RAT "rule out the objection that ..." 改为 "tests whether one architecture-informed translator changes the verdict" | 原文未改 | **未闭合** |
| P1-6 | §八：cross-family 改为 "clears the reported ε=.02 replacement margin but gains nothing"；Table 3 列标题 "Gate" → "ε=.02 reporting gate" | 两处原文均未改（"is non-inferior but gains nothing"；表头仍为 `Gate`） | **未闭合** |

### P2 级

| # | 审稿条目 | 本轮证据 | 状态 |
|---|---|---|---|
| P2-1 | §六：§3.3 "makes the null more informative, not less" 改为 "reduces concern that the negative result is specific to one hand-picked configuration" | 已按建议改写，且保留了 "confirmatory probe family quantified separately with joint bootstrap" | **已闭合** |
| P2-2 | §九：把中心结果放在 reconstruction≠task quality / PPL≠task quality | 本轮 GPU 数据**强化**该论断：concat 下重建 $R^2$ 随 $k$ 上升、PPL 随 $k$ 下降，CHG 仍显著为负；§6 已写入"两种组合下判定一致" | **已闭合且被强化** |
| P2-3 | §十：全文 copy-edit（"no tested strong-student mapper..." 句首小写、主语关系不清） | 原样存在："so mechanics is not the failure mode. no tested strong-student mapper establishes replacement:" | **未闭合** |

### 投稿项

| # | 审稿条目 | 现状 | 状态 |
|---|---|---|---|
| S-1 | §十三：ICLR 主文 ≤9 页 | 四个 PDF 均为 **19 页**（`cache_audit`、`cache_audit_anon`、`cache_audit_iclr2026`、`cache_audit/arxiv`）；article 版（`cache_audit/main.pdf`）References 从第 12 页起，即主文仍约 11 页 | **未闭合** |
| S-2 | §十三：双盲匿名 | `paper/cache_audit_iclr2026/main.tex` 已是 `\author{Anonymous}` ✅；但 `paper/cache_audit/main.tex` 与 `paper/cache_audit/arxiv/main.tex` 仍带真实作者与机构 | **部分闭合** |

### 计划外但已交付（GPU 计划的 G2 + 一个内部 bug）

| 条目 | 证据 | 价值 |
|---|---|---|
| 真实 4K/8K 长上下文（G2 原列为"可选"） | 4K (~3.8K) CHG −0.596、8K (~7.6K) CHG −0.577，均 gate FAIL | 把"长上下文未测"补成"越长退化越大" |
| `needle_target_tokens` 透传 bug + `@_register` 丢失 | 旧 "4K" 行实为 1K；已修复并加勘误 | 修掉一条**内部有效性**风险（原 4K 结论与 1K 冲突） |

---

## 三、总览与下一步

**完全闭合 3 项**：P0-1（实验，本轮核心）、P2-1、P2-2，外加计划外的 4K/8K 与 bug 修复。
**部分闭合 2 项**：P1-1（措辞已弱化，但留下重复残句）、S-2（ICLR 版已匿名，article/arXiv 版仍带作者）。
**未闭合 9 项**：P0-2、P0-3、P1-2、P1-3、P1-4、P1-5、P1-6、P2-3、S-1。

其中**没有任何一项需要 GPU**：G1 已是唯一的必做 GPU 项，G2 也已顺手完成。剩下的全是文稿/文献工作。

建议顺序：

1. **P0-3**（补两篇 related work）——审稿明确说"这次不能再不补"，且成本最低。
2. **P0-2**（收窄 H3 + 删 "blamed on the mapper" + 统一 "oracle probes" → "teacher-content probes"）。
3. **P1-2…P1-6 + P1-1 残句 + P2-3**（wording 与 copy-edit，一次改完）。
4. **S-1**（9 页匿名版）——与科学质量无关但直接 desk-reject。

需要你决定的两件事：

- **校准条数**：参考用 500×1,024，我们受 63GB 内存限制只做到 100×1,024。当前稿已如实标注差异。
  如果要在 rebuttal 里彻底堵住"仍不是复现"这一问，需要更大内存的机器重跑 500 条（GPU 计划估约需 300GB），
  否则建议就按现状把这句写成"同一 token 预算下的对齐检验"。
- **"closest published design" 的措辞**：现在 concat 已对齐，该表述比上一版更站得住，
  但严格审稿人仍可能抓住语料来源（真实 FineWeb-Edu vs 确定性生成）与条数差异。要不要再降半档到
  "the design we align most closely"？

---

## 附：本地操作记录

- 本地 `main` 已从 `080022f` 快进到 `b6c37c3`；工作区除既有的未跟踪 `output/` 外干净。
- 快进前，`paper/cache_audit_iclr2026/main_full_backup.tex` 与 `paper/iclr2026_template/*`
  这 10 个未跟踪文件与入库版本比对：9 个**逐字节相同**，1 个（`main_full_backup.tex`，73,695 字节，本地版更长）
  **不同**。已全部移到 `tmp/untracked_backup_20260913/` 保留，再由合并写回入库版本。
  如需要那份本地不同的备份，从该目录取回即可。

---

# 第二轮：按审稿清单执行修改（2026-09-13 晚）

依据上一节的 9 项未闭合条目，逐条落实。修改覆盖**五个构建**：
`paper/cache_audit/{main.tex,main_v2.tex,main.bbl,references.bib}`、
`paper/cache_audit/arxiv/`、`paper/cache_audit_anon/`、`paper/cache_audit_iclr2026/`。

## 已完成的修改

| 条目 | 改动 |
|---|---|
| P0-3 | 新增两条 bib（`cachebridge2026`、`contextreuse2026`，作者/标题按 arXiv 元数据核对）；Introduction 增加"文献已同时出现近原生保留与高于接收方基线增益"的定位句；相关工作增加一段（99.83% 保留、LongBench2 27.59%→34.48%）；Table 1 增加两行 |
| P0-2 | H3 定义改为"**tested teacher-derived cache content provides an incremental benefit when introduced into an otherwise native student cache**"；Introduction 的 "a null result cannot be blamed on the mapper" 改为 "is not specific to the end-to-end replacement score of a single mapper" |
| 术语 | "Oracle probes" 全篇统一为 "Teacher-content probes"，含**标题**与 PDF 元数据；仅保留图文件名 `fig3_oracle_probes.pdf` |
| P1-1 | 删除 family-wise 段落的重复句（保留 "bootstrap 95% upper estimate" 弱化措辞） |
| P1-2 | Figure 1 caption："capability upper bound" → "reference point" |
| P1-3 | §4.4："keeps the student's own cache dominant or localized" → "retains a controlled fraction of the student's cache or localizes the intervention to selected layers" |
| P1-4 | §5.6："the mapped keys steer attention to plausible but task-irrelevant positions" → "the observed asymmetry is consistent with key-side routing errors" |
| P1-5 | RAT："is to rule out the objection that…" → "tests whether one architecture-informed translator changes the verdict on a failure that could otherwise be read as…" |
| P1-6 | cross-family："is non-inferior but gains nothing" → "clears the reported ε=0.02 replacement margin but gains nothing"；Table 3 表头 "Gate" → "ε=0.02 reporting gate" |
| P2-3 | 修 "no tested strong-student mapper establishes replacement" 小写句首与主语关系；另修一处 `*form*` 字面星号 → `\emph{form}` |

镜像同步做法：`main_v2.tex` 与 `cache_audit/arxiv/main.tex` 与 `cache_audit/main.tex` 逐字节一致；
`cache_audit_anon/main.tex` = 正文 + 匿名化（去作者块、`pdfauthor={Anonymous}`、"accompanying repository"）；
`cache_audit_iclr2026/main.tex` 由脚本从正文重新生成，只保留 ICLR 类、作者块、`\scriptsize` 表宽与
`iclr2026_conference` 参考文献样式等版式差异。`.bbl` 已按新 bib 重新生成（21 → 23 条）。

## 验证（本地用便携 TeX 引擎构建）

| 构建 | 结果 |
|---|---|
| `cache_audit/main.tex`（article） | 通过，20 页，无未定义引用 |
| `cache_audit_anon/main.tex` | 通过，20 页，无未定义引用 |
| `cache_audit_iclr2026/main.tex`（完整版） | 通过，20 页，无未定义引用 |
| `cache_audit_iclr2026/main_submission.tex`（新，精简版） | 通过，21 页，无未定义引用 |

> 说明：本机无 pdflatex，PDF 是用便携引擎在临时目录构建、仅用于校验的；**仓库内的 PDF 需要在有 LaTeX
> 的机器上重新生成**。本机构建的分页与原有 pdflatex 构建略有差异（浮动体落位不同）。

## S-1：9 页投稿版进展

新建 `paper/cache_audit_iclr2026/main_submission.tex`（匿名，ICLR 版式），把下列材料**整体搬入附录**
（内容一字未删，主文只留指向句）：

相关工作总结表、相关工作的三条支线（单模型复用 / 表示对齐 / 冻结模型可控性）、Evidence Tiers 与多重性分析、
mapper landscape 图、cross-architecture 表、perplexity–accuracy 解耦与 channel asymmetry、
Implications 全文、What remains unmeasured 全文。

**结果：主文从 15 页压到 12 页，仍需再减 3 页才能满足 ICLR 的 9 页主文限制。**

剩余的 3 页只能靠"动证据/动分析"来换，候选与预估收益：

| 选项 | 预估节省 | 代价 |
|---|---|---|
| 主结果表瘦身（保留约 16 行、全表放附录） | ~0.6 页 | 主文不再一览所有 run |
| 主结果表 caption 精简（现约 30 行） | ~0.2 页 | 需保留 Gate/marker 定义 |
| §6 架构分析挪入附录（主文留 2 句） | ~0.9 页 | 审稿人认为这是本文最中心的两个洞见之一 |
| H2 节（约 1,350 词）细节挪附录 | ~0.8 页 | 主文只剩结论数字 |
| Introduction / 相关工作散文再压 | ~0.5 页 | 纯改写 |

其中"主结果表瘦身 + caption 精简 + H2 细节挪附录"约 1.6 页，可到 10.4 页；
再叠加 §6 挪附录才可能落到 9 页。**这一步等于决定"把哪块核心证据降级到附录"，属于科研取舍，
建议由作者拍板**；给出选项后我可以直接执行。
