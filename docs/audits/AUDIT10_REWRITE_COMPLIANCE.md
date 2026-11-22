# AUDIT-10 修订完成度记录

日期：2026-11-22
依据：`docs/audits/audit10_1.md`（第十轮审查）
工具链：本机 Windows 侧无 LaTeX，但 **WSL Ubuntu24.04 内装有 TeX Live 2023/Debian**
（即 2026-09 那批 PDF 用的同一引擎），全部 PDF 已重建（2026-11-22）。
本轮不涉及任何 GPU 实验、不重跑 run、不改任何数字。

## 一、逐项完成情况

| 编号 | 审查发现 | 状态 | 落点 |
|---|---|---|---|
| B1 | §5.2 锚点消融方向写反（去锚点实际把 PPL 从 501.6 降到 23.7） | ✅ 已修 | arxiv/iclr/zh/humanized×2 §5.2 + arxiv/iclr/zh 摘要 |
| B2 | ε=0.02 非劣门在 n=30 下无功效 | ✅ 已修 | arxiv/iclr/zh §3.2（H2 判定门） |
| B3 | §5.2 区间 −0.138~−0.392 与 Table 1 的 −0.596 冲突 | ✅ 已修 | arxiv/iclr/zh §5.2 + README |
| B4 | 未对账 heo2026 的“MLP 可救失败对” | ✅ 已修 | 新增附录 `app:heo-mlp` + 表 `tab:heo-diff`（arxiv/iclr/zh），§2 与 §5.2 各一句指针 |
| B5 | CacheBridge / MoT 已升 v2，表述过时 | ✅ 已修 | 三个 references.bib 更新标题+`note`；§2 改为“比设计不比数字”+ 时点声明 |
| C1 | 漏引 LatentPort（早于定稿） | ✅ 已补 | §2、`app:concurrent`、§7 |
| C2 | CacheLink 独立验证去耦 | ✅ 已补 | §6.1、§7、`app:concurrent` |
| C3 | 行为目标族未测，作用域需写死 | ✅ 已写 | `app:concurrent` + §8 |
| C4 | 领域已放弃严格零 re-prefill | ✅ 已写 | §7（“能力由重算而非更多缓存恢复”） |
| C6 | 确认性探针跑在带锚 RAT 上 | ⚠️ 仅 arxiv | arxiv §5.3 一句说明；iclr/zh 未加 |
| D1 | README 计数 171 → 177 | ✅ 已修 | `README.md` L44 |
| D2 | ε 敏感性前置 | ⚠️ 部分 | 由 B2 的量化说明部分覆盖；未加脚注 |
| D4 | 中英同步 | ✅ 已同步 | 见下 |

**新增被引（6 条，全部经 arXiv API 核验存在，作者/标题逐字取自 API）**：
`latentport2026`(2609.25053)、`cachelink2026`(2610.15509)、`kvlingo2026`(2609.32610)、
`bridgekv2026`(2611.11761)、`raracache2026`(2610.11358)、`acct2026`(2610.42116)。

## 二、各文件改动摘要

| 文件 | §2 | §3.2 | §5.2 | §5.3 | §6.1 | §7 | §8 | 附录 | 摘要 | bib |
|---|---|---|---|---|---|---|---|---|---|---|
| `paper/arxiv/main.tex` | 改 | 改 | 改 | 加1句 | 加1句 | 改 | 改 | +2 节 +1 表 | 改 | — |
| `paper/iclr2026/main.tex` | 改 | 改 | 改 | 未改 | 加1句 | 改 | 改 | +2 节 +1 表 | 改 | — |
| `paper/zh/main.tex` | 改 | 改 | 改 | 未改 | 未改 | 未改 | 未改 | +2 节 +1 表 | 改 | — |
| `paper/humanized/*/main.tex` | — | — | 改（仅方向） | — | — | — | — | — | — | — |
| 三个 `references.bib` | — | — | — | — | — | — | — | — | — | +6 条 +3 处修正 |

`humanized/` 只修了 B1（事实错误），其余新增内容未回填：它是“去 AI 味副本”，
如需启用，应先按 `paper/arxiv` 重新 humanize，而不是手工拼接。

中文版 `paper/zh` 落后于英文稿的三处（§5.3 RAT 说明、§6.1 CacheLink 一句、§7 同期工作段落）
已在下表登记；因中文版是“逐段对照”件，这三处会造成中英段落不对齐。

## 三、已执行的验证（本机实测）

```
python scripts/paper_consistency_check.py      → PASS 48 checks, FAIL 0 checks
```

覆盖：Table 1 全 23 行、probe 表 7 行、家族 CI 表 3 行、附录区间、派生算术
（ε/基线≈4%、2.5 PPL 差、23 点准确率差、阶梯 spread 0.0191）、摘要级数字 9/9、
无 1e3 陈旧端点、交叉引用、citation key。

另有自建检查（scratch，不入库），五个 tex 全部通过：

| 文件 | cite keys | bib | 缺失 | 未引 | 悬挂 ref | 环境平衡 |
|---|---|---|---|---|---|---|
| arxiv | 31 | 31 | 无 | 无 | 无 | ✓ |
| iclr2026 | 31 | 31 | 无 | 无 | 无 | ✓ |
| zh | 31 | 31 | 无 | 无 | 无 | ✓ |
| humanized/arxiv | 25 | 25 | 无 | 无 | 无 | ✓ |
| humanized/iclr2026 | 25 | 25 | 无 | 无 | 无 | ✓ |

## 四、未完成 / 需要在有 TeX 的机器上做

1. ~~重建 PDF~~ **已完成**（2026-11-22，WSL TeX Live 2023）。页数取自各稿 `main.aux`
   的 `\newlabel{sec:conclusion}`，不依赖 PDF 文本抽取：

   | 文件 | 页数 | 引擎 | 未定义引用 |
   |---|---|---|---|
   | `paper/arxiv/main.pdf` | 15（结论在第 10 页） | pdflatex×3 + bibtex | 0 |
   | `paper/iclr2026/main.pdf` | 14（结论在第 9 页） | pdflatex×3 + bibtex | 0 |
   | `paper/zh/main.pdf` | 15（结论在第 9 页） | xelatex×3 + bibtex | 0 |
   | `paper/humanized/arxiv/main.pdf` | 13 | pdflatex×3 + bibtex | 0 |
   | `paper/humanized/iclr2026/main.pdf` | 13（结论在第 9 页） | pdflatex×3 + bibtex | 0 |

   根目录两份分发 PDF 已用新产物覆盖，`Get-FileHash` 与 `paper/arxiv/main.pdf`、
   `paper/zh/main.pdf` 逐一相同。
2. **ICLR 9 页限制：已验证满足**。结论在第 9 页、附录 A 自第 11 页起，正文未溢出；
   原计划的压缩方案未启用。
3. ~~重打 arXiv 分发件~~ **已完成**：`arxiv_submission.zip`（108,932 B）与
   `arxiv_submission.tar.gz`（96,845 B）按原结构重打（`figures/` 层级 +
   `main.tex`/`main.bbl`/`references.bib`/`README_SUBMIT.md`）；解压到干净目录实测
   `pdflatex + bibtex + pdflatex×2` 退出码 0、15 页、0 未定义引用、0 `not found`，
   产物 685,777 B 与 `paper/arxiv/main.pdf` 逐字节一致。`README_SUBMIT.md` 的
   Comments 已更新为 “15 pages, 4 figures, 6 tables”，表单纯文本摘要已同步。
4. **中文版三处未同步**（可选）：§5.3 RAT 说明、§6.1 CacheLink 一句、§7 同期工作段落。
5. **`docs/paper/CLAIM_MAP.md`** 未更新（新增的 6 条外部引用与 2 个附录小节未登记）。

## 五、本轮不做的（维持审查报告判断）

- 不改任何数值、不重跑 run、不做 GPU 实验；
- 不实现“行为目标译码器”分辨实验（审查报告第三档建议，需 GPU）；
- 不动 `paper/archive/` 下历史版本。
