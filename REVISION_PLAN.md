# 论文修改方案（REVISION_PLAN）

**日期**：2026-08-30
**输入**：`paper_audit.md`（审稿1，Weak Accept/Major Revision）、`paper_audit2.md`（审稿2，Weak Reject/4分）、内部 `paper/cache_audit/review_report.md`、运行产物复核（`reports/runs/`）
**目标稿**：`paper/cache_audit/main.tex`（2026-08-29 21:14 修订版；`arxiv/main.tex` 与其 diff=0）
**状态基线**：两份外部审稿引用的旧稿文字（"lower bound on what a perfect translator"、"appendix-scale runs"、"CHG > 0 with CI excluding zero"、"21 runs"）在当前 tex 中均已不存在——审稿部分针对旧版；但**结构性问题（oracle 逻辑、claim 强度、统计口径）在当前稿仍然存在**。

---

## 0. 事实基线（复核结论，修改方案的前提）

| # | 事实 | 证据 |
|---|---|---|
| F1 | v1.5 协议的固定评测集受控阶梯**已存在** | `configs/v15_4b_1.7b_affine_c{30,200}_tail.yaml` 均 `eval_from_tail_n: 100`；metrics：c30 = −0.2493 [−0.350, −0.149]，c200 = −0.2438 [−0.337, −0.142]，Δ=0.005 |
| F2 | **8B→1.7B 已有 v1.5 干净运行，未入稿** | `v15-8b-to-1.7b-ridge-c500-20260830-131432`：CHG **−0.224** [−0.320, −0.123]，acc −0.21 [−0.34,−0.09]，n=100，self_kv ≈ 0（H1 在 8B 对通过）|
| F3 | 8B→0.6B / 8B 多 seed 运行均为 v1.1 旧协议 | `8b-to-*-s{42,123,456}`（20260829 01:xx），内部审稿已裁定 excluded，不可引用 |
| F4 | 三种子数据在 Limitations（−0.14/−0.47/−0.46） | 对应 `v15-4b-to-1.7b-affine-c30-s{42,43,44}`（n=30 评测协议，`eval_from_tail_n: 0`）|
| F5 | affine 来源的探针复跑**已做**（内部 A1 修复项） | §5.4 L526-531 + `v15-4b-to-1.7b-affine-c30-probes-s42-*` |
| F6 | native 教师内容注入数据**已存在**（内容上界证据） | Table "Native average" −0.334 [−0.508, −0.154]；v15 tail 运行内含 `ridge_native` 消融模式（c30 −0.266 / c200 −0.251）|
| F7 | 当前稿残留三处 "bounded at zero" 级强表述 | Intro 贡献段 L131；§5.4 标题 L504；Conclusion L640-641；Abstract 强句 L57-59 |
| F8 | Limitations 无 `\label` | `grep label{sec:` 无 limitations 项 |
| F9 | λ 惰性句已不在当前稿 | 全文仅 RAT 公式含 λ（L378）——内部 A4 已随旧稿消失 |
| F10 | 运行数口径混乱 | 论文 42 / README 30 / 内部审稿 21+ / 外部审稿引用 21 |

---

## 1. 审稿点处置总表

| 来源 | 审稿点 | 裁定 | 处置 |
|---|---|---|---|
| 审2§二/三 | Oracle probe 不是 bound（逻辑致命） | **采纳（最高优先）** | P0-1/2/3/4 + P0-5 + P1-4 |
| 审2§六 | §6 是直觉非不可能性证明 | **采纳** | P0-5 |
| 审2§七/审1minor | "capability in weights" 范畴错误、需降级 | **采纳** | P0-3/4 |
| 审1M1/审2§八 | 标题/摘要普适性 vs 单家族两对 | **采纳**（收窄+8B入稿） | P0-6 + P1-1 + P2-1 |
| 审2§四 | H2 gate 判据矛盾 | **采纳**（当前稿已拆 gate，补数值 ε） | P0-9 |
| 审2§五 | c30/c200 评测集混淆 | **部分采纳**（受控数据已在 F1；修表格呈现+补 learning curve） | P1-3 + P1-12 |
| 审1M2 | CI 不覆盖 mapper 训练方差 | **采纳**（三种子已有 F4，提升至正文并反转叙事） | P1-8 |
| 审1M3 | MoT/Heo 批评缺复现、Heo 差距未讨论 | **采纳讨论版**（复现降为 P2） | P1-5 |
| 审1M4 | RAT 定位需重排 | **采纳** | P1-15 |
| 审1M5 | text channel 无方法学 | **采纳**（先验证数据 F6 同级动作） | P1-6 |
| 审1minor#5 | 任务选择与长上下文动机脱节 | **采纳**（内部审稿漏项） | P1-7 |
| 审2§九 | 0.6B 重构为 prefill substitution | **采纳**（保留 near-chance 警示） | P1-14 |
| 审2§十 | 缺 over-parameterized 非线性译者 | **降级为可选**（claim 收窄后非 blocker） | P2-3 |
| 内部A1 | 探针内容源 | **已解决**（F5）+ 补内容上界入正文 | P1-4 |
| 内部A2 | 阶梯混淆 | 同审2§五 | P1-3/12 |
| 内部A3 | gate 矛盾 | 同审2§四 | P0-9 |
| 内部A4 | λ 惰性 | **已随旧稿消失**（F9），无需动作 | — |
| 内部A5 | 0.6B near-chance | 并入 P1-14 | P1-14 |
| 内部B1/D1 | multi-seed | 已补（F4），提升至正文 | P1-8 |
| 内部B2/B3 | 表格 eval-n 列、单位标注 | 采纳 | P1-12/13 |
| 内部C1 | "none" 过绝对 | 采纳 | P0-7/8 |
| 内部C2 | checklist 未成盒 | 采纳 | P1-11 |
| 内部D2 | 更细探针窗口 | 可选 | P2-4 |
| 内部D3 | R² 量化 | 采纳（低成本） | P1-16 |
| 内部E | 文献占位作者、句子长度 | 采纳 | P1-10 + 可选风格 pass |
| 导师复核 | 8B 弹药未用、运行数口径 | 新增动作 | P1-1 + P0-10 |

---

## 2. P0 —— 措辞校准（零 GPU，先做）

> 行号均指当前 `main.tex`（08-29 版）。每项改完跑 `pdflatex main && bibtex main && pdflatex main && pdflatex main` 验证。

### P0-1 Abstract 强句降级（L57-59）
```latex
% BEFORE:
The teacher's answer-relevant advantage, in short, does not survive
KV-space translation under zero re-prefill: it sits in the teacher's
parameters, where the cache does not carry it.
% AFTER:
Under zero re-prefill, no configuration we test lets the student
extract the teacher's answer-relevant advantage from its cache; the
evidence is consistent with an advantage that is weight-mediated
rather than recoverable from raw KV states by the tested translators.
```
同步：L38-39 的 pair 列表在 P1-1 落地后改为 `(8B$\to$1.7B, 4B$\to$1.7B, and 4B$\to$0.6B)`。

### P0-2 Intro 贡献段（L131）
```latex
% BEFORE: exploitability is bounded at zero
% AFTER:  no probe configuration with teacher content beats the
%         student's own cache
```

### P0-3 Conclusion（L640-642）
```latex
% BEFORE:
Oracle probes bound the student-readable advantage of the teacher's
cache at zero, and a near-native-perplexity cache still fails the
task. The teacher's capability is in its parameters.
% AFTER:
No probe configuration delivers a student-readable advantage from the
teacher's cache, and a near-native-perplexity cache still fails the
task. The evidence is consistent with a weight-mediated advantage:
recovering the teacher's edge from raw KV states would take more than
the tested translators provide (scope limits in
Section~\ref{sec:limitations}).
```
配套：Limitations 节（L617）加 `\label{sec:limitations}`（F8）。

### P0-4 §5.4 标题（L504）
```latex
% BEFORE: \subsection{H3: Oracle Probes Bound Exploitability at Zero}
% AFTER:  \subsection{H3: No Teacher-Content Probe Improves on the Student}
```

### P0-5 §6 降级为"架构性解释"（L564-588）
1. 节首（L567 前后）加定位句：
```latex
We read this section as an architectural explanation of the audit's
verdicts, not an impossibility result: the argument is per-head and
does not cover exploit paths that read the full multi-head,
multi-layer cache (Section~\ref{sec:limitations}).
```
2. L573-575：`invisible to the teacher's own cache and therefore unavailable to any translator, however nonlinear` → `invisible in that head's value entry, so the per-head student value is underdetermined by its teacher counterpart`。
3. L575-576：`so $V_S$ is not a function of $V_T$` → `so the per-head student value is not determined by the per-head teacher value alone`。
4. L588：`and that output does not carry the teacher's capability` → `and that output shows no measurable teacher advantage in our audit`。

### P0-6 标题（L23-24）+ hyperref pdftitle（L17）
```latex
% RECOMMENDED:
\title{Cache Translation Across Model Scale:\\
An Audit with Identity Controls and Oracle Probes}
```
理由：单家族数据撑不起 "Heterogeneous LLMs"；"Bounds" 一词随 P0-2/3/4 一并退出。备选（若 P2-1 跨家族落地）：`Across Heterogeneous LLMs` 可保留。

### P0-7 Table 1（L301-302）单元格
```latex
% BEFORE: \emph{none}: identity control + oracle bounds
% AFTER:  \emph{none left untested}: identity control + probes
```
表题（L281-282）`None of the cited evaluations uses an identity control or an exploitability probe.` →
`No cited evaluation isolates mechanics with an identity injection or reports accuracy at matched perplexity; MoT's injection-window ablations are the closest antecedent (Section~\ref{sec:survey}).`（内部 C1：承认 MoT 消融先例，可辩护差异是 zero-re-prefill + identity injection）

### P0-8 H2 gate 补数值 margin（L332-336）
在现有 replacement/gate 定义后加：
```latex
We fix $\epsilon = 0.02$ gold probability before evaluation, making
the replacement gate a non-inferiority test of
$H_0\colon \chg \le -\epsilon$ against $H_1\colon \chg > -\epsilon$.
```

### P0-9 Figure 1 说明句（审1 minor#2）
在 §1 或 §3.2 的 H2→H3 转移处补一句显式理由（现状 L338-341 已有雏形，提级到 Intro）：
```latex
H2's failure routes the audit to H3 because the probes measure what a
frozen student can extract from teacher-cache content without
training anything, so a null result cannot be blamed on the mapper.
```

### P0-10 运行数口径统一
从 `reports/runs/*/inject-eval/metrics.json` 重计"审计协议运行"数（建议计数规则：v1.2 及以后、inject-eval、入稿数字的运行），统一替换 paper/README/EXPERIMENT_LOG 三处。当前论文写 42，重计后以实际为准。

---

## 3. P1 —— 低成本实验 + 正文增强

### P1-1 8B→1.7B 入稿（零 GPU，F2）⭐ 最高性价比
- Table 2 加一行：`Ridge per-head, c=500, 8B→1.7B & — / — / — & $-0.224$ [$-0.320, -0.123$]`（acc/gold/PPL 从 metrics.json 补齐；注意标注 **ridge、c=500**，与 4B 主表 affine/c30-200 不同族）。
- §5.2 加一句：teacher 扩到 8B、同协议、self_kv=0、CHG 同为负且更强——H2 失败对 teacher 规模稳健。
- Abstract pair 列表、§4.5、Limitations 的 pair 数同步从 2 → 3。
- Limitations 中"单一 4B teacher"相关表述删除。

### P1-2 8B→0.6B v1.5 运行（1 次 GPU）
- 新建 `configs/v15_8b_0.6b_affine_c30_tail.yaml`（以 `v15_4b_1.7b_affine_c30_tail.yaml` 为模板，换模型路径）。
- 目的：补齐 capacity 轴第四点。预期两种结果都有信息量：
  - 若 parity（如 4B→0.6B）→ 支持"弱学生绝对能力决定 substitution"叙事；
  - 若 fail → 支持"teacher/student 比率"叙事。
- 落点：§5.2 或 Implications 的 phase-transition 段。

### P1-3 校准预算 learning curve（4 次 GPU）
- 固定 eval（tail 100）、affine per-head、c ∈ {10, 60, 100, 500}（c30/c200 tail 已有，F1）。
- 新建 `configs/v15_4b_1.7b_affine_c{10,60,100,500}_tail.yaml`。
- 产出：CHG vs c 曲线（新小图或表内一行），**永久关闭** A2/审2§五类攻击。
- 写作时与 F1 数字合并叙述："budget inert on a fixed evaluation set"。

### P1-4 内容上界证据提级（零 GPU，F6）
§5.4 把附录/表格中的 native 注入证据提为正文论点：
```latex
The content side has its own ceiling: injecting the teacher's native
(norm-rescaled) cache entries directly degrades the student to
$-0.334$ [$-0.508, -0.154$], so even untranslated teacher content in
student coordinates does not help -- the harm is not an artifact of
any particular mapper.
```
（这直接回应审2 的 off-manifold 反驳：问题不在"我们的映射器弄坏了内容"，native 内容同样无效。）

### P1-5 Heo/MoT 差距讨论段（零 GPU，审1M3）
在 §2.2 或 §5.2 加一段，归因 ridge-per-head −0.296 vs Heo 73–98% 留存率：
（a）Heo 设置非 strict zero-re-prefill / 含 replay 或 correction；（b）校准协议（配对样本、损失）不同；（c）pair 不同。明确我们的主张限定在 zero-re-prefill regime。**不做**完整 MoT 复现（P2-2）。

### P1-6 text-channel 方法学补写（零 GPU，审1M5）
- 先验证数据：定位 teacher-summary 基线运行产物（v1.2 期），确认 prompt 格式、n、eval 是否 disjoint；若 CI 缺失，对已存 per-sample 分数做 bootstrap（无 GPU）。
- 在 §4 加一小段或 §7 加脚注：prompt、n、CI、disjoint 与否，与主表数字可对齐。

### P1-7 任务选择论证（零 GPU，审1 minor#5）
§4.5 或 Limitations 加 2-3 句：MCQA 选型是为了 gold-probability 可控测量 + disjoint split 的审计需要；长上下文 prefill 复用是部署动机，任务多样性列为后续工作；明确当前结论 scope 为短上下文 MCQA。

### P1-8 三种子提级正文 + 叙事反转（零 GPU，F4）
§5.2 加：
```latex
A three-seed sweep of the headline mapper (affine, $c{=}30$, $n{=}30$
evaluation) bounds mapper-training variance: $-0.14$ (seed 42, the
reported configuration), $-0.47$ (43), $-0.46$ (44). The headline
number is the most favorable seed, and the sign is stable.
```
同时在 §3.1 或脚注声明：bootstrap CI 覆盖评测抽样不确定性，mapper 训练方差由 seed sweep 单独报告。Limitations 中对应句子去重。

### P1-9 channel asymmetry 补假设句（零 GPU）
§5.6 末尾加：
```latex
One hypothesis consistent with the asymmetry: mapped keys steer
attention routing while mapped values carry content, so key errors
misroute a fluent read while value errors corrupt it audibly.
```

### P1-10 文献占位作者（内部 E）
KVComm / Interlat / HCache / activation-steering 四条补真实作者或删除；`references.bib` 全查 "and others"/"Others"。

### P1-11 audit checklist 盒装（内部 C2）
§7 开头把三仪器+gate 排成 `itemize` 或 tcolorbox 盒，摘要/conclusion 引用之。

### P1-12 Table 2 加 eval-n 列（内部 B2）
所有行标注评测 n（30/63/100）；c30/c200 与 tail 受控行分栏或分表，消除 −0.138 vs −0.249 两套数字的歧义。

### P1-13 单位标注（内部 B3）
acc 与 gold 首次出现处及图题统一加 "(accuracy)" / "(gold prob.)"。

### P1-14 0.6B 重构：prefill substitution + near-chance 警示（审2§九 + 内部A5）
§7 加一段：
```latex
For the weak student the audit supports a different primitive: not
capability transfer but prefill substitution. At 0.6B, translated
injection matches self-prefill ($+0.010$), so a cached handoff saves
the student's prefill -- though at near-chance absolute accuracy
(0.302 vs.\ 0.25 random), this is parity, not endorsement. Whether a
capacity boundary separates the two regimes -- replacement below it,
failure above it -- is the question the 8B and 0.6B pairs begin to
answer, and the audit protocol is built to test it.
```
（若 P1-2 完成，把 8B→0.6B 结果填入。）

### P1-15 RAT 重定位（审1M4）
- Intro 贡献 2（L126-128）：把 RAT 从并列主方法改为"we additionally derive the most architecture-informed translator we could (RAT); it fails too, which strengthens the negative"。
- §5.3 末尾加一句定位句：RAT 的价值是排除"失败因为我们不够架构知情"这一反驳。

### P1-16 mapper 重构 R² 平台测量（内部 D3，低成本）
对 best mapper 在 held-out 校准对上报 per-(layer,head) 重建 R²，一句入 §6，量化"underdetermined"。（仅需 teacher/student 在校准集上的 cache 前向，无完整 eval。）

---

## 4. P2 —— 可选强化（冲更高档位再做）

| # | 项 | 触发条件 | 成本 |
|---|---|---|---|
| P2-1 | 跨家族 pair（Qwen→Llama/Gemma） | 若想保留/恢复 "heterogeneous" 措辞 | 高（tokenization/维度不对齐需新映射层） |
| P2-2 | MoT sans-replay 受控复现 | rebuttal 被要求实证时 | 高 |
| P2-3 | over-parameterized 译者（cross-attention/Perceiver 第 7 族） | 若审稿坚持"非线性未测" | 中高 |
| P2-4 | 更细探针窗口（per-octant） | 内部 D2 | 中 |
| P2-5 | 长上下文任务（长文档 QA） | 补齐动机-任务链 | 中高 |
| P2-6 | 全文风格 pass（SNL ~21 词/句） | 投稿前 | 零 GPU |

---

## 5. 驳回 / 降级的审稿点（rebuttal 备用）

1. **"必须证明不存在非线性 translator 才能下结论"（审2§二）** — 驳回其证明要求，采纳其降级处方。经验论文不能证全称否定；公平标准是 claim 匹配证据 + 诚实 scope。P0-1~5 即为落实。
2. **"必须补 over-parameterized oracle 才能保留 H3"（审2§十）** — 降为 P2-3。claim 收窄到 "the tested translator families" 后非 blocker；论文自有的 MLP mapper（−0.303/−0.236）+ P1-4 native 内容证据已部分封堵该反驳。
3. **"c30/c200 混淆故 'more data hurts' 无据"（审2§五）** — 部分驳回：受控数据已存在（F1：−0.249 vs −0.244，固定 n=100），审稿看的是旧版且漏掉正文受控对比；我们仍修表格呈现（P1-12）并补 learning curve（P1-3）。
4. **"正文未展示 multi-seed"（审1M2）** — 已过时：三种子数字在当前稿 Limitations 且有运行产物（F4）；动作是提级与反转叙事（P1-8）。
5. **"必须复现 MoT/Heo 之一"（审1M3）** — 降为 P2-2：对 methodology/audit 论文，归因讨论 + 明确 regime 限定（P1-5）在多数审稿场景已足够；完整复现作为 stretch。

---

## 6. 执行顺序与验证清单

```
Step 0  重编译当前 PDF（pdflatex×3+bibtex）；用当前稿重跑一轮独立审稿
        → 校准哪些审稿点已被 08-29 修订解决，避免重复劳动        [零 GPU]
Step 1  P0-1 ~ P0-10 措辞校准（含 \label{sec:limitations}）      [零 GPU]
        验证：编译 0 error；grep 全文无 "bounded at zero"、
        "however nonlinear"、"is not a function of $V_T$"（未限定版）
Step 2  P1-1（8B 入稿）+ P1-4/5/6/7/8/9/11/12/13/14/15 正文增强   [零 GPU]
        验证：每个新数字 grep 对应 reports/runs metrics.json；
        运行计数与 F10 统一
Step 3  P1-2（8B→0.6B）+ P1-3（learning curve c10/60/100/500）    [5 次 GPU]
        验证：pytest tests/ 133 通过；新 config 跑通且 self_kv≈0；
        结果落 reports/runs/ 带 git hash
Step 4  P1-10（bib）+ P1-16（R²）；终稿重编译 + 重跑外部审稿复核    [低成本]
Step 5  视目标档位决定 P2 子集
```

**完成判据**：
- [ ] 全文无 "bound...at zero" 级不可支撑表述；Abstract/Conclusion 与 Limitations 措辞一致
- [ ] 标题与实验范围匹配（within-family scale pairs 或跨家族落地）
- [ ] 每个入稿数字可溯源到 `reports/runs/*/inject-eval/metrics.json`
- [ ] H2 gate 有预注册 ε；表格标注 eval-n；运行数三处一致
- [ ] 0.6B 段含 near-chance 警示；RAT 定位为"架构知情假设也失败"
- [ ] bib 无占位作者
