# 第十轮独立审查（定稿 +2 月，文献复核版）

日期：2026-11-22
对象：`paper/arxiv/main.tex`（arXiv 完整稿，2026-09-21 定稿）、`paper/iclr2026/`、`paper/zh/`、`README.md`
方法：先读稿 → 联网逐条核验引用（arXiv API）→ 检索 2026-09 之后的新工作 → 用 `reports/runs/*/metrics.json`
      与 `scripts/aggregate_claims.py` 复核正文数字 → 做功效/样本量定量

## 一、总体结论

**结论方向成立，但正文有 1 处自相矛盾的事实错误（数据可定案），统计论证有 1 处结构性弱点，
相关工作的定位已被 2026-10/11 的新文献改写。**

前九轮审查（audit1–audit9）把注意力放在"数字↔证据一致、引用真实性、打包、匿名、页数"，
本轮补的是前九轮没覆盖的两类问题：**(i) 数字之间的一致性（而非数字与 run 的一致性）**，
**(ii) 定稿之后文献图景的变化**。前者抓到一个必改项，后者改变了论文的定位建议。

判定：**需 minor revision 后再上传/投稿。** 核心负结果没有被推翻；但若按现状上传，
一个读过 CacheLink / KV-Lingo / BridgeKV 的审稿人会给出"over-claims"与"stale related work"两条意见。

## 二、论文理解（我读到的论证结构）

论文把"跨规模缓存翻译"拆成三个可独立失败的环节，每个配一个仪器与一个门：

| 假说 | 问题 | 仪器 | 门 | 结果 |
|---|---|---|---|---|
| H1 机制 | 注入管线能否正确消费外来缓存？ | `self_kv` 恒等注入 | 逐样本 logit cos ≥ 0.9998 | 通过（min 0.99986） |
| H2 映射 | 某个映射器能否达到替换水平？ | CHG（gold-letter prob 的配对变化）+ 10⁴ bootstrap | CI 下界 > −ε（ε=0.02） | 强学生全族失败 |
| H3 可利用性 | 教师缓存里是否有冻结学生可用的内容？ | fraction / window / octant 探针（不训练新译码器） | 某探针 CI 排除零的正侧 | 无一行通过 |

证据规模：177 个含 metrics 的评测 run、181 个协议行（133 translations + 48 probes）、
7 族译码器、4 个 Qwen3 规模对 + 5 个替代学生、1K/4K/8K 长上下文。
两条"去耦"作为可迁移结论：**重建保真 ≠ 功能等价**（R² 0.854/0.481 仍同 CHG 带）、
**流畅 ≠ 能力**（PPL 23.7 但 acc 0.267）。

这个结构是本文最大的资产：它不是"我们的方法更好"，而是"这个问题的测量方法应该长这样"。

## 三、发现清单

### B 类：必须改

#### B1 ［事实错误·数据可定案］§5.2 的"去锚点"消融方向写反了

正文（`main.tex` 第 394–396 行）：

> removing RAT's analytic anchor **raises** suffix perplexity from 23.7 to 501.6 while changing
> CHG from −0.223 to −0.268 — fluency moves, the answer deficit does not.

`reports/runs/` 的实际记录：

| run | PPL | CHG | Acc |
|---|---|---|---|
| `v14-4b-to-1.7b-rat-c30-20260829-174525`（**带**锚） | **501.56** | −0.2682 | 0.2333 |
| `v14-4b-to-1.7b-rat-c30-noanchor-20260829-180839`（**去**锚） | **23.66** | −0.2231 | 0.2667 |

即：**去锚点把 PPL 从 501.6 降到 23.7**（改善 ~21×），CHG 也从 −0.268 升到 −0.223。

三处互相矛盾：

1. §5.2 说"去锚点把 PPL 抬到 501.6"——反了；
2. 摘要（第 58–59 行）说 "removing the analytic anchor … **improves** perplexity without improving
   answers"——与 §5.2 相反，但与数据和 Table 1 一致；
3. Table 1 的 RAT 两行（501.6 / 23.7）与数据一致，所以是 §5.2 的句子错，不是表错。

为什么前九轮没抓到：`scripts/paper_consistency_check.py` 只校验了派生算术
`23.7 − 21.2 = 2.5` 与 `noanchor` 行的准确率（第 199、213 行），**没有校验带锚 RAT 的 PPL 501.6，
也不校验消融的方向**。数字各自都对，方向错了，这是"数字↔证据"检查抓不到的一类缺陷。

建议改写（保持论证不变，只改方向与措辞）：

> …the anchor ablation moves fluency in the *opposite* direction to the answer deficit: dropping the
> analytic anchor lowers suffix perplexity from $501.6$ to $23.7$ and shifts CHG only from $-0.268$ to
> $-0.223$, i.e. the mapper becomes fluent while remaining far below the student.

顺带两处需要一起修：
- 摘要的 "without improving answers" 也偏强：CHG 实际改善了 0.045、准确率 23.3%→26.7%，
  建议改为 "improves perplexity far more than it improves answers"。
- §5.2 的小标题 "Capacity and anchoring do not close the gap" 本身没错（结论仍成立），但读者会
  追问"既然去锚点更好，为什么确认性探针跑在带锚版本上"——见 C6。

#### B2 ［统计结构性弱点］ε=0.02 的"替换门"在本组样本量下几乎没有功效，过/不过由方差决定

用各区间的宽度反推每样本 SD，再做单侧非劣检验（下界 > −ε）的样本量计算：

| 行 | n | CI 宽 | 反推 SD | 达到 ε=0.02 所需 n | n 处的功效 |
|---|---|---|---|---|---|
| affine c30（最强线性） | 30 | 0.349 | 0.49 | ≈ 2 300 | **0.04** |
| affine c200 | 63 | 0.231 | 0.47 | ≈ 2 100 | 0.05 |
| 弱学生 4B→0.6B | 63 | 0.195 | 0.39 | ≈ 1 500 | 0.06 |
| llama-1b rect（**唯一过线**） | 100 | 0.024 | 0.06 | ≈ 35 | **0.92** |

结论：在 n=30–63 下，任何 SD≈0.4–0.5 的译码器**都不可能**过 ε=0.02 的门（除非点估计 > +0.1），
这个门对它们没有信息量；而"过线"的几行能过，是因为它们在接近随机的学生上 CI 特别窄（SD≈0.06），
不是因为这些译码器更准。也就是说 **`Repl.` 列在很大程度上是精度列，不是质量列**。

论文已经在弱学生处承认了这一逻辑（"establishes neither degradation nor the reporting margin"），
但没有把它推广到全表，也没有报告所需样本量。审稿人会把"no translator reaches replacement level"
读成"这些译码器不行"，而严格读法只是"在 n≤100 下无法证明非劣"。

建议（任选其一，第一种最省事）：
1. 在 §3.2 加一句量化的精度声明："at $n{=}30$ the paired interval half-width is $\approx 0.17$, so the
   $\epsilon{=}0.02$ non-inferiority test would require $n\approx 2.3\times10^3$ same-size samples; the
   gate is therefore reported for completeness and the direction judgement carries the inference."
2. 把 `Repl.` 列改为 TOST 表述并给出 n\*（每行所需样本量）。
3. 删掉 `Repl.` 列，只保留 direction + 一致性区间，把 ε 讨论收到附录。

无论哪种，都要保留"direction 判定"作为主推断——那是对的，且不受此问题影响。

#### B3 ［内部一致性］§5.2 的区间 −0.138 ~ −0.392 与 Table 1 冲突

§5.2："Table 1 collects the mapper ladder. Every strong-student translation has a negative point
estimate … from −0.138 to −0.392."

但 Table 1 自身的强学生行包含 −0.596（4K，n=10）与 −0.577（8K，n=10），二者都是 4B→1.7B
（`configs/e4_longctx4k.yaml`、`e4_longctx8k.yaml` 确认 teacher=Qwen3-4B、student=Qwen3-1.7B）。
§6.4 也确实把它们当作"不同评测 regime"另述。所以这是**作用域未界定**，不是数据错。

建议：加限定词——"in the four-choice classification setting (Table 1, upper block) …; the long-context
rows of §6.4 reach −0.596." 或直接把区间写成 "from −0.138 down to −0.596 across all strong-student rows"。
`README.md` 的 "CHG −0.14 ~ −0.39" 同步修正。

#### B4 ［论证缺口］没有对账 heo2026 自己的"MLP 能救失败对"结论

本文把 heo2026 当作"最接近的已发表设计"并实现其线性形式（top-k 源层 + de-RoPE + 闭式 ridge），
这点做得对。但 heo2026 摘要明确写着：

> A nonlinear MLP recovers up to **+37 pp HellaSwag retention** on the failures.

而本文的 per-head MLP（−0.39 ~ −0.26）与 37M joint MLP（−0.30 ~ −0.26）**毫无帮助**，且用的正是
HellaSwag/ARC 四选一。这是审稿人第一个会问的问题："最接近的设计自己说 MLP 能救，你的 MLP 为什么不能？"

差异至少有四个维度，论文必须逐条给出可检验的说法：
1. **配对条件**：Heo 的方法要求 matched-KV 对（同 KV 头数、同头维）；本文的 MLP 是否在同一条件下比较；
2. **失败集选择**：Heo 的 +37pp 是在"两个退化剧烈的对"上报告的，属于事后选择失败集，不是全族平均；
3. **指标**：Heo 报 accuracy retention，本文报 gold-letter probability（分布更敏感、方差更大）；
4. **训练数据**：Heo 用 500×1024 FineWeb-Edu，本文用生成语料 100 条（附录表 A1 已披露）。

建议在 §5.2 或附录新增一小段 "Relation to the MLP result of \citet{heo2026}"，把这四维写清，
并把结论从"capacity does not close the gap"降级为"capacity does not close the gap **under our
calibration volume and matched-KV configuration**"。这是**降低主张、提高可信度**的改动。

#### B5 ［引用时效］两条承重引用在定稿后更新了版本，表述已不被现行版本支持

我逐条核验了 v1 与 v2（arXiv API），结论是**论文对 v1 的描述准确，不是编造**——这点要说清：

| 条目 | 论文表述 | v1（定稿时现行） | v2（现行） |
|---|---|---|---|
| `cachebridge2026` | "99.83% mean target retention on Qwen3" + "order of magnitude less calibration data" + "each target head matched to one source head" | **逐字吻合**（v1: 2026-09-01） | v2（2026-11-03）改标题为 "…Cross-Model KV **Transfer for Multi-Model LLM Serving**"，数字换成 **97.94% / 102.94% of re-prefill task quality**（5 对 / 3 族），"one tenth of the calibration data" 保留但主体从 Qwen3 换为 14B→32B |
| `mot2026` | 标题 "Mixture-of-Translators: Translating KV Caches…"；作者序 Lee, Song, Oh, Han, Park, **Jang**, Lim | **标题与 v1 一致** | v2（2026-10-03）改名为 "**KV Cache Translation across Heterogeneous Large Language Models**"；评测从 Qwen2.5/GPT-2/OPT 换成 Llama/Gemma/Qwen，报 57.6% acc、95.0% recovery |

风险：现在上传，审稿人按 v2 读，会认为论文 mischaracterize 了最新竞品（尤其"99.83%"这个数字在 v2 里
已经不存在）。这不是学术不端，是版本卫生问题，但代价很高。

建议：
1. 更新两条 bib（标题、`note = {arXiv v2, 2026-11-03}`）；
2. Related Work 里把 CacheBridge 的引用从"具体数字"改成"设计差异"（architecture-indexed 头匹配 +
   attention-aligned calibration + bounded mapper construction），数字只作括注且标明版本；
3. 加一句时点说明："as of the September 2026 versions we audited; both entries have since been revised,
   and our implementation comparison targets the design, not the reported numbers."

### C 类：重要（科学定位，建议改）

#### C1 漏引一篇**早于定稿**的直接支持性工作

`LatentPort`（arXiv:2609.25053，2026-09-07，比 9-21 定稿早两周）在混合模型上做跨模型状态交接，
摘要写明："**Translated attention KV alone leaves a large gap**"，且它自己的 near-native gate 也 failed。

这与本文 H2/H3 是同一个结论，且来自独立团队。补引的价值很高：
- 直接强化"不是我们一家测不出来"；
- 它的 "near-native gate failed" 与本文的三仪器方法论同构，可在 §7 作为方法学共识引用。

这条属于纯增益、零成本的修改，建议优先做。

#### C2 CacheLink 独立验证了本文的核心诊断，但同时打开了本文未测的目标族

`CacheLink`（arXiv:2610.15509，2026-10-11）的摘要几乎在复述本文 §6.1：

> we find that **reconstruction fidelity is a poor proxy for receiver utility**. Similar K/V reconstruction
> errors can lead to substantially different attention patterns, predictive distributions, and downstream accuracy.

这给本文一个明确的**优先权主张**：本文用 R²↔CHG 的去耦 + 恒等对照更早、更系统地给出了这个诊断，
可以在 §7 写成"independently corroborated after our freeze by \citet{cachelink2026}"。

但同时它构成威胁：CacheLink 用 **receiver behavior 直接学 coupled K/V correction**，在 4 对 / 7 面板上
"improves the equal-weight mean over all methods for every pair"。这正是一族本文**没有测**的译码器
（目标是行为而非重建误差）。

#### C3 两个"行为目标"译码器 + 一个混合状态工作，都在本文冻结之后

- `KV-Lingo`（arXiv:2609.32610，2026-09-26）：逐层线性 map，用**蒸馏**（目标模型预测散度）训练；
- `BridgeKV`（arXiv:2611.11761，2026-11-09）：large→small，learned local layer mixing + K/V 分离非线性 map，
  训练目标是对齐 next-token 预测，六对五数据集 30/30 优于微调小模型基线，闭合最多 80% 质量差；
- `Translating KV Caches Between Hybrid-Attention LMs`（arXiv:2611.12062，2026-11-09）：MLA/线性注意/稀疏注意，
  多数设定达到 native 水平。

这些都说明：**"稠密映射 + 重建目标 + 严格零 re-prefill"是一个已被绕开的组合**，本文的负结果严格属于这个组合。

建议把结论的作用域写死（这是本文最该做的一处措辞升级）：

- 标题/摘要层次：把 "no translator we test replaces" 保持不动（它是真的），
  但在 §1 Scope 与 §8 明确列出**未测族清单**：行为/蒸馏目标、selective recomputation、
  hybrid/recurrent state、adapter 微调；
- §7 的方法学结论反而因此更强："our instruments tell you *why* those families work: they optimize the
  receiver's computation, not the cache's reconstruction."

#### C4 领域正在放弃"严格零 re-prefill"，本文需要正面回应这个 regime 选择

- `RaReCache`（arXiv:2610.11358，2026-10-08）：只重算 30% 位置即保住 95–99% 目标精度；
- `ACCT`（arXiv:2610.42116，2026-10-30）：75% KV 削减下匹配或超过目标原生性能。

审稿人会问："你把最有效的解法（部分重算）排除在外，负结果的意义是什么？"

答案其实很强，但论文没写：**如果 30% 的重算就能拿回 95% 的能力，那么能力主要不在迁移过去的缓存里，
而在被重算的那些 token 上**——这恰好是本文"缓存不携带学生可读的任务增益"的另一种表述。建议在 §7
加 3–4 句把这个推论写出来，它会把"我们限制了设定"变成"我们的设定给出了机制性结论"。

#### C5 通道非对称有独立证据可以互证

本文 §6.3：只 map values 答对 36.7%、只 map keys 30.0%、两者都 map 20.0%，而 PPL 顺序相反。
`To Cache or Not to Cache`（arXiv:2610.15721，2026-10-11）发现 "**value reuse causes substantially larger
deviations than key reuse**"（安全行为偏移），并用 per-layer/per-head affine 修正 V。

两者指向同一个结论：K 与 V 通道在跨模型复用中的功能角色不对称。建议在 §6.3 引一句作外部佐证，
并把本文"without attention-map analysis we do not claim the mechanism"的自限保留——这是恰当的。

#### C6 确认性 H3 家族跑在"带锚 RAT"上，而该锚已被证明有害

§5.3 的 confirmatory family（3 个 fraction × 3 个 window）在 `RAT, c30` 上跑，而 B1 显示这个版本的
PPL 是 501.6、Acc 0.233（远差于去锚版本的 23.7 / 0.267）。用一个流畅性已崩的译码器作为"教师内容"
的来源，审稿人会质疑探针内容是否具有代表性。

好在论文已有 affine c30 的复现（全部非正），因此结论不依赖于此。建议：
- 说明选择 RAT 的理由（它是"最架构感知"的译码器，内容最接近"教师原生语义"）；
- 把 no-anchor RAT 提为**并列确认性来源**（run 已存在，成本≈0），或至少报告其 fraction 探针结果。

### D 类：小项

- **D1 计数口径**：`README.md` 写"171 个记录在案的评测 run"，正文写 177 runs / 181 行。
  我实跑 `scripts/aggregate_claims.py`：`177` runs、`181` rows（`133` translations + `48` probes），
  **与正文一致，README 的 171 是旧数**，建议同步。
- **D2 ε 敏感性**：ε∈{0.01,0.02,0.05} 的敏感性只在附录 A2，建议提到 §3.2 脚注一句话（审稿人不会翻附录找）。
- **D3 bib 形式**：`gist2023`、`stitching2021` 的 `booktitle` 是会议名但 DOI 是 arXiv DOI（audit9 已提，未改）。
- **D4 中文版同步**：以上所有改动需同步 `paper/zh/`，否则中英版本会出现 B1 那种"一边对一边错"。

## 四、评分

| 维度 | 前轮（audit9） | 本轮 | 说明 |
|---|---|---|---|
| Novelty | 8.0 | 8.0 | 三仪器框架仍是最早的系统化版本 |
| Technical quality | 8.5 | 8.0 | B1 正文自相矛盾 |
| Experimental rigor | 9.0 | 9.0 | 177 run、逐样本、10⁴ bootstrap 仍然很强 |
| Statistical rigor | 8.5 | 7.0 | B2 非劣门无功效、`Repl.` 列与精度混淆 |
| Related work | 9.0 | 7.0 | 定稿后出现 6 篇直接相关工作，其中 1 篇早于定稿未引 |
| Claim–evidence | 9.0 | 7.5 | B1、B3、B4 三处主张与表述/表不符 |
| Reproducibility | 8.5 | 8.5 | 脚本可跑，计数可复现 |
| Significance | 8.5 | 8.0 | 两条去耦持久，但作用域需收窄 |
| **Overall** | **7 / Accept** | **6.5 / 弱 Accept（minor revision）** | 结论不变，需一轮针对性修订 |

## 五、改进意见（按投入产出排序）

### 第一档：投稿/上传前必做（约半天）

1. 修 B1（§5.2 方向 + 摘要措辞 + 中文版同步）。
2. 修 B3（§5.2 区间加作用域限定 + README 的 −0.14~−0.39）。
3. 补 B5（两条 bib 更新到 v2，Related Work 去掉对单一版本数字的依赖，加时点声明）。
4. 补 C1（一句话引用 LatentPort）—— 零成本、纯增益。
5. 修 D1（README 计数 171 → 177/181）。

### 第二档：接收率的关键（约 1–2 天，主要是写作）

6. B2：在 §3.2 加一句量化的功效/精度声明（上面给了可直接用的英文句子）。
7. B4：新增 "Relation to the MLP result of heo2026" 一小节，四维对账，并相应降低 capacity 结论的强度。
8. C3 + C4：把作用域写死（未测族清单 + regime 说明 + "选择性重算之所以有效，正因为能力不在缓存里" 这个推论）。
9. C2：在 §7 主张优先权（"independently corroborated after our freeze by CacheLink"）。
10. C6：说明 RAT 选择理由，或补 no-anchor 的 fraction 探针（run 已有，只需补数）。

### 第三档：把"审计"升级为"判决性实验"（唯一值得再花 GPU 时间的改动）

**用本文自己的仪器去测一个行为目标译码器。** 现在的局面是两种解释并存，而本文的仪器正好能分开它们：

| 解释 | 预测 |
|---|---|
| ① 教师缓存不携带学生可读的任务增益（本文倾向的读法） | 无论目标函数怎么换，CHG 都 ≤ 0 |
| ② 重建目标（MSE/ridge）选错了目标，行为目标能救（CacheLink/KV-Lingo/BridgeKV 的读法） | 以 receiver 预测散度为损失时，CHG > 0 |

做法很便宜：把现有 `apcs/mapper` 的训练目标从"重建学生 KV"换成"最小化 翻译缓存下学生 next-token
预测与原生 prefill 预测的 KL"，其余协议（恒等对照、零 re-prefill、gold 概率、10⁴ bootstrap）一律不动。
- 若 CHG 仍 ≤ 0：本文的负结果升级为"即使换成行为目标也不行"，是对领域更强的判决；
- 若 CHG > 0：本文从"负结果审计"变成"负结果 + 关键分辨实验"，直接解释了 CacheLink/KV-Lingo 为何有效，
  同时仍保持"重建保真≠能力"的方法学贡献。

无论哪个结果都可发表，且这是当前唯一能用本文资产回答的开放问题。建议作为 v2 的主打增量。

### 不需要做的
- 不需要再补 mapper 家族的容量消融（affine/MLP/joint MLP 已饱和）；
- 不需要追新的长上下文长度（4K/8K 已足够支撑"不是长度问题"）；
- 不需要重跑 177 个 run（B1/B3 是写作问题，B2 是报告问题，均不涉及重算）。

## 六、检索记录与局限

**已核验（本轮实测）**
- 10 条 arXiv 引用逐条 `id_list` 解析：全部存在。2 处标题与现行版本不符（B5，均为 v2 改名）。
- 4 条承重引用的摘要逐字比对：`heo2026`（73–98%、500×1024 FineWeb-Edu、+37pp MLP）、
  `cachebridge2026` v1/v2、`contextreuse2026`（27.59→34.48、+6.89pp）、`mot2026` v1/v2。
- 新工作检索（arXiv API，按提交日期倒序 × 4 组查询）：`LSC`、`MoT`、`CacheBridge`、
  `Context-Reuse`、`CacheLink`、`RaReCache`、`ACCT`、`LatentPort`、`KV-Lingo`、`BridgeKV`、
  `Hybrid-Attention Translation`、`Safety Incompatibility`。
- 仓库内复核：`scripts/aggregate_claims.py` 实跑（177/181/133/48、max translation +0.0101、
  max probe +0.0196、正侧排除零 0 行）；RAT 两 run 的 PPL/CHG/Acc；4K/8K 配置的模型对。

**未覆盖（本轮未做，需知晓）**
- OpenAlex 本轮返回 429（限流），未完成 DOI 侧的二次核验；7 条非 arXiv 条目（SIGCOMM/EuroSys/
  NeurIPS/EMNLP）沿用 audit9 的核验结论，本轮未重跑。
- 未检索非 arXiv 渠道（OpenReview 审稿、期刊版、厂商技术报告），也未做 Google Scholar 引用追踪
  ——论文自身尚未公开上传，暂无可追踪的引用。
- 中文版 `paper/zh/` 未做数字级同步核查（沿用 audit9 的 `tmp/zh_number_parity_check.py` 结论）。
