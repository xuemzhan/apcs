# APCS Experiment Log

**Status**: ✅ v1.5 闭环完成 + 论文定稿（2026-08-29 深夜）——三假说终审 + 两轮独立审稿
**Paper**: `paper/cache_audit/main.pdf`（11 页）
**历史数据（v1.0 及以前）**: ⚠️ UNVERIFIED，见文末历史节

---

## v1.5 审稿修复闭环（2026-08-29 19:44–20:23，9 连跑）

针对两轮独立审稿的 MAJOR/MODERATE 逐条修复并验证：

| 审稿发现 | 实验修复 | 结果 |
|---|---|---|
| A1 探针用最差翻译器 | 探针换 affine c30 重跑 | 单调劣化仍成立，H3 对翻译器质量鲁棒 |
| B1 multi-seed 缺失 | affine c30 × seeds {43,44} | CHG −0.47 / −0.46；seed 42（−0.14）是最佳情形 |
| 新增 MAJOR：无非线性测试 | 实现 MLP mapper | CHG −0.303 / −0.236，非线性不帮助 |
| A2 规模阶梯混淆评估集 | `eval_from_tail_n=100` 固定评估集 | c30/c200 CHG −0.249/−0.244（差 0.005），校准预算**惰性** |

**三假说终审（v1.5，30 次真实 GPU 运行）**：
H1 ✅（logit cos 1.000）· H2 ⚠️（弱学生平局、强学生全族失败、多 seed/校准/非线性均惰性）·
H3 ❌（最佳翻译器探针 + PPL/acc 解耦 + oracle 上界三重确认）。

---

## v1.4 RAT 与 Oracle 探针（2026-08-29 17:45–18:10，6 连跑）

**新增**：RAT（残差锚定翻译器：embedding 对齐核 + 解析头映射 + 闭式低秩修正
+ 匈牙利头匹配 + sink 特判）· Oracle 探针模式（mix_α / win_*，probe_mode 门控）。

### RAT 主结果（4B→1.7B，gold 口径）

| run | 配置 | acc | gold | PPL |
|---|---|---|---|---|
| RAT c30 | anchor+sink+match, rank16 | 0.233 | 0.235 | 502 |
| RAT c30 nosink | 去掉 sink 覆盖 | 0.233 | 0.241 | 271,207 |
| RAT c30 noanchor | 去掉解析核, rank32 | 0.267 | 0.280 | **23.7** |
| RAT c200 | 全配置 | 0.238 | 0.276 | 1008 |
| RAT c200 (0.6B) | 全配置 | 0.206 | 0.232 | 790 |
| （对照）affine c30 | v1.2 | 0.367 | 0.365 | 46 |
| （上限）self_kv | 恒等注入 | 0.500 | 0.503 | 21.2 |

组件结论：(a) **sink 覆盖对 PPL 至关重要**（271k→502：position 0 的翻译
KV 是灾难源）；(b) **解析核有害**（去掉后 PPL 502→23.7）：min-norm pullback
假设在真实模型上不成立（V_t 的零空间分量占主导，embedding 几何对齐
不足以为其提供方向）；(c) 即便 PPL 修复到 ≈自 prefill 水平，**accuracy
仍 0.267 << 0.500** —— 重建质量与答案能力完全脱钩。

### Oracle 探针（H3 上界判定——本轮决定性实验）

| 探针 | acc | gold | 解读 |
|---|---|---|---|
| self_kv（α=0 基底） | 0.500 | 0.5031 | 学生自身 |
| mix_a25（25% 教师成分） | 0.367 | 0.3798 | 已受损 |
| mix_a50 / mix_a75 | 0.333 / 0.267 | 0.279 / 0.249 | **随 α 单调劣化** |
| win_high（顶层 1/3 换教师） | 0.500 | 0.5022 | 无害（顶层非答案通路） |
| win_mid / win_low | 0.300 / 0.267 | 0.265 / 0.224 | 有害 |

### H3 终审（原理性否定）

**在最有利条件下（75% 学生自身 KV + 仅 25% 教师成分；或教师成分仅占
无害层），教师 KV 的任何注入都不产生正收益** —— 教师缓存中不存在
学生可读的、超出学生自身处理所得的答案相关信息。教师优势在其权重
（FFN 电路/注意力回路），不在缓存状态。结合 PPL 23.7 但 acc 0.267 的
脱钩现象：**cache 翻译可以恢复"流畅读上下文"的能力，但恢复不了
"教师级的上下文理解"** —— 这为 cache-translation 研究路线（MoT/C2C/
LSC/KVComm 等）提供了首个带恒等对照与 oracle 上界的系统性审计证据。

### 三假说最终判定（论文核心结论）

| 假说 | 判定 | 证据链 |
|---|---|---|
| H1 机制无损 | ✅ | self_kv ≡ student（logit cos=1.0，21 跑稳定） |
| H2 替换级 | ⚠️ 仅弱学生 | 0.6B: +0.010 ns；1.7B: 全 mapper 族 −0.14~−0.30 |
| H3 能力迁移 | ❌ **原理性否定** | oracle 探针单调劣化 + PPL/acc 脱钩 + 全 mapper 族失败 |

### 论文定位（更新）

主叙事：**"Cache Translation Across Heterogeneous LLMs: An Audit with
Identity Controls and Oracle Bounds"** —— 对 cache-translation 路线的
方法论审计：(1) 恒等对照证明机制可解；(2) 规模阶梯证明线性族天花板；
(3) RAT 证明架构锚定不解决信息缺口；(4) oracle 探针给出可利用性上界
为零；(5) 弱学生替换级 + win_high 无害层的实用发现。

---

## v1.3 规模阶梯终审（2026-08-29 15:27–16:04，7 连跑，4B→1.7B 除注明外）

统一协议：统一 prompt 格式 · calib/eval 互斥切分 · self_kv 恒等对照 ·
native 尺度重标定 · summary 文本基线 · gold 口径主指标 · git hash 落盘。
评估集 n=63（hellaswag+arc 轮转、shuffle 后切分）；self_kv 全程 ≡ student
（gold 0.4965 vs 0.4967）—— **H1 在所有运行中稳定通过**。

### 规模阶梯主表（kv_both，gold 口径）

| run | mapper | calib | acc | gold | PPL | CHG gold [95% CI] |
|---|---|---|---|---|---|---|
| v12-c30 | affine per-head | 30 | 0.367 | 0.365 | 46 | −0.138 [−0.317, +0.029] |
| A | affine per-head | 200 | 0.254 | 0.258 | 56 | −0.239 [−0.354, −0.134] |
| E | affine λ=1e-2 | 200 | 0.254 | 0.257 | 56.5 | −0.240 |
| F | affine λ=1e-1 | 200 | 0.254 | 0.256 | 56.4 | −0.241 |
| B | affine_layer（少 8× 参数） | 200 | 0.206 | 0.222 | 8.1e4 | −0.275 |
| G | task_aware diag+α (β=0.3) | 200 | 0.270 | 0.280 | 142 | −0.216 |
| D | affine **4B→0.6B** | 200 | 0.302 | 0.283 | 72 | **+0.010** [−0.081, +0.102] |

参考：student 自 prefill acc 0.508 / gold 0.497；teacher 0.762 / 0.675；
summary 文本基线 0.413 / 0.391。

### 综合分析（终审）

1. **正则不是杠杆**（E≈F≈A）：λ 变 100 倍结果不变 —— 数据充足后解由
   最小二乘主导，正则无法挽回。
2. **per-head 结构必要**（B 大幅劣化）：头维不可合并，各头几何差异真实存在。
3. **规模非单调（核心发现）**：c30(0.365) > c200(0.258)。数据×6.7 后更差
   ⇒ per-head 线性假设在数据充足时收敛到"最优线性近似"，反而劣于 c30 时
   被 λ 收缩的准均值预测器（教科书式偏差-方差权衡）。**线性 per-head
   mapper 家族的天花板 ≈ gold 0.26–0.37，且已被双边夹逼确认。**
4. **任务目标在同等规模下优于重建**（G 0.270 > 同规模重建 ~0.21-0.25，
   与 v12 的方向一致），但距 self_kv（0.497）仍远，PPL 反而恶化。
5. **H3 终审（当前规模上限内）**：冻结学生无法通过任何已测 mapper
   （线性 per-head / 逐层 / 任务目标 ≤200 校准）利用教师特征超过自身水平。
   1.7B CHG 全负；0.6B CHG +0.010（ns，≈持平）。
6. **0.6B 替换级成立**：D 的 kv_both(0.302) ≈ student(0.286) ≈ self_kv
   （CHG +0.010 ns）—— 对弱学生，"免 re-prefill 的替换"已达到统计持平。

### 三假说终审

| 假说 | 判定 | 证据 |
|---|---|---|
| H1 机制无损 | ✅ 通过（全部运行稳定） | self_kv ≡ student，logit cos=1.0 |
| H2 线性映射达替换级 | ⚠️ 部分：0.6B 持平；1.7B 未达 | 天花板 0.26–0.37 vs 0.497 |
| H3 能力迁移（CHG>0） | ❌ 未支持（≤200 校准 + 线性/任务目标） | 全部 CHG ≤ +0.01 且 CI 含 0 |

### 核心目标判定

**未达到**（"学生具备教师能力"不成立）。已确立的等价结论：
- 工程上：KV 持久化→mapper→冷启动注入链路无损、可部署（H1+在线逐位一致）；
- 弱学生（0.6B）：替换级（不劣于自身 prefill）达到；
- 强学生（1.7B）：注入反而有害（−0.22~−0.30）——教师 KV 对已较强的学生
  是干扰源。

### 若继续冲 H3 的剩余路径（按第一性原理排序）

1. **非线性 mapper**（per-head 小 MLP，rank-MLP）——线性假设已被双边夹逼
   证伪，这是最直接的下一假设；需要 ≥500 校准样本（design §32 上限）；
2. **教师侧任务条件化 prefill**（P1.4 已实现未扫参）+ 任务目标联合训练；
3. **可利用性上界探针**（先于 mapper 训练）：直接把**学生自 KV 与教师 KV
   的逐层插值**注入（oracle 混合），若学生读不出任何教师成分带来的增益，
   则 H3 被 principled 否定，停止 mapper 研究，转向 replacement 叙事；
4. **学生侧轻量适配**（LoRA on attention out-proj，违背"冻结"前提但回答
   "差距在 mapper 还是学生"）。

---

## 协议 v1.2 三假说分解验证（2026-08-29，RTX 4090，HellaSwag+ARC 30/30 互斥切分）

**协议变更**：self_kv 恒等对照 · 统一 prompt 格式 · KV 范数诊断 · native 尺度
重标定 · calib_shuffle · Affine/低秩/共享基 mapper 接入 · 混合目标 task-aware
（diag 增量+逐层 α）· teacher-summary 基线 · matmul 提速（einsum→批量乘 ~67×）。

### 三假说判定结果

| 假说 | 实验 | 结果 | 判定 |
|---|---|---|---|
| **H1 机制无损** | self_kv 恒等注入 vs student_self | gold 0.5031 vs 0.5030，**逐样本 logit 余弦 = 1.000** | ✅ **通过** —— zero-prefill 注入通路数学无损 |
| **H2 映射质量** | ridge/affine kv_both vs self_kv | ridge: 0.207（CHG −0.296）；**affine: 0.365（CHG −0.138）**；PPL 46~88 vs self_kv 21 | ❌ 未达标 —— mapper 是全部损失来源 |
| **H3 可利用性** | task-aware 混合目标（30 校准样本） | acc 0.133~0.233，PPL 恶化至 18.7k —— 小样本过拟合 | ⏸ **不可判定**（需 H2 先达标；数据规模是前置条件） |

### 关键新证据

1. **KV 范数诊断**：教师/学生范数比 **K=1.58、V=7.44** —— 教师V 幅值是学生的 7 倍多；尺度重标定后 native 仍灾难（PPL 5.4e5，acc 0.167）⇒ **语义/结构失配为主，尺度为次**。
2. **中心化（bias）收益巨大**：per-head ridge → affine 使 kv_both acc 0.20→0.367、PPL 88→46、CHG −0.296→−0.138（接近 CI 含 0）。证实 V 均值/尺度错位是 mapper 损失的一阶项。
3. **通道解耦（重要）**：v_only acc 0.367 > k_only 0.30 > kv_both 0.20 —— kv_both 的 PPL 最好（88）但 acc 最差；mapped K 在改善流畅度的同时**破坏答案相关的注意力路由**。
4. **文本基线**：teacher-summary acc 0.367 < student 自 prefill 0.500 —— 文本压缩路线有信息损失上限，KV 路线天花板（=self_kv 0.50）更高，方法动机成立。
5. **PSR**：einsum→matmul 后 **−13 → −3.4**（~4×）；仍为负因含测量专用 PPL forward 与 cache 重建，部署形态（离线预变换+直接持久化学生KV）可转正。
6. **task-aware 过拟合**（30 样本 × 5.7 万 diag 参数 + β_task=0.7）：PPL 88→18.7k。按功效分析，需校准 100–500 样本（design §32 原目标）后重测 —— H3 的判定被数据规模阻塞，而非被原理否定。

### Gate 状态

- Gate A（工程）：✅ self_kv 恒等 + 持久化在线逐位一致
- Gate B（能力）：❌ 最好 0.365 vs 学生 0.503（affine）——差距明确归因于 mapper 泛化
- Gate C（系统）：❌ PSR −3.4（部署形态可解）
- **下一步（唯一关键路径）**：校准样本 30 → 100/200/500 阶梯 × affine(±unrotated) × task-aware(β 扫描) —— H2 达标后 H3 才可判定

---

## 协议 v1.1 首轮验证（2026-08-29 上午）

**协议变更**：同分布 calib/eval 互斥切分（20/20）· gold 概率/accuracy 主口径 ·
native 基线 · KV 持久化 + 在线阶段 · unrotated RoPE 对照 · cache 污染修复 ·
git hash 落盘（git f3051e1d + 工作树修改）。

**数据**：HellaSwag（test 侧 20 样本评估），teacher Qwen3-4B，λ=1e-3，
proportional 层映射，de_rope_k=true。

### 主结果（gold 概率口径 CHG = handoff − student，95% bootstrap CI）

| 模型对 | RoPE | handoff acc | CHG(gold) | 95% CI | p | TGRR | Gate B |
|---|---|---|---|---|---|---|---|
| 4B→1.7B | rotated | 0.25 | **−0.096** | [−0.307, +0.136] | 0.79 | −0.46 | ❌ FAIL |
| 4B→1.7B | unrotated | 0.25 | **−0.165** | [−0.388, +0.065] | 0.91 | −0.80 | ❌ FAIL |
| 4B→0.6B | rotated | 0.15 | **+0.010** | [−0.127, +0.158] | 0.41 | +0.025 | ❌ FAIL |
| 4B→0.6B | unrotated | 0.20 | **+0.013** | [−0.120, +0.153] | 0.42 | +0.031 | ❌ FAIL |
| 4B→1.7B | native（无参数） | 0.35 | −0.082 | [−0.324, +0.147] | 0.74 | −0.40 | ❌ FAIL |
| 4B→0.6B | native（无参数） | 0.20 | +0.029 | [−0.105, +0.194] | 0.38 | +0.070 | ❌ FAIL |

参考基线：teacher acc 0.70（gold 0.628）；student self-prefill acc 0.40/0.10
（gold 0.421/0.209）；PPL：student ≈9.5，kv_both 142–227，**native 注入
9.9e4–9.9e5（PPL 灾难被真实测得，证实早前外部审查的判断）**。

### 结论（诚实版）

1. **旧"CHG>0"不可复现**：在正确指标（gold/accuracy）+ 互斥校准 + 干净
   cache 下，kv_both 对 1.7B 学生为负（不显著），对 0.6B 学生≈0（不显著）。
   8/29 早晨 sweep 的正 CHG 由三个伪象叠加：置信度指标、同样本校准、
   cache 污染（见下）。
2. **置信度指标的欺骗性被直接演示**：同一批运行里 handoff 置信度
   （0.68–0.81）普遍高于 student（0.56–0.72），而 accuracy 更低——
   "更自信≠更正确"，主指标必须用 gold/accuracy。
3. **发现并修复 cache 污染 bug**：transformers 4.52 在 use_cache=False 且
   传入 cache 时仍原地 update —— 旧代码 compute_ppl 与评分共用 cache，
   suffix 被双重 attend，旧 ridge 分数全部受污染。已改为独立 cache。
4. **RoPE 对照**：unrotated（§23 本意）未挽救 1.7B（−0.165 vs −0.096），
   0.6B 两者≈持平；在此协议/规模下 RoPE 对齐方式不是主要瓶颈。
5. **PSR 为负（−13 ~ −15）**：CPU numpy 逐头变换是瓶颈 —— 当前实现的
   系统收益为净损失；GPU/batch 化 mapper 或预变换落盘（离线算好存
   student KV）是下一步。
6. **B1 持久化链路验证通过**：离线落盘（365MB store，sha256 manifest）→
   在线仅加载 Student、mapper 参数从盘恢复（mapper_source=offline_store）、
   结果与离线**逐位一致**。"持久化→迁移→冷启动"工程闭环成立。
7. **多数据集声明与执行一致化**：`_load_sample_rows` 改为轮转采样，
   不再静默只取第一个数据集。

### 下一步（按 Gate 顺序）

- Gate A 已过（zero-prefill 真实审计 + 持久化冷启动）。
- Gate B 未过：需提高校准量（20→100+）、多 seed、更长上下文与用户域
  数据，并试验 task-aware mapper（以教师答案为目标）与部分层注入
  （1.7B 学生的负 CHG 提示"全量注入破坏学生自有表征"假说）。
- Gate C：mapper GPU 化/离线预变换后重测 PSR。

---

## ⚠️ 历史数据（v1.0 及以前，全部 UNVERIFIED）

**DO NOT USE ANY NUMBERS FROM THIS SECTION** until independently verified.

Two independent experiments on the same model pair produce contradictory results. The repository lacks proper provenance documentation.

---

## Experimental Data

### Source 1: verification_results.json (21:12)
| Method | CHG | Status |
|--------|-----|--------|
| Ridge | +0.2745 | ⚠️ UNVERIFIED |
| Native | +0.1061 | ⚠️ UNVERIFIED |
| K-only | +0.0828 | ⚠️ UNVERIFIED |
| V-only | +0.1736 | ⚠️ UNVERIFIED |

### Source 2: comprehensive_results.json (22:11)
| Method | CHG | p-value | Status |
|--------|-----|---------|--------|
| Native | +0.2295 | 0.0004 | ⚠️ UNVERIFIED |
| V-only | +0.1754 | 0.0039 | ⚠️ UNVERIFIED |
| Ridge | +0.1348 | 0.0192 | ⚠️ UNVERIFIED |
| K-only | -0.0287 | 0.7422 | ⚠️ UNVERIFIED |

---

## Critical Issues Identified

1. **Contradictory Rankings**: Ridge vs Native ranking reverses between experiments
2. **No Prompt Data**: Prompt files not saved in repository
3. **Simulated Multi-seed**: Previous multi-seed data was fake (std=0.0)
4. **Protocol Unclear**: No documentation of experimental protocol
5. **PPL Catastrophe**: Direct injection causes PPL≈2.2M (not reported)

---

## Recommendations

1. **Do NOT submit** current paper version
2. **Run fresh experiments** with full logging
3. **Save all prompts** to files
4. **Document protocol** explicitly
5. **Verify reproducibility** on different machine

---

## Files to Review

- `CRITICAL_REVIEW_RESPONSE.md` - Detailed analysis of reviewer concerns
- `DATA_PROVENANCE.md` - Data tracking document
- `paper/arxiv/main.tex` - Updated paper (removes unverified claims)

---

## Next Steps

1. Create proper experimental protocol
2. Run experiments with full logging
3. Save all intermediate results
4. Verify reproducibility
5. Only then update paper with verified numbers
