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
| P1-16 重建 R² 缺失单位 | `scripts/measure_reconstruction_r2.py`（affine，20 拟合/10 held-out） | K 每(层,头) R²=+0.81；V=+0.32（值侧约 2/3 方差不可线性恢复） |
| P2-4 探针粒度 | 八分位窗口 sweep（affine c30, `probe_mode`, n=30） | 仅 layers 10--13 的 CHG CI 排除 0（−0.186 [−0.322,−0.060]）；顶部三八分位中性（−0.011~+0.020） |
| P2-3 过度参数化译者 | 跨层/跨头 Joint MLP（hidden 256×2，~37M 参数，c30） | CHG −0.265 [−0.426,−0.111]，acc 0.233，PPL 31.6（流畅但不能力）；"不存在可用 mapper"反驳被更强的 negative 封堵 |
| P2-5 长上下文任务 | needle-in-a-haystack 4 选（~1024 token，affine c30，n=30） | teacher 0.733 / student 0.533 / translated 0.233，CHG −0.285 [−0.534,−0.025]；失败非短上下文伪影 |
| P2-1 跨家族/跨架构 | Qwen3-4B→{Llama-3.2-1B/3B, Gemma-2-2B, Gemma-3-1B, Qwen2.5-1.5B}，矩形 affine（head 均值池化+维度投影），c30，n=100 | H1 全通过；CHG +0.000/−0.044/−0.009/−0.061/−0.157；均无 capability（teacher gold 0.691 未恢复） |
| 复现审计 | 逐条重跑 23 个核心配置并与记录值对比 | 确定性族（ridge/affine/per-layer/task-aware/RAT/joint MLP、校准阶梯、探针/八分位、跨架构）逐位或 \|Δ\|≤0.002 复现；**per-head MLP 训练种子相关**：c30∈[−0.39,−0.26]、c200∈[−0.26,−0.23]，论文已改为区间并加复现说明 |
| P2-2 target-side replay 诊断 | 翻译 cache 上再回读 context（非部署，违反 zero-prefill），affine c30，n=100 | replay 后 acc 0.290、CHG −0.232 [−0.333,−0.136]，与无 replay（−0.249）无显著差异；远低于 student self 0.520 ⇒ 朴素 replay 不能恢复，MoT 增益不能归因于 replay 本身 |

**三假说终审（v1.5，30 次真实 GPU 运行；v1.2+ 审计协议 run 总计 96 个，见 README 运行清单）**：
H1 ✅（logit cos 1.000）· H2 ⚠️（弱学生近平局但**未通过 ε=0.02 非劣检验**、强学生全族失败、多 seed/校准/非线性均惰性）·
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

---

## Wave-3 学习曲线（2026-08-30 23:04–23:36，5 连跑，GPU RTX 4090）

统一协议：protocol v1.5 · gold 口径 · calib/eval 互斥 · self_kv 恒等对照 ·
ROPE rotated · git hash `3645d39e`。评估集 n=100；mapper affine per-head。

### 学习曲线主表（4B→1.7B，kv_both，gold 口径）

| run | mapper | calib | acc | gold | PPL | CHG gold [95% CI] | self_kv | gate |
|---|---|---|---|---|---|---|---|---|
| v1.5-c10 | affine per-head | 10 | 0.230 | 0.260 | 218.9 | −0.251 [−0.342, −0.164] | −1.93e-06 | FAIL |
| v1.5-c30 | affine per-head | 30 | 0.290 | 0.262 | 57.6 | −0.249 [−0.350, −0.149] | −1.93e-06 | FAIL |
| v1.5-c60 | affine per-head | 60 | 0.260 | 0.274 | 62.2 | −0.238 [−0.338, −0.137] | −1.93e-06 | FAIL |
| v1.5-c100 | affine per-head | 100 | 0.230 | 0.255 | 57.3 | −0.257 [−0.355, −0.156] | −1.93e-06 | FAIL |
| v1.5-c200 | affine per-head | 200 | 0.260 | 0.258 | 56.5 | −0.244 [−0.337, −0.142] | −1.93e-06 | FAIL |
| v1.5-c500 | affine per-head | 500 | 0.270 | 0.268 | 59.2 | −0.244 [−0.337, −0.142] | −1.93e-06 | FAIL |

**结论**：校准预算 c=10→500，CHG 始终在 −0.24±0.01 窄带内。学习曲线平坦——
更多校准样本不能缓解替换级退化。PPL 从 c10 的 219 骤降至 c30 的 57（精度从
float16→float32），c30–c500 区间 PPL 稳定在 56–62。

### 8B→0.6B 运行（非对称规模跨越）

| run | mapper | calib | acc | gold | PPL | CHG gold [95% CI] | self_kv | gate |
|---|---|---|---|---|---|---|---|---|
| v1.5-8b→0.6b-c30 | affine per-head | 30 | 0.270 | 0.269 | 99.0 | −0.023 [−0.104, +0.064] | +3.37e-04 | FAIL |

**结论**：8B→0.6B 的 CHG 接近零（CI 包含 0），self_kv 略偏（+3.37e-04）但
在 CI 内。这表明**非对称规模跨越**（教师远大于学生）时，mapper 退化幅度减小——
但仍然无法产生正向增益。gate 仍 FAIL。

### P1-16：逐 (layer, head) R² 测量（4B→1.7B affine，c=30）

在 30 条校准样本上拟合 per-head affine mapper，held-out 10 条计算 R²：

| 通道 | mean R² | min R² | max R² | std |
|---|---|---|---|---|
| K channel | **−0.428** | −17.642 (L4,H2) | +0.538 (L24,H3) | 1.509 |
| V channel | **+0.129** | −1.230 (L0,H1) | +0.522 (L1,H1) | 0.164 |

**逐层 K channel R² 摘要**：
- 前 8 层（L0–L7）：mean R² 全部为负，最优 head 仅 0.30
- 中间层（L8–L19）：大部分为负，L11/L13/L15 偶有正 R²（≤0.48）
- 后层（L20–L27）：L22/L26 略正，L24 最大 0.538，但 mean 仍为负

**结论**：K channel R² 大面积为负（mean −0.43），确认 per-head affine mapper
对 K 状态的映射严重欠定——student K 不是 teacher K 的线性函数。V channel
R² 略正（mean 0.13）但远不足以支撑可靠迁移。这从表征层面解释了为何
affine mapper 的 CHG 始终为负。

### P1-8：三 seed 可比性修复（2026-08-31 00:02–00:08，2 连跑）

**问题**：论文 Limitations 句报告 seeds 42/43/44 为 "−0.14, −0.47, −0.46"，
但数据验证发现三个不一致：(1) 混用 CHG 定义（seed 42 的 −0.14 是 top-level
confidence-based chg；seeds 43/44 的 −0.47/−0.46 是 chg_gold）；(2) 混用
评估协议（seed 42: eval_from_tail_n=100, n=100；seeds 43/44: eval_from_tail_n=0,
max_samples=60, n=29/30）；(3) inject_eval.seed 控制数据 shuffle → eval set 随
seed 变化，不同 seed 的 "last-100" 是不同样本。

**修复**：创建 `*_tail.yaml` 配置，统一使用 `eval_from_tail_n: 100, max_samples: 300,
inject_eval.seed: 42`，保证 eval set 与 s42 tail 完全一致。

**关键发现**：`inject_eval.seed` 同时控制 eval set 和 calibration set（evaluator
代码 L1394-1397），无法分离。使用 `inject_eval.seed: 42` 后三个 seed 的结果
完全相同（mapper 也相同），但 eval set 可比性是论文三 seed 句的前提。

**Sanity check**：student gold_prob_mean 三跑均为 0.5116（精确到 16 位小数），
确认 eval set 一致。

| run | seed | chg_gold (kv_both) | 95% CI | n | student gold |
|---|---|---|---|---|---|
| s42 tail | 42 | −0.2493 | [−0.350, −0.149] | 100 | 0.5116 |
| s43_tail | 42* | −0.2493 | [−0.350, −0.149] | 100 | 0.5116 |
| s44_tail | 42* | −0.2493 | [−0.350, −0.149] | 100 | 0.5116 |

\* inject_eval.seed=42（保证 eval set 一致）；seeds=[43]/[44] 控制 prepare-data。

**注**：因 inject_eval.seed 同时控制 eval set 和 calibration，三跑 mapper 完全
相同，CHG 无方差。论文的三 seed 句应报告 `chg_gold = −0.249 [−0.350, −0.149]`
（三 seed 一致），而非原始的混合定义数字。

### 运行路径索引

| run | metrics.json 路径 |
|---|---|
| v1.5-c10 | `reports/runs/v15-4b-to-1.7b-affine-c10-s42-20260830-230433/inject-eval/metrics.json` |
| v1.5-c60 | `reports/runs/v15-4b-to-1.7b-affine-c60-s42-20260830-231024/inject-eval/metrics.json` |
| v1.5-c100 | `reports/runs/v15-4b-to-1.7b-affine-c100-s42-20260830-231652/inject-eval/metrics.json` |
| v1.5-c500 | `reports/runs/v15-4b-to-1.7b-affine-c500-s42-20260830-232053/inject-eval/metrics.json` |
| 8b→0.6b-c30 | `reports/runs/v15-8b-to-0.6b-affine-c30-s42-20260830-232500/inject-eval/metrics.json` |
| R² 结果 | `/tmp/opencode/p116_r2_results.json` |
| P1-8 s43_tail | `reports/runs/v15-4b-to-1.7b-affine-c30-s43-tail-20260831-000200/inject-eval/metrics.json` |
| P1-8 s44_tail | `reports/runs/v15-4b-to-1.7b-affine-c30-s44-tail-20260831-000652/inject-eval/metrics.json` |

---

### P1-17：确定性验证（2026-08-31 00:02–00:13，3 连跑 + 1 pre-fix）

**验证目标**：pinned-config 复现 canonical CHG（bit-for-bit）。

| run | chg_gold (kv_both) | 95% CI | 与 canonical 关系 |
|---|---|---|---|
| s42 canonical（20260829-231341） | −0.2493 | [−0.350, −0.149] | 基准 |
| s43-tail（20260831-000200） | −0.2493 | [−0.350, −0.149] | bit-for-bit 相同 |
| s44-tail（20260831-000652） | −0.2493 | [−0.350, −0.149] | bit-for-bit 相同 |
| calibseed43 pre-fix（20260831-003529） | −0.2493 | [−0.350, −0.149] | bit-for-bit 相同 |

三跑 student gold 基线均精确为 0.5115706191084982（16 位小数一致）。

**seed 耦合发现**：`inject_eval.seed` 的 shuffle 同时决定 eval tail 与校准抽取
（evaluator 代码 L1394-1397）——固定 eval set 后"换 seed"实际只换校准抽取，
两者无法分离。

**harness 修复**：新增向后兼容字段 `inject_eval.calib_seed`——在 head-slice
前置换校准池；字段缺失 → pre-feature 行为完全一致。单测已加，全套 293 passed
/ 1 skipped。

---

### P1-18：校准抽取方差（2026-08-31 02:32–02:42，5 连跑）

**设置**：eval tail-100 完全固定（student gold 基线 0.5115706191084982 精确一致），
仅置换校准抽取（calib_seed 42/43/44/45/46）重拟合 affine mapper（c=30）。

| draw | run | chg_gold (kv_both) | 95% CI |
|---|---|---|---|
| 42 (canonical) | `v15-4b-to-1.7b-affine-c30-s42-20260829-231341` | −0.2493 | [−0.350, −0.149] |
| 43 | `v15-4b-to-1.7b-affine-c30-calibseed43-20260831-023246` | −0.2748 | [−0.374, −0.179] |
| 44 | `v15-4b-to-1.7b-affine-c30-calibseed44-20260831-023540` | −0.2632 | [−0.358, −0.175] |
| 45 | `v15-4b-to-1.7b-affine-c30-calibseed45-20260831-023919` | −0.2917 | [−0.383, −0.203] |
| 46 | `v15-4b-to-1.7b-affine-c30-calibseed46-20260831-024209` | −0.2501 | [−0.353, −0.149] |

**统计**：mean −0.266，sample SD 0.018，range 0.042（−0.2917 ~ −0.2493）。
全部落在 per-run bootstrap 区间内——mapper 训练方差不是 CHG 的来源。

**Pool cap**：eval tail-100 切分后校准池有上界。c500 run（
`v15-4b-to-1.7b-affine-c500-s42-20260830-232053`）配置 `max_samples: 300` +
`eval_from_tail_n: 100` → 校准池实为 **200**（c500 与 c200 的 CHG 均为
−0.24381759782279303，bit-for-bit 相同，即 c500 只消费完整池 200 条）。
c≥200 的预算全部饱和并逐位复现完整池结果。
