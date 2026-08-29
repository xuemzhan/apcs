# APCS Experiment Log

**Status**: ✅ 协议 v1.3 规模阶梯完成（2026-08-29 傍晚）——三假说终审可用
**历史数据（v1.0 及以前）**: ⚠️ UNVERIFIED，见文末历史节

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
