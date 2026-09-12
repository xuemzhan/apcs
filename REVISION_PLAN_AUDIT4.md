# 修改方案（依据 paper_audit4_1.md，6/10 Weak Accept → 目标 7/10）

审稿人结论：**不再需要补实验**，只需修统计表述与若干残留措辞。以下按"非 GPU / GPU"拆分。

---

## A. 非 GPU 方案（第一批，零 GPU）

### A1（P0，最重要）ε=0.02 的时间线统一

问题：正文/摘要 8 处称 ε=0.02 为 pre-specified / confirmatory（L58、L130、L365、L657、L932、L943、L1036…），
而附录 A 自己写明数值 ε 于 2026-09-12 才确定、晚于 8/30–31 的阶梯 run。属内部矛盾。

处置：
1. 证据分层：**confirmatory = gain gate（$CI_{lower}>0$，与 ε 无关，无配置通过）**；
   **secondary/reporting = replacement gate（$CI_{lower}>-\epsilon$，ε=0.02）**。
2. 删除 §3.3 "the replacement and gain gates at ε=0.02 ... were fixed before the runs"。
3. 其余 "pre-specified margin" → "the reporting margin ε=0.02"。
4. 新增 ε 敏感性表（ε=0.01 / 0.02 / 0.05 下各自通过的配置），由现有 CI 直接算出。

### A2 H2 措辞（未拒绝 $H_0$ ≠ 证明 inferiority）

- §5.2 标题 → "No Strong-Student Mapper Establishes Replacement; the Weak-Student Result Is Inconclusive"
- Conclusion → "No tested strong-student map establishes replacement under the audit gate."

### A3 Conclusion 计数错误

现写 "the one configuration that clears our replacement margin"，实际为两个近随机学生 / 三个配置。
改为不计数表述："the only configurations that clear the chosen margin correspond to two near-chance
students and show no measurable gain."

### A4 Figure 1 残留措辞

- caption "teacher's full prefill as the capability upper bound" → "teacher full-prefill reference"。
- † 标记拆分：† = gate **form** fixed before the audit runs；‡ = numerical margin selected later as a
  reporting tolerance。

### A5 Introduction 的 H1 定义统一

"a foreign cache is consumed without loss" → 与 Figure 1 / §5.1 统一为 **injection mechanics
(student-space cache)**。

### A6 H3 family-wise 的证据级别与算法

- 标注 "the family-wise max-bootstrap bound is a revision-added multiplicity analysis over the
  pre-specified confirmatory probe family"。
- 附录补 $T^{*(b)}=\max_j \bar\Delta_j^{*(b)}$，并说明所有配置共用同一组 bootstrap 索引。

### A7 §6 两处软化

- "a gap in these parameters, which a cache does not transport" → "which raw KV transfer does not
  explicitly transport"。
- MoT 的 "exactly the upper-layer correction" → "provides a target-side correction pathway that
  strict zero-reprefill disallows"。

### A8 附录补 Heo-style 实现差异表（回应实现保真度追问）

已核实流程（写入附录）：

- 校准 200 条（与 affine c200 行同预算）；`apcs/alignment/topk.py` 以 `fit_frac=0.7` 在 70% 上拟合
  per-(student layer, teacher layer) ridge、在剩余 30% 上算 held-out R²；
- 每个 student 层独立选 top-$k$（$k\in\{1,3,5\}$），候选源层为全部 36 层教师层；
- 选层后用**完整 200 条**重拟合 per-head ridge，$\lambda=10^{-3}$；评测为固定 tail-100；
- 待确认项：K/V 是否各自独立选层（读 `topk.py` 调用参数）。

### A9 gold 指标细节

声明 "we retain only four-choice examples"（HellaSwag / ARC / needle 均为四选一；写前核对加载器）。

### A10 可选：§6 叙事升级

把 "KV reconstruction ⇏ task preservation" 提升为与 "fluency ⇏ capability" 并列的第二层 decoupling，
§6 标题可改为 "Why Reconstruction Is Not Enough"。纯改写，不动数据。

### A11 精简 permutation 说明（不参与任何 gate）

---

## B. GPU 方案（第二批）

### B1（推荐，1 次 GPU）Heo-style mapper 的重建诊断

目的：解释 PPL 随 $k$ 爆炸（78.5 → 2629.6 → 68966.1），并回应"是方法本身失败还是实现/条件数问题"。

步骤：对 $k\in\{1,3,5\}$ 的三个 Heo-style run，按 affine/joint-MLP 相同口径报告
$R^2_K$、$R^2_V$（20 拟合 / 10 held-out context）。

- 若重建随 $k$ 变差 → pair / 几何问题；
- 若重建变好而 PPL 爆炸 → 与 joint-MLP 结论呼应：**更好的局部重建可能破坏 attention 几何**。

### B2（可选）8K 上下文扩展

审稿人明确说非必需（4K 已足够说明论文没有拿 n=20 的噪声强行讲故事）。

---

## C. 提交层（非实验）

- 双盲投稿用 `paper/cache_audit_iclr2026/`（匿名）或 `paper/cache_audit_anon/`；
  当前 19 页长文版首页含姓名/邮箱/CETC，只适合作 arXiv / camera-ready。
- 附录中的仓库信息在双盲期建议中性化。

## D. 执行顺序

1. 第一批：A1–A9（+A10/A11 可选），四份稿件同步重编验证。
2. 第二批：B1（1 次 GPU）→ 回填 §6 与附录。
3. 预期：按审稿人意见，完成 A 类后即"无明显结构性审稿漏洞"，评分可望 7/10。
