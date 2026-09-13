# APCS 论文创新点与技术关键

## 论文定位

**审计论文（Audit Paper）**：不是提出新的 cache translation 方法，而是对现有方法（MoT、C2C、LSC、KVComm 等）的**核心声称**进行系统性检验。

---

## 核心创新点

### 1. 三假说分离框架（Three-Hypothesis Audit）

**问题**：现有评估将三个可独立失败的问题混为一谈

| 假说 | 问什么 | 为什么重要 |
|---|---|---|
| **H1 机制无损** | 注入管道是否正确消费了外来 cache？ | 机制失败 ≠ 映射失败 |
| **H2 替换级** | 映射器是否达到替换级（不低于学生自身）？ | 核心能力迁移声称 |
| **H3 可利用性** | 教师 cache 是否包含学生可读的优势？ | 决定性上限 |

**创新**：每个假说有**独立仪器**和**门控条件**，失败可归因。

---

### 2. 恒等对照（Identity Control）

**方法**：将学生**自己的 cache** 通过完整的翻译-注入管道

**作用**：
- 验证机制正确性（logit cosine = 1.000）
- 隔离机制损失与映射损失
- 确认 zero-prefill 计数器准确

**发现**：H1 通过（CHG = +0.0001），证明机制不是失败模式。

---

### 3. Oracle 探针（Oracle Probes）

**方法**：不训练任何翻译器，直接混合教师/学生 content

**两种探针**：
- **分数探针**：α·学生 + (1-α)·教师，α ∈ {0.25, 0.5, 0.75}
- **窗口探针**：仅在 bottom/mid/top 三层替换

**关键发现**：
- 25% 教师 content 即导致 accuracy 从 0.500 降至 0.367
- 下降在 α 上**单调**
- 仅 top 三层替换无害（但也不提升）
- **结论**：可利用性上限为零

---

### 4. PPL-Accuracy 解耦（Perplexity-Accuracy Decoupling）

**发现**：
- 一个翻译器可恢复近原生 PPL（23.7 vs 21.2）
- 但 accuracy 仍为 0.267 vs 0.500
- **PPL 不能作为翻译质量的证据**

**意义**：挑战了 serving 文献中用 PPL 评估 cache 翻译的惯例。

---

### 5. 架构级失败分析（Architectural Analysis）

**数学推导**：
```
V = W_V · h （cache 是残差流状态的线性投影）

问题1：π(W_V)V 仅返回 h 在 W_V 行空间上的分量
       正交分量（2560-dim 中的大部分）对 cache 不可见

问题2：学生 V_S = W_V^S · R · h 依赖不可见分量
       → V_S 不是 V_T 的函数
       → 映射任务本质欠定
```

**结论**：教师优势在其**参数**（FFN 电路），不在 cache。

---

### 6. 审计协议即 Checklist

**打包交付**：未来任何 cache-translation 声称可在**一次评估**中验证

**必备三仪器**：
1. 恒等对照
2. 带区间 gold-probability 指标
3. Oracle 探针

---

## 技术关键

### 关键 1：Mapper Ladder（六族映射器对比）

| 映射器 | 核心思想 | CHG |
|---|---|---|
| Ridge per-head | 128×128 per-head 闭式解 | -0.296 |
| **Affine per-head** | +中心化+截距 | **-0.138**（最佳）|
| Affine per-layer | 参数少8× | -0.275 |
| Task-aware | 任务损失混合 | -0.216 |
| RAT | 架构锚定（W_V 伪逆） | -0.221 |
| MLP | 非线性 | -0.303 |

**发现**：中心化是单一最大收益；非线性无帮助。

### 关键 2：Calibration Budget 惰性

- c=30: CHG = -0.249
- c=200: CHG = -0.244
- **差值 0.005** → 映射器已收敛，更多数据无法关闭差距

### 关键 3：Channel Asymmetry

- Value-only: accuracy 0.367, PPL 88
- Key-only: accuracy 0.300, PPL 253
- Both: accuracy 0.200, PPL 1.8×10⁷

**发现**：映射 keys 修复 fluency 但损坏 routing。

### 关键 4：弱学生 vs 强学生

| 学生 | Retention | 含义 |
|---|---|---|
| 1.7B（强） | 全族失败 | 教师优势无法迁移 |
| **0.6B（弱）** | **+0.010（平局）** | 弱学生可达替换级 |

**部署含义**：cache translation 仅对弱学生有价值。

---

## 与现有工作的关系

| 工作 | APCS 审计发现 |
|---|---|
| MoT | 其 reported quality 含 target-side replay（非 strict zero re-prefill）|
| C2C | 假设线性映射足够 → 被 H3 否定 |
| LSC | 假设单一 latent space 适配所有层 → 被窗口探针否定 |
| KVComm | 假设 ranked layers 携带信号 → 被 oracle 探针否定 |

---

## 论文贡献总结

1. **框架贡献**：三假说审计框架 + 恒等对照 + Oracle 探针
2. **实证贡献**：6族映射器 × 校准阶梯 × 42次审计运行
3. **理论贡献**：架构级失败分析（V 不是 h 的可逆函数）
4. **方法论贡献**：PPL 不能作为翻译质量证据
5. **工具贡献**：可复用的审计协议 checklist

---

## 核心结论

> **教师的能力在其参数（FFN 电路），不在 cache。Cache 不携带能力，因此 cache translation 在 frozen student + zero re-prefill 下本质上无法迁移能力。**
