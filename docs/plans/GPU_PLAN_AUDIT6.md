# GPU 实验方案（依据 paper_audit6_1.md）

第六轮审稿的结论：文稿层面只剩**措辞与引用**问题（零 GPU，另行处理），
真正需要 GPU 的只有一件事——把 Heo-style 实验从"设计元素移植"升级为"该设计下的严格检验"。

---

## G1（唯一必做）Heo 设计下的严格检验：concatenation + 参考校准口径

**背景（审稿人核实）**：参考工作把选中的 top-$k$ 源层 KV **拼接（concatenate）**后拟合 ridge，
并用 **500 条 FineWeb-Edu、每条 1,024 token** 做校准；我们当前实现是**取平均**、用 **200 条审计上下文**校准。
因此现结果只能称 "Heo-inspired ablation"，不能称复现。本轮把它做到位。

**步骤**

1. 把源层组合从 average 改为 **concatenate**：`apcs/alignment/topk.py` 选出 top-$k$ 后，
   在 `apcs/mapper/`（ridge 路径）里把选中层的 KV 在特征维拼接再拟合；维度随之变为 $k\times D_t$，
   需相应调整 ridge 的正则矩阵尺寸。
2. 校准语料对齐参考口径：新增 500 条 FineWeb-Edu 风格、每条 1,024 token 的校准集
   （若离线不可得，用 `hf_dataset` 的等价长序列配置，并在附录如实注明语料来源与差异）。
3. 其余保持审计协议不变：Qwen3-4B→1.7B、固定 tail-100 评测、`self_kv` 恒等对照、
   `de_rope_k: true`、λ=1e-3、$k\in\{1,3,5\}$。
4. 每个 run 必须保留 `capability_score_artifact.json`（逐样本分数）与选层结果
   （建议落盘 `layer_mapping.json`，上一轮已发现该文件缺失）。
5. 已知问题：当前只有 `v2-smoke-20260913-012705`（n=30, CHG −0.2696 [−0.506,−0.019], PPL 6966），
   属烟雾测试，不可引用；需补齐三个正式 run。

**判据与写作分支**

| 结果 | 论文处理 |
|---|---|
| 仍显著为负（CI 上界 < 0） | §2.2/§5.2 可把措辞升级为"**在该设计下同样失败**"，附录表注明本轮已对齐 concatenation 与校准口径 |
| 接近 0 或转正 | 说明此前失败部分来自我们的平均式实现：把 §6 的 "in our averaged variant" 限定扩展为一段"组合方式敏感性"讨论，并保留"其设计仍不足以带来 student-readable gain"的结论（若 CI 含 0） |
| PPL/重建仍崩塌 | 报告该实现下的 $R^2_K/R^2_V$（沿用 B1 口径），与 k 的关系一并给出 |

**成本**：代码改动约半天 + 3 次 GPU run（校准集若需长序列生成，再加 1 次准备运行）。

---

## G2（可选）8K 上下文

第五轮审稿明确说非必需；仅当目标 venue 特别看重长上下文时再补（1–2 次 run，显存为主要风险）。

---

## 与文稿的衔接

- 拿到 G1 结果后，我会更新：§2.2 Heo 段、Table 2 三行（或新增行）、附录实现差异表（把 concatenation 一列从 "ours: average" 改为已对齐）、§6 组合方式讨论、以及摘要中相应的半句。
- 在 G1 结果到位之前，文稿保持第六轮建议的保守定位：**"Heo-inspired top-k/de-RoPE ablation under our audit regime"**。
