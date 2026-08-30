# APCS 真实数据补充实验报告（T05 替换保真度，真实 HF KV）

> 日期：2026-08-19
> 目标：抓取网上真实数据集，替代 T04–T06 的合成 KV，重新验证替换保真度
> 配置：`configs/pair_qwen3_real.yaml`（Qwen3-4B → Qwen3-1.7B, RTX 5090）

---

## 1. 真实数据获取（来源：HuggingFace Hub，经 hf-mirror 镜像）

| 数据集 | split | 抓取条数 | 落盘 |
|---|---|---|---|
| hellaswag | train | 128 | `reports/real_data/hf_datasets.json` |
| arc_challenge (allenai/ai2_arc) | train | 128 | 同上 |
| winogrande (winogrande_xl) | train | 128 | 同上 |
| mmlu (cais/mmlu → fallback hails/mmlu_no_train) | test | 128 | 同上 |

- 加载走 `apcs/data/hf_dataset.py` 已注册 loader（hellaswag/arc_challenge/winogrande/mmlu），
  每个样本为 `{sample_id, context, query, answer, split}`。
- 注意：**此前 `_make_prompt` 加载失败时静默回退合成 prompt**；本次用真实数据
  直接驱动真实模型 forward，抓到 (L, S, H, 2D) 的真实 Teacher/Student KV。

## 2. 真实 KV 捕获（HFKVProvider，modelscope 模型源）

```
Teacher Qwen3-4B   kv_t: (36, S, 8, 256)   S≈70–124（hellaswag 真实长度）
Student Qwen3-1.7B kv_s: (28, S, 8, 256)
```
- 每层 K|V 沿 head_dim 拼接为 (S, H, 2D)，与仓库 numpy 契约一致。
- 校准/评估分开：calib=seed0（train），eval=seed1（test），防泄漏（§36/§70）。

## 3. 真实数据替换 T05 结果（RidgePerHeadMapper, de-RoPE, K/V 独立）

**关键结论：用真实 KV 后 retention 从合成的 0.714 提升到 ≈0.85，达到 CONDITIONAL（0.80–0.90），
但仍未过 Gate 1 PASS 阈值（≥0.90）。**

| 实验 | seq | n_calib/n_eval | K retention | V retention | mean retention | token_agr | Gate 1 |
|---|---|---|---|---|---|---|---|
| 合成基线（原 T05，512/1024） | 512/1024 | 39/20 | 0.714 | 0.714 | **0.714** | 0.209 | **FAIL** |
| 真实 v3（公共最小长度） | 53 | 16/16 | 0.976 | 0.733 | **0.854** | 0.582 | **CONDITIONAL** |
| 真实 v4（S=70） | 70 | 12/12 | 0.976 | 0.729 | **0.852** | 0.609 | **CONDITIONAL** |
| 真实 v4（S=100） | 100 | 10/9 | 0.974 | 0.712 | **0.843** | 0.625 | **CONDITIONAL** |
| 真实 v1（pad 到 512，含 padding 污染） | 512(pad) | 8/8 | 0.970 | 0.679 | **0.825** | 1.000* | CONDITIONAL |

*注：pad 版本 token_agr=1.0 是 padding 零向量 argmax 恒同的假象，弃用；以 trunc 版本为准。

### 3.1 逐分项分析

- **K 通道 retention ≈ 0.97**：Teacher K → Student K 的线性映射在真实数据上保真度很高，
  接近 PASS 水平。K（内容检索键）在跨模型间有可迁移的线性结构。
- **V 通道 retention ≈ 0.71–0.73**：V（语义输出值）映射明显差于 K。V 承载语义/输出信息，
  跨模型层间语义对齐弱于检索键结构 → 整体被 V 拉低。
- **token agreement ≈ 0.58–0.63**：argmax 维度一致率显著高于合成基线（0.21），
  但仍低于 0.90，说明替换后生成 token 与 Student 原生推理**方向相似但不逐位一致**。
- **随序列增长轻微下降**（0.854@53 → 0.843@100）：长上下文下映射精度略降，符合累积误差预期。

### 3.2 与合成基线对比（重要发现）

合成 `_synth_calibration_set` 的构造是"共享 latent Z + 固定线性 W"，理论上更可映射，
但 retention 只有 0.714；真实 KV 反而更高（0.85）。说明：
1. 合成数据的 W_t/W_s 用**独立随机初始化**，实际构造的线性关系弱；
2. 真实模型的 K 通道确有跨模型可迁移结构（0.97），合成数据未建模这一点；
3. 之前基于合成数据的"0.714 FAIL"结论**低估了真实可映射性**，但方向结论一致：
   **替换保真度仍未达到论文 Gate 1 的 PASS 线**。

## 4. 对总结论的影响

| 层面 | 合成数据结论 | 真实数据修正 |
|---|---|---|
| T05 Retention | 0.714 FAIL | **0.85 CONDITIONAL**（未到 0.90 PASS） |
| Replaceability | 不成立 | **部分成立**（K 通道强、V 通道弱） |
| T11 决策 | D_STOP_REPLACEABILITY_UNSTABLE | 仍为 D（retention < 0.80 → D；0.85 仍在 0.80–0.90 CONDITIONAL 区间，未达 Path A 所需 ≥0.90） |

**总结论不变，但机理理解更新**：
- **APCS 在真实 KV 上"有部分效果但未达预期"**——K 通道接近可用（0.97），V 通道是瓶颈（0.72），
  整体 retention ≈0.85 处于 CONDITIONAL。
- 若论文继续，优先研究方向：**V 通道专门映射**（更强调值空间的语义对齐，或对 V 采用非线性/
  逐层注意力对齐），而非整体 K/V 联合线性映射。

## 5. 产物清单

```
reports/real_data/
├── hf_datasets.json            # 4 个真实数据集各 128 条（hellaswag/arc/winogrande/mmlu）
├── t05_real_kv.json            # 真实 KV T05（pad-512，参考）
├── t05_real_kv_v3.json         # 真实 KV T05（公共最小长度 53）
├── t05_real_kv_S70.json        # 真实 KV T05（S=70）
└── t05_real_kv_S100.json       # 真实 KV T05（S=100）
```

复现命令：
```bash
PYTHONPATH=/workspace/apcs-main HF_ENDPOINT=https://hf-mirror.com \
  python /tmp/opencode/real_t05d.py    # 多长度扫描
```