# APCS 实验结论记录（CONCL 2026-08-19）

> 本文件固化此前两轮实验的最终结论，作为后续优化工作的基线。

## 1. 主实验结论（真实轨，合成 KV 数据）

运行：`reports/runs/qwen3-4b-to-1.7b-real-20260819-000742/`（Qwen3-4B → Qwen3-1.7B，RTX 5090）

- T00–T04 工程正确性全 PASS；**T05 Gate 1 FAIL**（合成 retention=0.714 < 0.80）
- T09 Gate 2A "PASS" 但官方标注 `[SIMULATED]`（得分源为合成，§75 禁用作证据）
- T11 = `D_STOP_REPLACEABILITY_UNSTABLE`；T13 gate2a_rate=0.0
- 主结论：**合成 KV 上 APCS 线性/低秩映射无有效果，Replaceability 不成立**

## 2. 真实数据补充实验结论（HF 真实 KV，本轮获取）

数据源：HuggingFace（hf-mirror），hellaswag/arc_challenge/winogrande/mmlu 各 128 条
真实 KV：HFKVProvider 真实 forward 捕获 (L,S,H,2D)，替代合成校准集

| 数据 | mean retention | K | V | Gate 1 |
|---|---|---|---|---|
| 合成（原 T05） | 0.714 | 0.714 | 0.714 | FAIL |
| **真实 KV** | **≈0.85** | **≈0.98** | **≈0.72** | **CONDITIONAL** |

- **K 通道保真度 ≈0.98（接近 PASS）**：Teacher K → Student K 存在可迁移线性结构
- **V 通道保真度 ≈0.72（瓶颈）**：值空间跨模型语义对齐弱，拉低整体
- token agreement ≈0.58–0.63（远高于合成 0.21，仍 < 0.90）
- 随序列增长轻微下降（0.854@53 → 0.843@100）
- **结论修正：真实 KV 上 APCS"有部分效果但未达预期"**——从 FAIL 升到 CONDITIONAL，
  但整体仍未过 Gate 1 PASS（≥0.90），T11 仍为 D。

## 3. 待优化方向（基于结论）

1. **V 通道映射是核心瓶颈**（0.72 vs K 的 0.98）。优先专攻 V 通道。
2. **V 的 de-RoPE 存疑**：RoPE 只旋转 Q/K，V 不做旋转，当前代码对 K/V 无差别应用
   de-RoPE，可能损害 V 通道 → 需验证"V 跳过 de-RoPE"是否提升 V retention。
3. 校准样本量与 λ 调参：真实 KV 下 12–16 个校准样本已到 0.85，增加样本 + 调 λ 可能冲 0.90。
4. 目标：整体 retention ≥ 0.90（Gate 1 PASS），为后续能力迁移提供前提。

## 4. 产物位置

```
reports/real_data/hf_datasets.json             # 4 真实数据集各 128 条
reports/real_data/t05_real_kv*.json            # 真实 KV T05 多 seq 结果
reports/REAL_RUN_ANALYSIS_20260819.md          # 主实验分析
reports/REAL_DATA_ANALYSIS_20260819.md         # 真实数据补充实验分析
```

## 5. opt1 轮 GPU 运行结果（V 跳过 de-RoPE 验证，2026-08-19 07:15）

产物：`reports/real_data/opt1_v_nodrope.json`（RTX 5090，真实 hellaswag KV，seq=70, calib=12, eval=12）

受控 A/B 校验：`V_de_rope_on=0.72855`、`K=0.97609`、`mean_current=0.85232` 与 S70 基线
（`t05_real_kv_S70.json`）**逐位一致** —— 同一数据切分下仅翻转 V 的 de-RoPE 开关。

| 通道 | de-RoPE ON（现状） | de-RoPE OFF（优化） | Δ |
|---|---|---|---|
| K retention | 0.9761 | 0.9761（不变） | — |
| V retention | 0.7286 | **0.7581** | **+0.0296**（+4.1% 相对） |
| V token_agr | 0.3333 | 0.3438 | +0.010 |
| **mean retention** | **0.8523** | **0.8671** | **+0.015** |
| Gate 1 | CONDITIONAL | CONDITIONAL | 不变 |

结论：

1. **假设方向被验证**：RoPE 只旋转 Q/K，V 从不旋转；对 V 施加 de-RoPE 是数学上错误的
   反向旋转。去掉后 V retention +0.03，方向与理论一致 —— 这是真实存在的建模瑕疵，修正正确。
2. **但增益量级不足**：mean 0.852→0.867（+0.015），Gate 1 仍 CONDITIONAL，T11 仍 D。
   opt1 只是"去掉了一个错误"，未建立 V 的跨模型映射能力。
3. **V 瓶颈本质不是 de-RoPE**：修正后 V=0.758 vs K=0.976 差距 0.22 依然巨大，与主实验
   R²≈0.003 自洽 —— 线性 Ridge 对 V 值空间的跨模型结构解释力≈0。
4. **PASS 缺口**：K=0.976 固定时 mean ≥ 0.90 需 **V ≥ 0.824**，当前 0.758，
   还差 +0.066（约 8.7% 相对），是本次所得（+0.03）的两倍多。
5. **统计局限**：单一切分、无 bootstrap CI，+0.03 不可当证据引用。

## 6. opt1 轮代码审查结论（对照预期方向）

审查对象：工作区未提交 diff（23 文件，+1317/−261，commit b45c4fd 之后）。
验证：60 项相关测试全部通过；LSP error 全部为既有问题，无新增；de-RoPE 开关在
fit/transform 两侧一致；K|V 布局在 capture 与 split 两侧一致。

| 预期方向 | 代码状态 | 判定 |
|---|---|---|
| #2 V 跳过 de-RoPE | `_de_rope_for_kind`：V 默认关闭，已固化进全部 3 个配置 | ✅ 完全对路 |
| #3 λ 分通道 | `ridge_lambda_k/v` 旋钮就位 | ⚠️ 半程（未调参） |
| #3 校准样本量 | `real_calibration_samples` 旋钮就位（默认 16） | ⚠️ 半程（默认保守） |
| **#1 V 专项映射** | **未实现**，仍线性 Ridge per-head | ❌ 缺口 |
| 真实化（真实 Student 自产 KV） | HFKVProvider 落地 + T01 real replay + TorchBackend 全链路 | ✅ 对路 |

关键改动：`_real_kv_splits`（公共最小长度裁剪，禁 padding 污染；calib/eval 无重叠校验）、
HFKVProvider（modelscope 源，train/test 隔离，Teacher/Student 不同驻 GPU）、TorchBackend
（prefill 捕获→numpy offload→inject→zero-prefill decode，§52 计数器 + 显式 raise）、
T01 `_real_replay`（native vs numpy 重建 cache 的 logits/token 对比）、T10 CUDA proxy
（诚实标注 `timing_evidence=cuda_proxy_not_end_to_end`）。

**总体判定：工程方向对路，科学攻坚未启动。**
- 已吃到：de-RoPE 修正（+0.03）+ 真实 KV 全链路，正式 T05 重跑将落在 ≈0.867 CONDITIONAL；
- 未动手：分析结论头号方向 V 专项映射（V 0.758→需 0.824），剩余 +0.033 mean 缺口无对策；
- 诚实性保持良好（offline_demo 语义未虚报）。

**下一步（按性价比）**：
1. λ/n_calib 扫描（零代码）：`ridge_lambda_v` ∈ {1e-4..1e-1} × `real_calibration_samples` ∈ {16, 32, 64}；
2. 若线性饱和，实现 V 专用非线性/注意力输出对齐映射；
3. 正式重跑 T05（`provider.kv=hf, de_rope_v=false`）拿合规基线作扫描对照。