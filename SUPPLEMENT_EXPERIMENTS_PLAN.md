# 补充实验方案（交给 GPU 机器执行）

来源：`paper_audit3_1.md` 审稿意见（6/10 Weak Accept）。目标是把审稿人最后几个攻击面封住。

## 所有 run 的通用要求

1. 启动方式：`python -m apcs.cli inject-eval --config <yaml> --new-run`
2. 必须在**同一 git commit**下跑，并保留 `inject-eval/metrics.json` 与
   `inject-eval/inject_eval/capability_score_artifact.json`（**逐样本分数必须保留**，否则无法做置信区间与功效计算）。
3. 对照可比性：统一 `inject_eval.seed: 42`、`calib_eval_split: true`、`eval_from_tail_n: 100`（除长上下文实验另注），
   并确认每个 run 的 `ridge_self_kv`（恒等对照）gold CHG 落在 ±0.001 内——这是该 run 有效的前提。
4. 命名：新 run 名加后缀便于回填论文，例如 `-heo-k3-`、`-align-`、`-persist-`、`-online-`、`-4k-`。

---

## E1（最高优先级）Heo-style top-k 跨层 + de-RoPE ridge

**目的**：复现与本文最接近、且同属 strict zero-re-prefill 的已发表方法（top-k 跨层源选择 + de-RoPE 键映射的闭式线性映射）。
它是审稿意见里第一条、也是唯一可能改变论文结论的实验。

**为什么现在缺**：论文的 mapper ladder 全是 **proportional 逐层对应**；`de_rope_k: true` 已开，缺的是**数据驱动的 top-k 跨层源选择**。

**主要步骤**

1. 在 layer map 构造处（`apcs/inference/evaluator.py::_ensure_mapper`，当前调用
   `apcs/alignment/runner.py::proportional_mapping`）增加一种新策略：
   `mapper.layer_selection: topk` + `mapper.topk: k` + `mapper.topk_metric: ridge_r2`（默认）。
2. 选择准则：用**校准 split** 的 K/V（de-RoPE 后）对每个 (student layer $s$, teacher layer $t$) 拟合一条 per-head ridge 并算 held-out R²，
   按 R² 取该 $s$ 的 top-$k$ 个 teacher 层，写入 `layer_map`（即 `layer_map[s] = [t1..tk]`）。
   备选准则：CCA/CKA。把选出的层表落盘到 run 目录（`layer_mapping.json`）便于审计。
3. 映射本身不改：复用现有 per-head **ridge**（Heo 为闭式线性），`separate_kv: true`、`de_rope_k: true`、
   `ridge_lambda_k/v: 0.001`。
4. 新建 3 个配置，除下面两项外**与 `configs/v15_4b_1.7b_affine_c200_tail.yaml` 完全一致**
   （同对 Qwen3-4B→1.7B、`c=200`、固定 tail-100、`ablation_modes: [self_kv, kv_both]`）：
   `configs/v15_4b_1.7b_heo_topk{1,3,5}_c200_tail.yaml`，`topk: k`、`mapper.type: ridge`。
5. 跑 3 次（k=1,3,5）。若其中某个 k 明显最好，再用 `mapper.type: affine` 补跑 1 次。

**产出**：3–4 个 run（含 `layer_mapping.json`、`metrics.json`、逐样本分数）。

**判据（必须按此汇报，不要只报均值）**：gold CHG 的 95% CI。

- CI 下界 > −0.02 → **replacement 通过**（论文需切换叙事：失败来自逐层对应，而非缓存翻译本身）；
- CI 上界 < 0 → 显著劣化；
- 其它 → 未确立。

**成本**：实现约半天 + 3–4 次 GPU run。

---

## E2 joint MLP 的 train / held-out 重建诊断

**目的**：回答"~37M position-wise joint MLP 失败，是**表征不可恢复**还是**优化/容量不足**"。
最好的结果形态是：held-out 重建很好但下游仍差 —— 即"缓存状态保真度 ≠ 能力保真度"。

**主要步骤**

1. 扩展 `scripts/measure_reconstruction_r2.py`（增加 `--mapper joint_mlp` 分支，或新写一个脚本），
   在同样的 **20 条拟合 / 10 条 held-out context** 上报告 K/V 的 train R² 与 held-out R²（或对应重建 loss）。
2. 同时记录 joint MLP 训练结束时的拟合 loss（当前只在下游指标里可见）。
3. 输出到 `reports/reconstruction_r2/joint_mlp_reconstruction.json`。

**判据**：train R² ≈ 1 且 held-out R² 低 → 泛化限制；train R² 也低 → 优化/容量问题；
held-out R² 也高 → 最强结论（重建不保证能力）。

**成本**：代码 + 1 次 GPU run（20+10 条 context 的前向）。

---

## E3 两个边界学生的 token-aligned control

**目的**：目前唯一通过 replacement gate 的翻译配置（Llama-3.2-1B，+0.000 [−0.012,+0.012]）与卡在 margin 的
Gemma-2-2B 都没有 token 对齐控制；若对齐后结果变化，"唯一 pass 的配置"这一 headline 会变。

**主要步骤**：直接跑已存在的两个配置（无需改代码）

1. `configs/v15_4b_x_llama1b_rect_align_c30.yaml`
2. `configs/v15_4b_x_gemma2-2b_rect_align_c30.yaml`

**产出**：2 个 run（n=100，含逐样本分数）。

**判据**：与未对齐行（Table 3 前 5 行）逐行对比 gold CHG 与 gate 结论。

**成本**：2 次 GPU run。

---

## E4 更长上下文（把 "long-context" 做实）

**目的**：现在所谓 long-context 只有 ~1024 token，审稿人已指出名不符实。

**主要步骤**

1. 复制 `configs/v15_4b_1.7b_affine_c30_longctx.yaml` → `configs/v15_4b_1.7b_affine_c30_longctx4k.yaml`：
   `context_lengths: [4096]`、`inject_eval.max_samples: 40`、`inject_eval.calib_samples: 20`（显存考虑）、`eval_from_tail_n: 0`。
2. 先跑 4096；若显存允许，再复制一份 `[8192]` 跑。

**判据**：teacher / student / translated 的 acc 与 gold CHG；恒等对照仍需在 ±0.001 内。

**成本**：1–2 次 GPU run（显存是主要风险）。

---

## E5 冷启动（持久化）成对产物

**目的**：论文现在明确写"未做该声明"，因为只有 v1.1 旧协议的单次 online run。
补一对同协议产物后，可把这条从"缺口"升级为结论。

**主要步骤**

1. 配置 A：基于 `configs/v15_4b_1.7b_affine_c30_tail.yaml`，设 `inject_eval.persist_kv: true` → 离线 run，
   产出 run 目录下的 `kv_store/`（含 manifest）。
2. 配置 B：同参数，另设 `inject_eval.online_kv_dir: <A 的 run 目录>/kv_store` → 走 `evaluate_online`（**不加载 Teacher**）。
3. 两次必须同 `seed`、同 `calib_eval_split`、同 `eval_from_tail_n`。

**产出**：2 个 run，metrics 分别 `online: false / true`，且均保留逐样本分数。

**判据**：两次的逐样本 gold 概率一致（报告最大绝对差）；若不一致，如实记录差值，不要修饰。

**成本**：2 次 GPU run（第二次不需要 teacher）。

---

## E6 text-channel 基线补逐样本分数（让它有 CI）

**目的**：现在只有 n=30 的聚合值、无 CI，却在 Implications 里被当作"诚实退路"。

**主要步骤**：复制 `configs/v12_4b_1.7b_taskmix.yaml`，设 `inject_eval.summary_baseline: true`、
`eval_from_tail_n: 100`、`max_samples: 300`，跑一次。

**判据**：`summary` 行的 acc/gold，以及**逐样本分数**（用于 bootstrap CI）。

**成本**：1 次 GPU run。

---

## E7 复现覆盖补齐（让附录清单与产物一致）

**目的**：论文的复现审计目前不覆盖长上下文行与 3 个跨架构学生。

**主要步骤**

1. 用 `configs/repro/` 下对应配置补跑：llama32-3b、gemma3-1b、qwen25-1.5b 的 rect（各 1 次）。
2. 排查两个长上下文 repro 目录为何只有 `config.json`/`provider.json`（看 `stdout.log`），修好后补跑 1 次。

**成本**：4 次 GPU run + 1 次排错。

---

## E8（可选）开放式任务，证明不是 answer-letter 打分的产物

**目的**：现有任务全是 multiple-choice，外部效度偏窄。

**主要步骤**：加一个短答任务适配器（GSM8K exact match 或 TriviaQA short answer），用同一注入管线跑
student / teacher / translated / identity 四行，n ≥ 100，报告 exact match 与逐样本分数。

**成本**：适配器代码 + 2–3 次 GPU run。

---

## 执行顺序与回填方式

| 顺序 | 实验 | 优先级 | 回填位置 |
|---|---|---|---|
| 1 | E1 Heo-style baseline | P0 | Table 2 新增行 + §2.2/§5.2 改写 + Abstract |
| 2 | E3 边界对齐 | P1 | Table 3 新增两行 + §5.4 |
| 3 | E2 joint MLP 诊断 | P1 | 附录新增小表 + §5.2 一句 |
| 4 | E5 冷启动成对 | P2 | §5.1 与 Limitations 的"未测量"改为结论 |
| 5 | E4 4K 上下文 | P2 | Table 2 长上下文行 + Limitations 措辞 |
| 6 | E6 text-channel | P2 | §7 该句加 CI |
| 7 | E7 复现覆盖 | P3 | 附录复现清单 |
| 8 | E8 开放式任务 | P3 | 新增小节（可选） |

**E1 需预先决定的事**：如果 Heo-style 通过了 replacement gate，论文叙事要从"强学生全族失败"
切换为"失败源自朴素逐层对应，而非缓存翻译本身"。建议先写好两个版本的 §5.2 与 Abstract 句，跑完直接二选一，
避免事后调整口径。
