"""T10 System Cost（design.md §4 / §38 / §49 / §53 / §54）。

═══════════════════════════════════════════════════════════════════════════════
严格区分三种场景：

Scenario A (Natural Handoff): §4.1
    Teacher 已运行。Teacher Prefill 是沉没成本。
    PSR_A = 1 - (T_map + T_load + T_query) / T_prefill_S
    论文主图 Figure 3 Pareto 用这一口径。

Scenario B (Teacher-for-Transfer): §4.2
    Teacher 本来不需要运行，只是为了产生 Cache 而执行。
    Cost_B = T_prefill_T + T_map + T_load + T_query
    禁止用 PSR_A 宣称 Scenario B 有端到端收益。

Scenario C (One-Teacher-Many-Student): §4.3
    一次 Teacher Prefill 复用于 N 个 Student / Query。
    N_BE = min{N: Cost_handoff(N) < Cost_baseline(N)}
    必须通过真实硬件实验得到，不能理论估算后当论文结果。

计时规范（§49）：
    warmup + sync + ≥10 repeats + P50 / P95。

测量项（§38）：
    teacher_prefill, map, H2D/load, query_prefill, decode,
    student_full_prefill, VRAM, RAM, cache_bytes

单卡执行策略（§53）：
    Teacher Load → Forward → Capture → CPU Offload →
    Teacher Unload → CUDA Cleanup → Student Load → Map → Inject → Decode

KV 存储（§54）：
    禁止 Dataset × All Layers × All Heads × All Tokens 全保存。
    必须 streaming；full debug KV ≤ 10 samples。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np

from ..io.runs import write_json
from ..metrics import psr_a
from ..utils import percentile


def _cache_bytes(L: int, S: int, H: int, D: int, dtype_bytes: int = 2) -> int:
    """§38 cache_bytes：单模型 KV cache 总字节数（bfloat16 = 2 字节/元素）。

    为什么这样算：
        - 每层 KV cache 形状为 (seq, kv_heads, head_dim)，即 S×H×D 个元素；
        - 共 L 层 → ×L；K 与 V 各一份 → ×2；
        - 乘以 dtype_bytes 得到字节数。
    注意 H 取 num_kv_heads（GQA 下 ≠ num_attention_heads），并先算 KV cache 而
    非模型权重，因此该体积决定了 §53 中 CPU offload / 注入的传输量上限。
    """
    # K + V 双份，乘 2 后得到的是字节数（调用方再 /1024/1024 换算 MB）
    return L * S * H * D * dtype_bytes * 2


def _simulate_timings(ctx: int, seed: int) -> dict[str, float]:
    """离线模拟各组件耗时（ms），全部随 context 线性增长。

    为什么这样算：
        - prefill / map / load / query 都与序列长度成正比（逐 token 处理 KV）；
        - 常数项 = 固定开销（模型加载、kernel 启动），斜率 = 每 token 增量；
        - 真实实验替换为 cuda 计时（time.perf_counter + torch.cuda.synchronize），
          并按 §49 规范 warmup + ≥10 repeats + 取 P50/P95。
    """
    return {
        # Teacher Prefill 斜率最大：Teacher 层数更多（如 Qwen3-4B 36 层）
        "teacher_prefill": 8.0 + ctx * 0.012,
        # Student 自 Prefill 是 PSR_A 的分母（T_prefill_S），斜率次之（28 层）
        "student_prefill": 3.0 + ctx * 0.006,
        # map：逐层投影，开销随层数 × 序列长度线性增长
        "map": 0.5 + ctx * 0.0008,
        # load：H2D / Cache Load 与 KV 字节数成正比（见 _cache_bytes）
        "load": 0.3 + ctx * 0.0005,
        # query：解码查询阶段，开销最轻
        "query": 0.4 + ctx * 0.001,
    }


def run_system_cost(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T10 入口：三种 Scenario 报告 + §38 全部测量项 + §53 单卡流程。

    结构：
        1. 读取 context_lengths_extended / seeds / timing 配置；
        2. 逐 context × seed 模拟各组件耗时；
        3. 逐 context 计算 Scenario A (PSR_A) / B (Cost_B) / C (N_BE)；
        4. 跨 seed 聚合 P50/P95（§49 计时规范）；
        5. 落 system.json + summary.md（§63 产物规范）。
    """
    # P1 扩展长度档位（主实验之外的 8192 / 16384），KV 体积与耗时都随长度变化
    contexts = cfg.get("context_lengths_extended", [1024, 4096, 8192, 16384])
    seeds = cfg.get("seeds", [0, 1, 2])
    repeats = int(cfg.get("timing", {}).get("repeats", 10))
    warmup = int(cfg.get("timing", {}).get("warmup", 2))

    # §38 KV cache 体积：以 Teacher 的 KV head 配置估算两个模型的 cache 体积
    n_t = cfg["teacher"].get("num_layers", 36)
    n_s = cfg["student"].get("num_layers", 28)
    H = cfg["teacher"].get("num_kv_heads", 8)
    D = cfg["teacher"].get("head_dim", 128)

    # 计时来源：provider.timing == "hf" → HFTimingProvider 真实 CUDA 计时（§49），
    # GPU 不可用时 open() 显式 raise（§75，不静默回退线性公式）；
    # 否则回退 _simulate_timings 线性公式（offline_demo=True）。
    timing_kind = cfg.get("provider", {}).get("timing", "synthetic").lower()
    if timing_kind in {"hf", "artifact"}:
        from ..providers import providers_ctx, write_provider_manifest

        with providers_ctx(cfg, need=("timing",)) as ps:
            timing_description = ps["timing"].describe()
            write_provider_manifest(run_dir, timing=ps["timing"])

        def _measure(ctx: int, seed: int) -> dict[str, float]:
            with providers_ctx(cfg, need=("timing",)) as ps:
                return ps["timing"].measure(ctx, seed)

        timing_evidence = str(
            timing_description.get("timing_evidence", "cuda_proxy_not_end_to_end")
        )
        real_timing = timing_evidence == "end_to_end_handoff"
    else:

        def _measure(ctx: int, seed: int) -> dict[str, float]:
            return _simulate_timings(ctx, seed=seed)

        real_timing = False
        timing_evidence = "synthetic_formula"
        timing_description = {"evidence_grade": "synthetic"}

    per_ctx = []
    for ctx in contexts:
        per_seed = []
        for seed in seeds:
            timings = _measure(ctx, seed=int(seed))
            # §4.1 Scenario A：Teacher Prefill 是沉没成本，分子只算 handoff 增量开销；
            # PSR_A > 0 说明 (T_map+T_load+T_query) < T_prefill_S，迁移才有系统收益
            pa = psr_a(
                timings["map"],
                timings["load"],
                timings["query"],
                timings["student_prefill"],
            )
            # §4.2 Scenario B：Teacher 只为迁移而跑，Cost_B 全量计费（含 T_prefill_T）；
            # 严禁用 Scenario A 的 PSR_A 口径宣称 B 有端到端收益（§38）
            cost_b = (
                timings["teacher_prefill"]
                + timings["map"]
                + timings["load"]
                + timings["query"]
            )
            # §4.3 Scenario C：解 N_BE 方程
            # Cost_handoff(N) < Cost_baseline(N) ⇔ N > (T_prefill_T+T_map)/(T_prefill_S−T_load−T_query)
            # 每次复用省下 (T_prefill_S − T_load − T_query)；分母 ≤ 0 说明复用永远不划算 → ∞
            slope = timings["student_prefill"] - (timings["load"] + timings["query"])
            if slope <= 0:
                n_be = float("inf")
            else:
                n_be = (timings["teacher_prefill"] + timings["map"]) / slope
                n_be = max(1, int(round(n_be)))  # 至少 1 个 Query 才有意义
            per_seed.append(
                {
                    "psr_a": pa,
                    "cost_b": cost_b,
                    "n_be": n_be,
                    "timings": timings,
                }
            )

        # §49 P50 / P95 across seeds：跨 seed 取分位数，p50 为中位、p95 反映尾部波动
        psr_vals = [x["psr_a"] for x in per_seed]
        cost_vals = [x["cost_b"] for x in per_seed]
        per_ctx.append(
            {
                "context": ctx,
                "psr_a_p50": percentile(psr_vals, 0.50),
                "psr_a_p95": percentile(psr_vals, 0.95),
                # N_BE 取整后跨 seed 的中位数
                "n_be_median": sorted(x["n_be"] for x in per_seed)[len(per_seed) // 2],
                "cost_b_p50": percentile(cost_vals, 0.50),
                "cost_b_p95": percentile(cost_vals, 0.95),
                # §38 cache_bytes：KB→MB 换算在 summary 表格里做
                "teacher_cache_bytes": _cache_bytes(n_t, ctx, H, D),
                "student_cache_bytes": _cache_bytes(n_s, ctx, H, D),
            }
        )

    # §38 VRAM / RAM（离线估值；真实实验读 torch.cuda.max_memory_allocated + psutil）
    # 按 层数 × 4096 参数 × 2 字节（bfloat16）估算权重大小，只反映数量级
    vram_mb_est = (n_t * 4096 * 2) / (1024 * 1024)  # 估算 Teacher 权重大小
    ram_mb_est = (n_s * 4096 * 2) / (1024 * 1024)
    if real_timing:
        # provider.timing=hf：真实 VRAM 实测（torch.cuda.max_memory_allocated）
        from ..providers import providers_ctx

        with providers_ctx(cfg, need=("timing",)) as ps:
            vram_mb_est = float(ps["timing"].measure_vram())

    metrics = {
        "task": "T10",
        "contexts": contexts,
        "repeats": repeats,
        "warmup": warmup,
        "per_context": per_ctx,
        "vram_teacher_mb_est": vram_mb_est,
        "ram_student_mb_est": ram_mb_est,
        # 三种场景定义落盘，供 T11 verdict 与 Figure 3 Pareto 按场景取口径
        "scenario_definitions": {
            "A": "natural handoff; PSR_A = 1 - (T_map+T_load+T_query)/T_prefill_S",
            "B": "teacher-for-transfer; Cost = T_prefill_T + T_map + T_load + T_query",
            "C": "one-teacher-many-students; N_BE = ceil((T_prefill_T + T_map) / (T_prefill_S - T_load - T_query))",
        },
        # §53 单卡流水线顺序：Teacher 完整卸载 + CUDA 清理后再加载 Student，
        # 避免 Teacher/Student 同时在显存造成峰值超限
        "single_card_pipeline_order": [
            "teacher_load",
            "forward",
            "capture",
            "cpu_offload",
            "teacher_unload",
            "cuda_cleanup",
            "student_load",
            "map",
            "inject",
            "decode",
        ],
        # ---- §75 诚实性标注（架构审查 P1-4 修复 + real-gpu 接入）----
        # 合成路径：耗时来自 _simulate_timings 线性公式（非实测），VRAM/RAM 为
        #   量级估算 → PSR_A / Cost_B / N_BE 全部是推导值，offline_demo=True；
        # 真实路径（provider.timing=hf）：真实 CUDA 计时（§49 warmup+repeats+
        #   P50/P95）+ VRAM 实测，offline_demo=False。
        "offline_demo": not real_timing,
        "timing_evidence": timing_evidence,
        "evidence_grade": (
            "measured_system"
            if real_timing
            else ("proxy" if timing_kind == "hf" else "synthetic")
        ),
        "timing_provenance": timing_description,
        "note": (
            "T10 使用通过审计的 HandoffPipeline 端到端 artifact。"
            if real_timing
            else (
                "T10 全部耗时由 _simulate_timings 线性公式生成，"
                "PSR_A / Cost_B / N_BE 均为推导值而非实测。"
                if timing_evidence == "synthetic_formula"
                else "T10 使用 CUDA 代理算子诊断，不是端到端计时。"
            )
        ),
    }
    write_json(run_dir / "system.json", metrics)
    warning = ""
    if not real_timing:
        warning = (
            "> ⚠️ **offline demo**：耗时来自线性公式模拟。\n\n"
            if timing_evidence == "synthetic_formula"
            else "> ⚠️ **CUDA proxy**：并非 HandoffPipeline 端到端计时。\n\n"
        )
    md = (
        "# T10 System Cost\n\n"
        + warning
        + f"- VRAM (Teacher est): {vram_mb_est:.1f} MB\n"
        f"- RAM (Student est): {ram_mb_est:.1f} MB\n"
        f"- Repeats: {repeats}, Warmup: {warmup}\n\n"
        "| ctx | PSR_A p50 | PSR_A p95 | Cost_B p50 (ms) | Cost_B p95 (ms) | N_BE | "
        "teacher_KV (MB) | student_KV (MB) |\n"
        "| --: | --------: | --------: | --------------: | --------------: | ---: | "
        "-------------: | --------------: |\n"
        + "\n".join(
            f"| {r['context']} | {r['psr_a_p50']:.3f} | "
            f"{r['psr_a_p95']:.3f} | {r['cost_b_p50']:.2f} | "
            f"{r['cost_b_p95']:.2f} | {r['n_be_median']} | "
            f"{r['teacher_cache_bytes']/1024/1024:.1f} | "
            f"{r['student_cache_bytes']/1024/1024:.1f} |"
            for r in per_ctx
        )
        + "\n\n## Single-card pipeline order (§53)\n"
        + " → ".join(metrics["single_card_pipeline_order"])
        + "\n"
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    # ◆ 修复（design-gap-review B7 同类问题）：此前返回 "metrics": {} —— 
    #   CLI 会据此写出**空的 metrics.json**，而 fig3（CHG–PSR_A Pareto）与
    #   fig4（Context Scaling）都从 t10_metrics["per_context"] 取数 → 拿不到数据。
    #   现在 metrics 与 system 落同一份内容：system.json 保持 §63 规范不变，
    #   metrics.json 同时可用，figures / T11 均可直接消费。
    return {"status": "OK", "system": metrics, "metrics": metrics, "summary": md}
