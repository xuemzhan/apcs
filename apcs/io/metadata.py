"""§64 config.yaml schema 校验 + §65 metrics.json 字段补全。

═══════════════════════════════════════════════════════════════════════════════
§64 config.yaml 必须记录的字段：
    teacher_model, teacher_commit, student_model, student_commit,
    tokenizer, dtype, attention_implementation, rope_config, cache_layout,
    context_length, dataset, split, seed, mapper, rank, source_top_k,
    alpha_max, loss_weights, hardware, cache_residency

§65 metrics.json 字段：
    student_score, teacher_score, handoff_score, retention, chg, tgrr,
    pcr, psr_a, next_token_kl, jcr, attention_output_cosine, linear_cka,
    principal_angle, effective_rank

本模块提供：
    - validate_config(cfg):  校验 cfg 是否覆盖 §64 字段，缺则警告
    - enrich_config(cfg):    自动从 teacher/student 段补出 §64 标准字段
    - enrich_metrics(metrics, ...): 把 metrics 补齐到 §65 标准字段
    - write_run_metadata(cfg, metrics, run_dir): 一次性把 §64/§65 全写入

YAML 读 → 写 完整流程（本模块是最后一环）：
    1. 读：apcs.io.load_config(path) 用 yaml.safe_load 解析 config.yaml，
       并把 ${run.timestamp} 占位符展开为运行时刻时间戳（%Y%m%d-%H%M%S）。
    2. 补：enrich_config(cfg) 从 teacher/student/mapper/loss 嵌套段
       setdefault 出 §64 扁平标准字段（不覆盖用户显式配置）。
    3. 校验：validate_config(enriched_cfg) 列出仍缺失的 §64 字段。
    4. 写：write_run_metadata(...) 组合 §64 + §65 成 metadata.json，
       经 apcs.io.runs.write_json 落盘（JSON，UTF-8，ensure_ascii=False）。

seed / rng 可复现性说明：
    §64 的 seed 字段记录实验随机种子（默认取 cfg.seeds[0]）。
    实际随机状态由 apcs.utils.set_seed(seed) 固定 Python / NumPy / Torch 的 rng；
    本模块只负责把 seed 持久化进 metadata.json，作为"该 run 从哪个 rng 状态
    出发"的审计依据——重放时凭 run_id + seed 即可复现同分布采样。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any


# ---- §64 标准字段 ----
# 分组含义（与 config.yaml 嵌套段对应）：
#   [teacher/student]     模型名 + commit，另含共享 tokenizer / dtype / attention_implementation
#   [rope]                rope_config(theta, scaling)
#   [cache]               cache_layout / context_length / cache_residency
#   [dataset]             dataset / split
#   [repro]               seed（rng 起点，配合 apcs.utils.set_seed）
#   [mapper]              mapper / rank / source_top_k / alpha_max（§20/§22 超参）
#   [loss]                loss_weights（任务/自回归/教师/attention/正则 五路加权）
#   [hardware]            cuda_available / device

REQUIRED_64 = [
    "teacher_model",
    "teacher_commit",
    "student_model",
    "student_commit",
    "tokenizer",
    "dtype",
    "attention_implementation",
    "rope_config",
    "cache_layout",
    "context_length",
    "dataset",
    "split",
    "seed",
    "mapper",
    "rank",
    "source_top_k",
    "alpha_max",
    "loss_weights",
    "hardware",
    "cache_residency",
]

# ---- §65 标准字段 ----
# 分组含义（详见 apcs.metrics / design.md §3/§4/§45/§48/§51）：
#   [分数]      student_score / teacher_score / handoff_score（§3 三个能力分）
#   [能力迁移]  retention / chg / tgrr（§3.1–§3.3 核心端点）
#   [系统收益]  pcr / psr_a（§3.4 参数压缩比 / §4 Scenario A Prefill 节省率）
#   [行为]      next_token_kl / jcr（§45 输出分布 KL 与判断一致性）
#   [几何]      attention_output_cosine / linear_cka / principal_angle / effective_rank
#               （§40/§62 机制诊断，仅 T12 写，其余 task 为 null）

REQUIRED_65 = [
    "student_score",
    "teacher_score",
    "handoff_score",
    "retention",
    "chg",
    "tgrr",
    "pcr",
    "psr_a",
    "next_token_kl",
    "jcr",
    "attention_output_cosine",
    "linear_cka",
    "principal_angle",
    "effective_rank",
]


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """校验 cfg 是否覆盖 §64 全部字段；返回缺失字段名列表。

    输入：任意 dict（通常为 enrich_config 输出）。
    输出：缺失字段名列表（空列表 = §64 schema 完整）。
    注意：只做存在性检查，不做类型/取值校验（§64 语义校验由调用方负责）。
    """
    missing: list[str] = []
    for field in REQUIRED_64:
        if field not in cfg:  # 键缺失即记为缺失（值为 None 也算存在）
            missing.append(field)
    return missing


def enrich_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """从 cfg['teacher'] / cfg['student'] / cfg['mapper'] 等段补出 §64 标准字段。

    输入：原始 config.yaml 解析出的嵌套 dict（含 teacher/student/mapper/loss 段）。
    输出：新的顶层扁平 dict——§64 字段齐全，多余嵌套段原样保留。

    不覆盖已存在的字段（setdefault 语义：已有值优先，保留用户显式配置）。
    返回新的 cfg（不修改原对象）。
    """
    # 取各嵌套段（缺段按空 dict 处理，后续 setdefault 落默认值）
    teacher = cfg.get("teacher", {})
    student = cfg.get("student", {})
    mapper = cfg.get("mapper", {})
    loss = cfg.get("loss", {})

    out = dict(cfg)
    # 顶层 §64 字段补全：setdefault 保证"显式配置 > 推断值 > 硬编码默认值"优先级
    out.setdefault("teacher_model", teacher.get("model_id"))
    out.setdefault("teacher_commit", teacher.get("revision"))
    out.setdefault("student_model", student.get("model_id"))
    out.setdefault("student_commit", student.get("revision"))
    out.setdefault("tokenizer", teacher.get("model_id"))  # 默认共享 tokenizer
    out.setdefault("dtype", teacher.get("dtype"))
    out.setdefault("attention_implementation", teacher.get("attention_implementation"))
    # rope 配置打包成嵌套 dict：theta + scaling（RoPE 往返 §30 依赖 theta 一致）
    out.setdefault("rope_config", {
        "theta": teacher.get("rope_theta"),
        "scaling": teacher.get("rope_scaling"),
    })
    out.setdefault("cache_layout", "batch_first")  # 默认 layout
    # context_length 取主序列首个长度（cfg.context_lengths[0]，如 1024）
    out.setdefault("context_length", cfg.get("context_lengths", [1024])[0])
    # dataset 取 teacher_advantage.primary（§16 优势集主 benchmark）
    out.setdefault("dataset", (cfg.get("datasets", {}).get("teacher_advantage", {}) or {}).get("primary"))
    out.setdefault("split", "test")
    # seed：rng 起点，配合 apcs.utils.set_seed 复现同分布采样
    out.setdefault("seed", cfg.get("seeds", [0])[0])
    out.setdefault("mapper", mapper.get("type", "ridge"))
    out.setdefault("rank", mapper.get("rank", 16))
    out.setdefault("source_top_k", mapper.get("source_top_k", 2))
    out.setdefault("alpha_max", mapper.get("alpha_max", 0.5))
    # loss_weights：五路加权 scoring 系数（§22/§24 损失为加权和）
    #   lambda_task     主任务（task 能力）权重，默认 1.0
    #   lambda_self     student 自回归一致权重，默认 0.1
    #   lambda_teacher  teacher 蒸馏/指导权重，默认 0.1
    #   lambda_att      attention 输出对齐权重，默认 1.0（优先于 raw KV MSE）
    #   lambda_reg      正则项权重，默认 0.01（惩罚过大的合成残差/参数）
    out.setdefault("loss_weights", {
        "lambda_task": loss.get("lambda_task", 1.0),
        "lambda_self": loss.get("lambda_self", 0.1),
        "lambda_teacher": loss.get("lambda_teacher", 0.1),
        "lambda_att": loss.get("lambda_att", 1.0),
        "lambda_reg": loss.get("lambda_reg", 0.01),
    })
    # hardware：运行时探测 cuda 可用性 + 设备映射（_cuda_available 惰性导入 torch）
    out.setdefault("hardware", {
        "cuda_available": _cuda_available(),
        "device": teacher.get("device_map", "cpu"),
    })
    # cache_residency：缓存驻留层 R0 gpu / R1 cpu / R2 nvme（§54）
    out.setdefault("cache_residency", cfg.get("cache_residency", "cpu"))
    return out


def enrich_metrics(metrics: dict[str, Any], per_method: list[dict] | None = None) -> dict[str, Any]:
    """把 metrics 补齐到 §65 标准字段。

    已有字段保留；缺失字段从 per_method / 上下文推断；
    实在推断不出则填 null（如 psr_a 仅 T10 有、几何字段仅 T12 有）。

    参数：
        metrics:  当前 metrics dict（task 主指标，键名随 task 各异）
        per_method: 可选，从 T09 per_method 中提取 base_plus_adv 等指标

    返回：新的 dict，保证 §65 全部 14 个键都存在（值可能为 None）。
    """
    out = dict(metrics)
    # §65 字段初始化：先全部置 None，再逐项覆盖，保证 schema 完整性
    for field in REQUIRED_65:
        out.setdefault(field, None)

    # 从 per_method 推断（T09 §37 输出结构：[{method, score, chg, tgrr, ...}, ...]）
    if per_method:
        for row in per_method:
            m = row.get("method")
            if m == "student":
                # student 自 prefill 基线分 → student_score
                out["student_score"] = row.get("score")
            elif m == "teacher":
                # teacher 参考上限分 → teacher_score
                out["teacher_score"] = row.get("score")
            elif m == "base_plus_adv":
                # APCS（Base + Advantage）主实验法：其 score 即 handoff_score，
                # 同时带回 §3.1–§3.3 核心端点与 §45 行为一致性指标
                out["handoff_score"] = row.get("score")
                out["chg"] = row.get("chg")          # = handoff − student_self，首要科学端点
                out["tgrr"] = row.get("tgrr")        # Teacher gap 恢复率
                out["retention"] = row.get("retention")  # = handoff / student_self
                out["jcr"] = row.get("jcr_vs_student")   # §45 判断一致性 vs student

    # 从 system.json 推断 PSR_A
    # （metrics 通常不含 psr_a，由 system 单独存；这里保留 None 让 caller 补）
    return out


def write_run_metadata(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
    per_method: list[dict] | None = None,
) -> dict[str, Any]:
    """一次性把 §64 + §65 全字段写到 run_dir/metadata.json。

    输入：
        cfg:     原始 config dict（内部先 enrich_config 补全 §64）
        metrics: 原始 metrics dict（内部先 enrich_metrics 补全 §65）
        run_dir: 当前 run 的任务目录 reports/runs/<run_id>/<task>/
        per_method: 可选，透传给 enrich_metrics 用于 §65 推断

    输出：metadata.json 落盘 + 返回写出的 payload dict。
    依赖 apcs.io.runs.write_json 完成 JSON 序列化（UTF-8，ensure_ascii=False）。
    """
    from ..io.runs import write_json

    # 三步流水线：补全 → 校验 → 组装
    enriched_cfg = enrich_config(cfg)
    enriched_metrics = enrich_metrics(metrics, per_method)
    missing_cfg = validate_config(enriched_cfg)

    # metadata.json 顶层结构（§64/§65 schema 版本化 + 审计信息）：
    payload = {
        "generated_at": _dt.datetime.now().isoformat(),  # 写入时刻（ISO-8601），溯源用
        "config_enriched": enriched_cfg,                  # 补全后的 §64 配置（含 seed/rng 起点）
        "config_missing_fields": missing_cfg,             # 仍缺失字段清单（空列表=完整）
        "metrics_enriched": enriched_metrics,             # 补全后的 §65 指标
        "config_schema": REQUIRED_64,                     # §64 字段清单（schema 版本自描述）
        "metrics_schema": REQUIRED_65,                    # §65 字段清单（schema 版本自描述）
    }
    write_json(run_dir / "metadata.json", payload)
    return payload


def _cuda_available() -> bool:
    """§64 hardware.cuda_available 检测。

    惰性导入 torch：CPU-only 环境（无 torch）时返回 False 而非抛异常，
    保证 metadata 补全流程在无 GPU 依赖下也可运行（CI / 单元测试）。
    """
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except ImportError:
        return False  # torch 未安装：视为无 CUDA