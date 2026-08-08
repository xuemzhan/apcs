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

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any


# ---- §64 标准字段 ----

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
    """校验 cfg 是否覆盖 §64 全部字段；返回缺失字段名列表。"""
    missing: list[str] = []
    for field in REQUIRED_64:
        if field not in cfg:
            missing.append(field)
    return missing


def enrich_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """从 cfg['teacher'] / cfg['student'] / cfg['mapper'] 等段补出 §64 标准字段。

    不覆盖已存在的字段（保留用户显式配置）。
    返回新的 cfg（不修改原对象）。
    """
    teacher = cfg.get("teacher", {})
    student = cfg.get("student", {})
    mapper = cfg.get("mapper", {})
    loss = cfg.get("loss", {})

    out = dict(cfg)
    # 顶层 §64 字段补全
    out.setdefault("teacher_model", teacher.get("model_id"))
    out.setdefault("teacher_commit", teacher.get("revision"))
    out.setdefault("student_model", student.get("model_id"))
    out.setdefault("student_commit", student.get("revision"))
    out.setdefault("tokenizer", teacher.get("model_id"))  # 默认共享 tokenizer
    out.setdefault("dtype", teacher.get("dtype"))
    out.setdefault("attention_implementation", teacher.get("attention_implementation"))
    out.setdefault("rope_config", {
        "theta": teacher.get("rope_theta"),
        "scaling": teacher.get("rope_scaling"),
    })
    out.setdefault("cache_layout", "batch_first")  # 默认 layout
    out.setdefault("context_length", cfg.get("context_lengths", [1024])[0])
    out.setdefault("dataset", (cfg.get("datasets", {}).get("teacher_advantage", {}) or {}).get("primary"))
    out.setdefault("split", "test")
    out.setdefault("seed", cfg.get("seeds", [0])[0])
    out.setdefault("mapper", mapper.get("type", "ridge"))
    out.setdefault("rank", mapper.get("rank", 16))
    out.setdefault("source_top_k", mapper.get("source_top_k", 2))
    out.setdefault("alpha_max", mapper.get("alpha_max", 0.5))
    out.setdefault("loss_weights", {
        "lambda_task": loss.get("lambda_task", 1.0),
        "lambda_self": loss.get("lambda_self", 0.1),
        "lambda_teacher": loss.get("lambda_teacher", 0.1),
        "lambda_att": loss.get("lambda_att", 1.0),
        "lambda_reg": loss.get("lambda_reg", 0.01),
    })
    out.setdefault("hardware", {
        "cuda_available": _cuda_available(),
        "device": teacher.get("device_map", "cpu"),
    })
    out.setdefault("cache_residency", cfg.get("cache_residency", "cpu"))
    return out


def enrich_metrics(metrics: dict[str, Any], per_method: list[dict] | None = None) -> dict[str, Any]:
    """把 metrics 补齐到 §65 标准字段。

    已有字段保留；缺失字段从 per_method / 上下文推断；
    实在推断不出则填 null。

    参数：
        metrics:  当前 metrics dict
        per_method: 可选，从 T09 per_method 中提取 base_plus_adv 等指标
    """
    out = dict(metrics)
    # §65 字段初始化
    for field in REQUIRED_65:
        out.setdefault(field, None)

    # 从 per_method 推断
    if per_method:
        for row in per_method:
            m = row.get("method")
            if m == "student":
                out["student_score"] = row.get("score")
            elif m == "teacher":
                out["teacher_score"] = row.get("score")
            elif m == "base_plus_adv":
                out["handoff_score"] = row.get("score")
                out["chg"] = row.get("chg")
                out["tgrr"] = row.get("tgrr")
                out["retention"] = row.get("retention")
                out["jcr"] = row.get("jcr_vs_student")

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

    返回写入路径的 dict。
    """
    from ..io.runs import write_json

    enriched_cfg = enrich_config(cfg)
    enriched_metrics = enrich_metrics(metrics, per_method)
    missing_cfg = validate_config(enriched_cfg)

    payload = {
        "generated_at": _dt.datetime.now().isoformat(),
        "config_enriched": enriched_cfg,
        "config_missing_fields": missing_cfg,
        "metrics_enriched": enriched_metrics,
        "config_schema": REQUIRED_64,
        "metrics_schema": REQUIRED_65,
    }
    write_json(run_dir / "metadata.json", payload)
    return payload


def _cuda_available() -> bool:
    """§64 hardware.cuda_available 检测。"""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except ImportError:
        return False