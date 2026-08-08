"""APCS 实验 CLI 入口（design.md §63, §70, §71, §72）。

═══════════════════════════════════════════════════════════════════════════════
每个 task 一个 subcommand：t00..t12，加上 ablation / multiturn。

§63 Run 数据规范：每个 Run 写到
    reports/runs/<run_id>/<task>/{
        config.json, metrics.json, system.json, geometry.json,
        summary.md, stdout.log, task_report.md
    }

§63 stdout.log 由本 CLI 通过 redirect_stdout/stderr 捕获。

§70 PREREGISTRATION.md 在 t07 (Teacher Gap Freeze) 后自动生成。

§72 强制规则：
    - 一次只允许执行一个 Task
    - 每个 Task 有 Acceptance Test
    - Gate FAIL 后不能跳过（orchestrator 校验）
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .io import load_config
from .io.runs import ensure_run_dir, write_json, write_text
from .orchestrator import write_task_report

# task id → (module, function, title, docstring objective)
TASKS = {
    "t00": ("apcs.compat.scanner", "run_compat_scan", "T00 Compatibility Scanner",
            "读取 teacher/student 架构，输出 model_compatibility.json 并判定 G1/G2/G3。"),
    "t01": ("apcs.replay.runner", "run_self_kv_replay", "T01 Self-KV Replay",
            "验证 Student 自产 KV 注入与原生推理等价（Gate 0）。"),
    "t02": ("apcs.rope.runner", "run_rope_roundtrip", "T02 RoPE Round-trip",
            "3 layers × 3 heads × 128 tokens 抽样验证 K → de-RoPE → re-RoPE ≈ K。"),
    "t03": ("apcs.alignment.runner", "run_layer_alignment", "T03 Layer Alignment",
            "构造 Teacher↔Student 4 种层映射策略：proportional / last / data-driven / geometry-aware。"),
    "t04": ("apcs.mapper.runner", "run_ridge_baseline", "T04 Ridge Baseline",
            "Full Ridge 校准，输出 R² / KV cosine / attn-output cosine。"),
    "t05": ("apcs.mapper.runner", "run_replacement", "T05 Replacement",
            "跨模型 KV 替换 + Retention + KL + Token Agreement + Latency（Gate 1）。"),
    "t06": ("apcs.mapper.runner", "run_lightweight_mapper", "T06 Lightweight Mapper",
            "Low-rank 8/16/32 + Shared basis → PCR vs Retention（Figure 1）。"),
    "t07": ("apcs.capability.runner", "run_teacher_gap_freeze", "T07 Teacher Gap Freeze",
            "Teacher-Student gap 分桶（Low/Med/High）；触发 PREREGISTRATION.md 生成（§70）。"),
    "t08": ("apcs.advantage.runner", "run_advantage_train", "T08 Advantage State Training",
            "Source-Layer Mixer + K/V 独立低秩残差 + RMS Calibration + Bounded α。"),
    "t09": ("apcs.capability.runner", "run_main_capability", "T09 Main Capability",
            "CHG / TGRR / JCR 主结果 + bootstrap CI + permutation test + Gap strata 报告（Gate 2A）。"),
    "t10": ("apcs.system.runner", "run_system_cost", "T10 System Cost",
            "Scenario A/B/C + PSR_A + N_BE + cache_bytes/VRAM/RAM + §53 单卡流程。"),
    "t11": ("apcs.decision.runner", "run_mvp_decision", "T11 MVP Decision",
            "读 T05/T09/T10 输出 Path A/B/C/D（§69）。"),
    "t12": ("apcs.geometry.runner", "run_geometry_diagnostics", "T12 Geometry Diagnostics",
            "CKA / principal angle / attn-output cosine / effective rank / head correlation（Figure 7）。"),
    "t13": ("apcs.generalization.runner", "run_generalization", "T13 Generalization",
            "扫描所有 run_id 的 T05/T09 Gate 2A 状态，输出 Path A 在 ≥ 2 个 Pair 上的可复现性（§11 / §44 / §69）。"),
    "ablation": ("apcs.ablation.runner", "run_ablation", "§47 Ablation",
                 "A1 Rank / A6 de-RoPE / A8 KV / A3 / A9 / A10 消融。"),
    "multiturn": ("apcs.multiturn.runner", "run_multi_turn", "§46 Multi-turn",
                  "1/5/10/20 轮 CHG / KL / JCR / task score / latency（Figure 5）。"),
    "compliance": ("apcs.compliance", "run_compliance_check", "§52 Compliance",
                   "跑全部 8 条禁止项检查器，输出 violations 列表。"),
}


def _import_attr(module_path: str, attr: str):
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _write_task_report(
    cfg: dict[str, Any],
    run_dir: Path,
    task: str,
    result: dict,
    status: str,
) -> None:
    """§73 Task Report 12 字段。"""
    output_files = sorted(
        str(p.relative_to(run_dir)) for p in run_dir.glob("*") if p.is_file()
    )
    write_task_report(
        run_dir=run_dir,
        task_id=task,
        status=status,
        objective=TASKS[task][3],
        model_pair=(cfg["teacher"]["model_id"], cfg["student"]["model_id"]),
        dataset=str(cfg.get("datasets", {}).get("teacher_advantage", {}).get("primary", "")),
        config_summary={
            "context_lengths": cfg.get("context_lengths"),
            "seeds": cfg.get("seeds"),
            "mapper": cfg.get("mapper"),
            "advantage": cfg.get("advantage"),
        },
        implementation=TASKS[task][2],
        output_files=output_files,
        key_metrics=result.get("metrics", {}),
        statistical_check={
            "n_samples": result.get("metrics", {}).get("n_samples_per_seed"),
            "seeds": cfg.get("seeds"),
            "ci": cfg.get("statistics", {}).get("ci"),
        },
        behavior_check={"jcr": "见 metrics.jcr_vs_student" if "per_method" in result.get("metrics", {}) else "n/a"},
        geometry_check={"mean_cka": result.get("geometry", {}).get("mean_cka") if "geometry" in result else "n/a"},
        system_cost=result.get("system", {}),
        acceptance_criteria={"gate": status},
        failure_analysis=None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apcs", description="APCS 实验 CLI")
    parser.add_argument("task", choices=sorted(TASKS.keys()))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None,
                        help="覆盖默认 run_id；不传则用 config 中的 run_id")
    parser.add_argument("--no-prereg", action="store_true",
                        help="跳过 PREREGISTRATION.md 生成")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_id = args.run_id or cfg["experiment"]["run_id"]
    base = Path(cfg["output"]["base_dir"])
    run_root = ensure_run_dir(base, run_id)
    # 每个 task 的产物放在 `<run_root>/<task>/` 下，方便 T11 聚合
    run_dir = run_root / args.task
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", cfg)

    module_path, func_name, title, objective = TASKS[args.task]
    print(f"[apcs] task={args.task} title={title}")
    print(f"[apcs] run_id={run_id}")
    print(f"[apcs] run_dir={run_dir}")

    fn = _import_attr(module_path, func_name)

    # §63 stdout.log：捕获 stdout/stderr 到文件
    log_path = run_dir / "stdout.log"
    log_f = log_path.open("w", encoding="utf-8")

    @contextlib.contextmanager
    def _tee(stream):
        class _Tee:
            def write(self, s):
                stream.write(s)
                log_f.write(s)
                log_f.flush()

            def flush(self):
                stream.flush()
                log_f.flush()

        yield _Tee()

    with contextlib.redirect_stdout(_tee(sys.stdout)), contextlib.redirect_stderr(_tee(sys.stderr)):
        try:
            result = fn(cfg, run_dir)
            status = result.get("status", "UNKNOWN")
        except Exception as e:  # noqa: BLE001
            log_f.write(f"\n[ERROR] {type(e).__name__}: {e}\n")
            log_f.flush()
            raise
    log_f.close()

    write_json(run_dir / "metrics.json", result.get("metrics", {}))
    if "system" in result:
        write_json(run_dir / "system.json", result["system"])
    if "geometry" in result:
        write_json(run_dir / "geometry.json", result["geometry"])
    write_text(run_dir / "summary.md", result.get("summary", ""))

    # §64/§65 metadata.json（标准字段补全）
    try:
        from .io.metadata import write_run_metadata

        write_run_metadata(
            cfg,
            result.get("metrics", {}),
            run_dir,
            per_method=result.get("metrics", {}).get("per_method"),
        )
    except Exception as e:  # noqa: BLE001
        log_f = log_path.open("a", encoding="utf-8")
        log_f.write(f"\n[metadata warn] {type(e).__name__}: {e}\n")
        log_f.close()

    # §73 Task Report
    _write_task_report(cfg, run_dir, args.task, result, status)

    # §70 PREREGISTRATION：t07 后自动生成
    if args.task == "t07" and not args.no_prereg:
        from .prereg import generate_prereg

        generate_prereg(cfg, run_root / "PREREGISTRATION.md")

    print(f"[apcs] DONE. status={status}")
    print(json.dumps(result.get("metrics", {}), indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "OK"} else 1


if __name__ == "__main__":
    sys.exit(main())