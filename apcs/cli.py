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

# §71 子命令映射表：task id → (模块路径, 入口函数名, 显示标题, 目标说明)。
# 每个子命令运行时用 `_import_attr` 动态导入对应 runner，
# 统一按 runner(cfg, run_dir) → dict 的契约调用（见 main 内注释）。
#   模块路径   — 该 task 的 runner 所在包，如 apcs.compat.scanner
#   入口函数名 — 实际执行的入口函数，如 run_compat_scan
#   显示标题   — 命令行打印的 task 名称（如 "T00 Compatibility Scanner"）
#   目标说明   — §73 Task Report 的 OBJECTIVE 字段内容
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
    """按 TASKS 表的 (模块, 函数) 动态加载 runner 入口。

    只在函数内 import importlib（延迟加载），避免 CLI 启动时
    把 13+3 个 runner 全部 import 一遍拖慢冷启动。
    """
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
    """§73 Task Report 12 字段。

    集中从三处取数组合成标准报告：
        - cfg     → MODEL_PAIR / DATASET / CONFIG / STATISTICAL_CHECK
        - result  → KEY_METRICS / BEHAVIOR_CHECK / GEOMETRY_CHECK / SYSTEM_COST
        - run_dir → OUTPUT_FILES（该 task 目录下所有文件的相对路径列表）
    status 即该 task 的 gate 状态（PASS/OK/FAIL/CONDITIONAL），
    写入 ACCEPTANCE_CRITERIA.gate。
    """
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
    """CLI 主入口：解析子命令 → 加载配置 → 建 run 目录 → 跑 runner → 落产物。

    标准输出语义（供脚本/CI 消费）：
        - `[apcs] task=...` / `[apcs] run_id=...` / `[apcs] run_dir=...`
          运行头信息（开始即打印，便于确认命令被正确分发）；
        - 运行期间所有 stdout/stderr 同时 tee 到 `<run_dir>/stdout.log`（§63）；
        - 末尾 `[apcs] DONE. status=...` + 缩进的 metrics JSON（人工可读）；
        - 退出码：status ∈ {PASS, OK} → 0；FAIL/CONDITIONAL/异常 → 1。
    """
    # ---- 参数解析 ----
    # task 为位置参数（argparse choices 限定为 TASKS 的 key，非法任务直接拒绝）；
    # --config 必填（整个实验的入口配置）；--run-id 可选覆盖；
    # --no-prereg 仅对 t07 生效，跳过 PREREGISTRATION.md 生成。
    parser = argparse.ArgumentParser(prog="apcs", description="APCS 实验 CLI")
    parser.add_argument("task", choices=sorted(TASKS.keys()))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None,
                        help="覆盖默认 run_id；不传则用 config 中的 run_id")
    parser.add_argument("--no-prereg", action="store_true",
                        help="跳过 PREREGISTRATION.md 生成")
    args = parser.parse_args(argv)

    # ---- 配置解析：load_config 展开 ${...} 占位符（§64），再定 run 路径 ----
    cfg = load_config(args.config)
    # run_id 默认取 cfg.experiment.run_id；T11/T13 靠它跨 task 共享数据
    run_id = args.run_id or cfg["experiment"]["run_id"]
    base = Path(cfg["output"]["base_dir"])
    # run_root = <base>/<run_id>/，整个 experiment 共享；PREREGISTRATION.md 也放这
    run_root = ensure_run_dir(base, run_id)
    # 每个 task 的产物放在 `<run_root>/<task>/` 下，方便 T11 聚合
    run_dir = run_root / args.task
    run_dir.mkdir(parents=True, exist_ok=True)

    # §63 落一份完整 cfg（合并默认 + 用户后的最终值）到 run 目录
    write_json(run_dir / "config.json", cfg)

    module_path, func_name, title, objective = TASKS[args.task]
    print(f"[apcs] task={args.task} title={title}")
    print(f"[apcs] run_id={run_id}")
    print(f"[apcs] run_dir={run_dir}")

    fn = _import_attr(module_path, func_name)

    # §63 stdout.log：捕获 stdout/stderr 到文件
    # _tee 是双写代理：同一行同时写真实 stream（终端可见）和 log_f（落盘）。
    log_path = run_dir / "stdout.log"
    log_f = log_path.open("w", encoding="utf-8")

    @contextlib.contextmanager
    def _tee(stream):
        class _Tee:
            def write(self, s):
                stream.write(s)
                log_f.write(s)
                log_f.flush()  # 每行 flush：崩溃/超时时日志不丢

            def flush(self):
                stream.flush()
                log_f.flush()

        yield _Tee()

    # runner 契约：fn(cfg, run_dir) → dict；必需键 status，
    # 可选键 metrics / system / geometry / summary（下方按需落盘）。
    with contextlib.redirect_stdout(_tee(sys.stdout)), contextlib.redirect_stderr(_tee(sys.stderr)):
        try:
            result = fn(cfg, run_dir)
            status = result.get("status", "UNKNOWN")
        except Exception as e:  # noqa: BLE001
            # 异常先写进 stdout.log 再向外抛，保持运行记录完整
            log_f.write(f"\n[ERROR] {type(e).__name__}: {e}\n")
            log_f.flush()
            raise
    log_f.close()

    # ---- §63 按标准产物清单落盘：metrics / system / geometry / summary ----
    # metrics.json 全 task 必写；system.json 仅 T10、geometry.json 仅 T12 有值
    write_json(run_dir / "metrics.json", result.get("metrics", {}))
    if "system" in result:
        write_json(run_dir / "system.json", result["system"])
    if "geometry" in result:
        write_json(run_dir / "geometry.json", result["geometry"])
    write_text(run_dir / "summary.md", result.get("summary", ""))

    # §64/§65 metadata.json（标准字段补全）
    # 独立 try：metadata 失败只记 warn，不阻断 task 主产物（非关键路径）
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

    # §73 Task Report：汇总 12 字段，写到 task_report.md
    _write_task_report(cfg, run_dir, args.task, result, status)

    # §70 PREREGISTRATION：t07 后自动生成（冻结 Models/Dataset/…/Gates）。
    # 写在 run_root 而非 run_dir —— 它是整个 experiment 的 manifest
    if args.task == "t07" and not args.no_prereg:
        from .prereg import generate_prereg

        generate_prereg(cfg, run_root / "PREREGISTRATION.md")

    # ---- 标准输出收尾：状态行 + 指标 JSON + 退出码（PASS/OK → 0）----
    print(f"[apcs] DONE. status={status}")
    print(json.dumps(result.get("metrics", {}), indent=2, ensure_ascii=False))
    return 0 if status in {"PASS", "OK"} else 1


if __name__ == "__main__":
    sys.exit(main())