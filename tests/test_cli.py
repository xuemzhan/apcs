"""CLI 端到端测试：每个 task 都能跑通且产出标准产物（真实 configs/pair_qwen3.yaml）。

覆盖：
    t00（§28）：产出 model_compatibility.json 且含 verdict
    t02（§30）：RoPE round-trip 精度达标 → CLI 返回 rc==0
    t07（§70）：run_root 下自动生成 PREREGISTRATION.md（Models/Gates/Negative Result Policy）
    t11（§39/§69）：bug-6 修复回归 —— run_id 必须从 run_dir.parent.name 提取，
        在共享 run_id 根目录精确定位 T05/T09/T10 产物，禁止 mtime fallback
        误读其他 run；无 task 子目录时回退用 run_dir.name。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apcs.cli import main as cli_main
from apcs.decision.runner import run_mvp_decision

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pair_qwen3.yaml"


def _setup_run(tmp_path: Path) -> Path:
    """把真实 pair_qwen3.yaml 的 base_dir 重定向到 tmp_path/runs 并落盘为可跑配置。"""
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    cfg["output"]["base_dir"] = str(tmp_path / "runs")
    cfg_path = tmp_path / "pair.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _run(task: str, cfg_path: Path):
    """调用 CLI 入口跑单个 task；FAIL/CONDITIONAL 视为"跑完但 gate 失败"，不属异常。"""
    rc = cli_main([task, "--config", str(cfg_path)])
    # OK / PASS 也接受；FAIL/CONDTIONAL 视为跑完但 gate 失败
    return rc


def _read(run_dir_root: Path, task: str, file_name: str):
    """读取 run 产物文本：优先按固定 run_id 路径，否则兼容任意 run_id 子目录。"""
    p = run_dir_root / "runs" / "qwen3-4b-to-1.7b-20260808-122727" / task / file_name
    if p.exists():
        return p.read_text(encoding="utf-8")
    # 兼容任意 run_id 子目录
    for run in (run_dir_root / "runs").iterdir():
        if (run / task / file_name).exists():
            return (run / task / file_name).read_text(encoding="utf-8")
    return None


def test_cli_t00_emits_model_compatibility(tmp_path):
    """§28 T00 端到端：跑通并产出 model_compatibility.json（含 verdict）。"""
    cfg_path = _setup_run(tmp_path)
    _run("t00", cfg_path)
    txt = _read(tmp_path, "t00", "model_compatibility.json")
    assert txt is not None
    payload = json.loads(txt)
    assert "compatibility" in payload
    assert "verdict" in payload["compatibility"]


def test_cli_t02_rope_passes(tmp_path):
    """§30 T02 端到端：RoPE round-trip 精度达标 → CLI 返回 rc==0。"""
    cfg_path = _setup_run(tmp_path)
    rc = _run("t02", cfg_path)
    assert rc == 0  # T02 cosine > 0.9999


def test_cli_t07_triggers_preregistration(tmp_path):
    """§70 T07 端到端：run_root 下自动生成 PREREGISTRATION.md（含 Models/Gates/Negative Result Policy）。"""
    cfg_path = _setup_run(tmp_path)
    _run("t07", cfg_path)
    # PREREGISTRATION.md 应该在 run_root 下
    found = False
    for run in (tmp_path / "runs").iterdir():
        if (run / "PREREGISTRATION.md").exists():
            text = (run / "PREREGISTRATION.md").read_text(encoding="utf-8")
            assert "PREREGISTRATION.md" in text
            assert "Models" in text
            assert "Gates" in text
            assert "Negative Result Policy" in text
            found = True
    assert found, "PREREGISTRATION.md 未生成"


def test_cli_t11_finds_shared_run_id(tmp_path):
    """T11 在共享 run_id 下能正确读到 T05/T09/T10。"""
    cfg_path = _setup_run(tmp_path)
    # 注入 stub 数据
    base_dir = tmp_path / "runs"
    # 跑 t00 先创建一个 run 目录
    _run("t00", cfg_path)
    # 找到这个 run 并注入 stub metrics
    for run in base_dir.iterdir():
        if not run.is_dir():
            continue
        for task, payload in [
            ("t05", {"mean_retention": 0.92, "gate1": "PASS"}),
            (
                "t09",
                {
                    "per_method": [
                        {"method": "base_plus_adv", "chg": 0.16, "tgrr": 0.53}
                    ],
                    "chg_bootstrap": {"point": 0.16, "ci_low": 0.05, "ci_high": 0.27},
                },
            ),
        ]:
            (run / task).mkdir(exist_ok=True)
            (run / task / "metrics.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        (run / "t10").mkdir(exist_ok=True)
        (run / "t10" / "system.json").write_text(
            json.dumps({"per_context": [{"psr_a_p50": 0.35}]}), encoding="utf-8"
        )
        break

    rc = _run("t11", cfg_path)
    # OK 或 PASS 都算成功；不应再误判 D_STOP
    # 找到 t11 的 metrics.json
    for run in base_dir.iterdir():
        if not run.is_dir():
            continue
        m = run / "t11" / "metrics.json"
        if m.exists():
            payload = json.loads(m.read_text(encoding="utf-8"))
            assert payload["verdict"] in {
                "A_RUNTIME_CAPABILITY_TRANSFER",
                "B_EFFICIENT_STATE_HANDOFF",
                "C_MECHANISM_BOUNDARY",
                "C_INCONCLUSIVE",
            }
            # 不能再误判 D_STOP
            assert payload["verdict"] != "D_STOP_REPLACEABILITY_UNSTABLE"
            assert payload["sources"]["t05_found"]
            assert payload["sources"]["t09_found"]
            assert payload["sources"]["t10_found"]
            return
    pytest.fail("T11 未产出 metrics.json")


# ── bug-6 修复：T11 的 run_id 应从 run_dir.parent.name（run_id 根目录）提取 ──
# 命名约定（design.md §39 / runner.py docstring §69 L16-18）：run_id 形如
# `<name>-<timestamp>`，一个 experiment 内所有 task 共享同一 run_id，报告根为
# `<base>/<run_id>/`。CLI 把 run_dir 建为 `<base>/<run_id>/<task>/`，因此
# run_id 是 run_dir.parent.name，而不是 run_dir.name（即 "t11"）。


def _write_t11_stub_run(
    base: Path,
    run_id: str,
    *,
    retention: float,
    chg: float,
    tgrr: float,
    psr_a: float,
) -> None:
    """在 base/<run_id>/ 下写入 T05/T09/T10 stub 数据（T11 判定所需）。"""
    run_root = base / run_id
    (run_root / "t05").mkdir(parents=True, exist_ok=True)
    (run_root / "t05" / "metrics.json").write_text(
        json.dumps({"mean_retention": retention}), encoding="utf-8"
    )
    (run_root / "t09").mkdir(parents=True, exist_ok=True)
    (run_root / "t09" / "metrics.json").write_text(
        json.dumps(
            {
                "per_method": [
                    {"method": "base_plus_adv", "chg": chg, "tgrr": tgrr}
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_root / "t10").mkdir(parents=True, exist_ok=True)
    (run_root / "t10" / "system.json").write_text(
        json.dumps({"per_context": [{"psr_a_p50": psr_a}]}), encoding="utf-8"
    )


def test_cli_t11_reads_exact_run_id_single_run(tmp_path):
    """T11 的 run_dir 是 <base>/<run_id>/t11/：应精确读到该 run_id 的产物。"""
    base = tmp_path / "base"
    run_id = "myexp-20260101-000000"
    _write_t11_stub_run(base, run_id, retention=0.95, chg=0.2, tgrr=0.1, psr_a=0.5)
    run_dir = base / run_id / "t11"
    run_dir.mkdir(parents=True)
    res = run_mvp_decision({"output": {"base_dir": str(base)}}, run_dir)
    m = res["metrics"]
    assert m["sources"]["t05_found"] is True
    assert m["sources"]["t09_found"] is True
    assert m["sources"]["t10_found"] is True
    assert m["retention"] == 0.95
    assert m["verdict"] == "A_RUNTIME_CAPABILITY_TRANSFER"


def test_cli_t11_uses_exact_run_id_not_mtime_fallback(tmp_path):
    """多 run 时必须按 run_id 精确查找，不能因 mtime fallback 读到别的 run。

    Given: 两个 run，旧 run retention=0.95（A），新 run retention=0.2（D）
    When:  对旧 run 的 t11 目录执行 run_mvp_decision
    Then:  读到的是旧 run 的数据（retention=0.95, verdict=A）
    """
    import os
    import time

    base = tmp_path / "base"
    old_id = "myexp-20260101-000000"
    new_id = "myexp-20260202-000000"
    _write_t11_stub_run(base, old_id, retention=0.95, chg=0.2, tgrr=0.1, psr_a=0.5)
    _write_t11_stub_run(base, new_id, retention=0.2, chg=0.0, tgrr=0.0, psr_a=0.0)
    # 强制让新 run 目录 mtime 更新：若代码走 mtime fallback 一定会先选它
    t = time.time()
    os.utime(base / old_id, (t, t))
    os.utime(base / new_id, (t + 10, t + 10))

    run_dir = base / old_id / "t11"
    run_dir.mkdir(parents=True)
    res = run_mvp_decision({"output": {"base_dir": str(base)}}, run_dir)
    m = res["metrics"]
    assert m["sources"]["t05_found"] is True
    # 精确按 run_id 定位 → 读到旧 run（A）；mtime fallback 会读到新 run → D_STOP
    assert m["retention"] == 0.95
    assert m["verdict"] == "A_RUNTIME_CAPABILITY_TRANSFER"


def test_cli_t11_fallback_to_run_dir_name_without_task_subdir(tmp_path):
    """run_dir 直接是 run_id 根目录（无 task 子目录）时，从 run_dir.name 提取。"""
    base = tmp_path / "base"
    run_id = "myexp-20260101-000000"
    _write_t11_stub_run(base, run_id, retention=0.95, chg=0.2, tgrr=0.1, psr_a=0.5)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    res = run_mvp_decision({"output": {"base_dir": str(base)}}, run_dir)
    assert res["metrics"]["sources"]["t05_found"] is True
    assert res["metrics"]["retention"] == 0.95