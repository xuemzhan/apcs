"""CLI 端到端测试：每个 task 都能跑通且产出标准产物。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apcs.cli import main as cli_main

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pair_qwen3.yaml"


def _setup_run(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    cfg["output"]["base_dir"] = str(tmp_path / "runs")
    cfg_path = tmp_path / "pair.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _run(task: str, cfg_path: Path):
    rc = cli_main([task, "--config", str(cfg_path)])
    # OK / PASS 也接受；FAIL/CONDTIONAL 视为跑完但 gate 失败
    return rc


def _read(run_dir_root: Path, task: str, file_name: str):
    p = run_dir_root / "runs" / "qwen3-4b-to-1.7b-20260808-122727" / task / file_name
    if p.exists():
        return p.read_text(encoding="utf-8")
    # 兼容任意 run_id 子目录
    for run in (run_dir_root / "runs").iterdir():
        if (run / task / file_name).exists():
            return (run / task / file_name).read_text(encoding="utf-8")
    return None


def test_cli_t00_emits_model_compatibility(tmp_path):
    cfg_path = _setup_run(tmp_path)
    _run("t00", cfg_path)
    txt = _read(tmp_path, "t00", "model_compatibility.json")
    assert txt is not None
    payload = json.loads(txt)
    assert "compatibility" in payload
    assert "verdict" in payload["compatibility"]


def test_cli_t02_rope_passes(tmp_path):
    cfg_path = _setup_run(tmp_path)
    rc = _run("t02", cfg_path)
    assert rc == 0  # T02 cosine > 0.9999


def test_cli_t07_triggers_preregistration(tmp_path):
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