"""§70 Pre-registration manifest 生成器。

═══════════════════════════════════════════════════════════════════════════════
论文要求在 Test 前生成 PREREGISTRATION.md 冻结以下内容：
    - Models        (Teacher + Student + commit)
    - Dataset       (Primary Benchmark + splits)
    - Hyperparams   (Rank candidates, top-k, α_max, loss range)
    - Seeds
    - Statistics    (primary metric, test, CI)
    - Gates         (Retention, CHG, Behavior, PSR_A)
    - Negative Result Policy (CHG≤0 不重新筛 test dataset)

本模块读取 cfg，生成 PREREGISTRATION.md。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any


def _gate_number(v: Any) -> Any:
    """把 Gate 阈值统一为数值渲染（§70 design.md: CHG > 0 / PSR_A > 0 是数值阈值）。

    - bool（True/False 开关）→ 0：布尔标志只表示该 Gate 是否启用，
      阈值语义是数值 0（"> 0"），渲染 `> True`/`> False` 是 B5 字面量 bug；
    - int/float → 原样；
    - None → 0（未配置时按设计阈值 CHG > 0 占位）；
    - 其他（如字符串描述）→ 原样 str(v)。
    """
    if isinstance(v, bool):
        return 0
    if v is None:
        return 0
    return v


def generate_prereg(cfg: dict[str, Any], output_path: Path) -> None:
    """生成 PREREGISTRATION.md 写到 output_path。"""
    teacher = cfg["teacher"]
    student = cfg["student"]
    datasets = cfg.get("datasets", {})
    mapper = cfg.get("mapper", {})
    advantage = cfg.get("advantage", {})
    stats = cfg.get("statistics", {})
    gates = cfg.get("gates", {})

    lines = [
        "# PREREGISTRATION.md",
        "",
        f"_Generated at: {_dt.datetime.now().isoformat()}_",
        "",
        "## 1. Models",
        "",
        f"- Teacher: `{teacher.get('model_id')}` @ `{teacher.get('revision')}`",
        f"  - dtype: `{teacher.get('dtype')}`",
        f"  - attention: `{teacher.get('attention_implementation')}`",
        f"- Student: `{student.get('model_id')}` @ `{student.get('revision')}`",
        f"  - dtype: `{student.get('dtype')}`",
        f"  - attention: `{student.get('attention_implementation')}`",
        f"  - freeze: `{student.get('freeze')}`",
        "",
        "## 2. Dataset",
        "",
        f"- Primary teacher-advantage set: `{datasets.get('teacher_advantage', {}).get('primary')}`",
        f"- Splits: `{datasets.get('teacher_advantage', {}).get('splits')}`",
        f"- Fidelity set: `{datasets.get('fidelity')}`",
        f"- Long-context set: `{datasets.get('long_context')}`",
        f"- Behavior-sensitive set: `{datasets.get('behavior_sensitive')}`",
        "",
        "## 3. Hyperparameters",
        "",
        f"- Rank candidates: 8 / 16 / 32 (A1)",
        f"- source_top_k: `{mapper.get('source_top_k')}`",
        f"- α_max: `{mapper.get('alpha_max')}` (Bounded, A10)",
        f"- de-RoPE: `{mapper.get('de_rope')}` (A6)",
        f"- shared basis: `{mapper.get('shared_basis')}` (A2)",
        f"- separate K/V: `{mapper.get('separate_kv')}` (A8)",
        f"- advantage rank: `{advantage.get('rank')}`",
        f"- advantage separate KV: K=`{advantage.get('key')}`, V=`{advantage.get('value')}`",
        f"- RMS calibration: `{advantage.get('rms_calibration')}` (A9)",
        f"- Query gate: `{cfg.get('query_gate', {}).get('enabled')}` (A11)",
        "",
        "## 4. Seeds",
        "",
        f"- seeds: `{cfg.get('seeds')}`",
        "",
        "## 5. Statistics (§51)",
        "",
        f"- bootstrap_n: `{stats.get('bootstrap_n')}`",
        f"- CI: `{stats.get('ci')}`",
        f"- paired_test: `{stats.get('paired_test')}`",
        "",
        "## 6. Gates (§5, §6, §7)",
        "",
        f"- Gate 0 (Self-KV Replay): all samples PASS",
        f"- Gate 1 (Retention): `≥ {_gate_number(gates.get('retention_min'))}` PASS",
        f"                     `≥ {_gate_number(gates.get('retention_strong'))}` STRONG",
        f"- Gate 2A (CHG): `> {_gate_number(gates.get('chg_positive'))}` AND bootstrap CI lower > 0 AND TGRR > 0",
        f"- Gate 2A (PSR_A): `> {_gate_number(gates.get('psr_a_positive'))}`",
        "",
        "## 7. Negative Result Policy",
        "",
        "CHG ≤ 0 时 **不重新筛选 test dataset**；保留全部负向结果。",
        "若 Replacement 不稳定（Retention < 0.80），停止 Capability Transfer 扩展，",
        "优先研究 Alignment / Direction Asymmetry / Geometry。",
        "",
        "## 8. Forbidden (§52)",
        "",
        "1. Student 在主实验中重新读取 X（除 calibration / eval 阶段）。",
        "2. 微调 Student 主体后仍称 Runtime State Transfer。",
        "3. Test 调参。",
        "4. 只挑 Teacher-win Test Sample。",
        "5. 隐藏 Teacher Prefill 成本。",
        "6. 隐藏 H2D / Cache Load。",
        "7. 用 R² / Cosine / CKA 代替 CHG。",
        "8. Cache 注入失败后 Silent Re-prefill。",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")