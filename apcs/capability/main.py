"""T09 Main Capability（design.md §37 / §45 / §48 / §51）。

═══════════════════════════════════════════════════════════════════════════════
必须比较的方法（§37）：
    Student, Teacher, Text-Handoff, Ridge, Base Only, Base+Advantage, Full APCS

主指标：
    CHG (primary), TGRR, Retention, JCR (§45), KL, Token Agreement

统计规范（§51）：
    3 Seeds, Mean / Std, 95% CI, Paired Bootstrap + Paired Permutation Test

bug-5 修复：旧实现对 3 个 seed-均值做 bootstrap（n=3），CI 几乎退化为单点。
        新实现改为：每个 seed 计算 per-sample diff 列表（n=samples），再 bootstrap。
bug-6 修复：删除 main.py 中未使用的 dead import jcr → 改为真实使用 JCR。
        JCR = handoff 决策 vs Student self-prefill 决策的一致率。
bug-9 修复：从 cfg 读取 teacher/student 层数与 head 配置。
bug-9（科学诚实性）修复：移除全部内置 hash（PYTHONHASHSEED 加盐，跨进程不可复现），
        RNG 种子统一经 _stable_seed（zlib.crc32）跨进程稳定派生；每 seed 通过
        独立偏移改变得分，使不同 cfg seeds 得不同 CHG / CI / p；T09 结果自标注
        offline demo（合成数据），真实能力迁移须以 design.md §75 真实 Test 为准。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import zlib

import numpy as np

from ..data import synthetic_teacher_advantage_set
from ..io.runs import write_json
from ..metrics import bootstrap_ci, chg, jcr, retention, tgrr


# 每个 kind 的 per-seed 偏移幅度（seed 决定的真实偏移，见 _simulate_scores）
_OFFSET_SPAN = {
    "student": 0.01,
    "teacher": 0.05,
    "text": 0.01,
    "ridge": 0.01,
    "base_only": 0.01,
    "base_plus_adv": 0.02,
    "full_apcs": 0.02,
}


def _stable_seed(seed: int, kind: str, sample_id: str = "") -> int:
    """跨进程稳定的种子派生：zlib.crc32(f"{seed}:{kind}:{sample_id}") & 0x7FFFFFFF。

    禁止使用 Python 内置 hash 函数：其受 PYTHONHASHSEED 加盐，同一字符串在
    不同进程/解释器运行中哈希值不同，导致结果不可复现。所有 RNG 种子必须
    经本函数派生（crc32 与字符串编码无关，跨进程确定性一致）。
    """
    return zlib.crc32(f"{seed}:{kind}:{sample_id}".encode("utf-8")) & 0x7FFFFFFF


def _simulate_scores(seed: int, kind: str) -> float:
    """离线合成各方法得分（teacher > base+adv > base > student）。

    offline demo：得分为合成数据，不是真实实验测量值。
    每个 seed 通过 _stable_seed 派生独立偏移（span 见 _OFFSET_SPAN），叠加
    sample noise（σ≈0.05），因此不同 seed 得不同 CHG / CI / p；固定 cfg seeds
    重复运行结果完全一致（跨进程可复现）。
    """
    base = {
        "student": 0.50,
        "teacher": 0.80,
        "text": 0.55,
        "ridge": 0.58,
        "base_only": 0.60,
        "base_plus_adv": 0.66,
        "full_apcs": 0.70,
    }[kind]
    offset_rng = np.random.default_rng(_stable_seed(seed, f"offset:{kind}"))
    offset = (offset_rng.random() * 2 - 1) * _OFFSET_SPAN[kind]
    noise_rng = np.random.default_rng(_stable_seed(seed, f"noise:{kind}"))
    return float(np.clip(base + offset + noise_rng.normal(0, 0.05), 0, 1))


def _simulate_decision(seed: int, kind: str, sample_id: str) -> int:
    """离线合成决策 ID（0/1），用于 §45 JCR 计算。

    真实实现中 decision 可以是：
        - judge: prefer A or B
        - ranker: top-1 candidate index
        - multi-choice: option index
    """
    rng = np.random.default_rng(_stable_seed(seed, kind, sample_id))
    base_p = {
        "student": 0.55,
        "teacher": 0.85,
        "text": 0.58,
        "ridge": 0.60,
        "base_only": 0.62,
        "base_plus_adv": 0.68,
        "full_apcs": 0.72,
    }[kind]
    return int(rng.random() < base_p)


def _paired_permutation_test(
    a: np.ndarray, b: np.ndarray, n_perm: int = 5000, rng_seed: int = 0
) -> float:
    """§51 Paired Permutation Test。

    原假设 H0: a 与 b 同分布（mean_a - mean_b == 0）。
    在每对 (a_i, b_i) 内随机交换符号；统计量 = mean(a - b)。
    p-value = P(permuted_stat >= observed_stat | H0)。
    """
    rng = np.random.default_rng(rng_seed)
    diff = np.asarray(a) - np.asarray(b)
    observed = float(np.mean(diff))
    n = len(diff)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_stat = float(np.mean(diff * signs))
        if perm_stat >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def run_main_capability(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T09 主能力实验入口（§37）。"""
    seeds = cfg.get("seeds", [0, 1, 2])
    n_samples = int(cfg.get("t09_n_samples", 32))
    samples = synthetic_teacher_advantage_set(n=n_samples)
    methods = [
        "student",
        "teacher",
        "text",
        "ridge",
        "base_only",
        "base_plus_adv",
        "full_apcs",
    ]

    # 收集每个 (method, seed, sample_id) 的得分与决策
    # 形状：scores[method] -> list[seed][sample_id] -> (score, decision)
    scores: dict[str, list[list[float]]] = {m: [] for m in methods}
    decisions: dict[str, list[list[int]]] = {m: [] for m in methods}
    for seed in seeds:
        for m in methods:
            score_row, dec_row = [], []
            for s in samples:
                seed_for_sample = _stable_seed(seed, m, s.sample_id)
                score_row.append(_simulate_scores(seed_for_sample, m))
                dec_row.append(_simulate_decision(seed, m, s.sample_id))
            scores[m].append(score_row)
            decisions[m].append(dec_row)

    # 全局聚合（跨 seed 求 mean）
    def mean_scores_across_seeds(m: str) -> np.ndarray:
        """返回长度为 n_samples 的数组：每个 sample 在 seeds 上的均值。"""
        per_seed = np.asarray(scores[m])  # (n_seeds, n_samples)
        return per_seed.mean(axis=0)

    s_self = float(np.mean(mean_scores_across_seeds("student")))
    s_teach = float(np.mean(mean_scores_across_seeds("teacher")))
    rows = []
    for m in ["text", "ridge", "base_only", "base_plus_adv", "full_apcs"]:
        s_arr = mean_scores_across_seeds(m)
        s = float(np.mean(s_arr))
        # §45 JCR：以 student 为 reference
        jcr_val = float(
            np.mean(
                [
                    jcr(decisions[m][i], decisions["student"][i])
                    for i in range(len(seeds))
                ]
            )
        )
        rows.append(
            {
                "method": m,
                "score": s,
                "score_std": float(np.std(s_arr)),  # §51 Std
                "retention": retention(s, s_self),
                "chg": chg(s, s_self),
                "tgrr": tgrr(s, s_self, s_teach),
                "jcr_vs_student": jcr_val,  # §45
            }
        )

    # §51 Bootstrap CI on CHG（per-sample diff，bug-5 修复）
    student_arr = mean_scores_across_seeds("student")
    base_plus_adv_arr = mean_scores_across_seeds("base_plus_adv")
    diff = base_plus_adv_arr - student_arr  # (n_samples,)
    point, lo, hi = bootstrap_ci(diff.tolist(), n_boot=2000, rng=np.random.default_rng(0))
    # §51 Paired Permutation Test
    p_value = _paired_permutation_test(base_plus_adv_arr, student_arr, n_perm=2000)

    # §48 Gap strata 报告：用 sample 的 teacher-student gap 分桶
    gap_strata_report = _gap_strata_report(scores, decisions, seeds)

    metrics = {
        "task": "T09",
        "student_score": s_self,
        "teacher_score": s_teach,
        "teacher_gap": s_teach - s_self,
        "per_method": rows,
        "chg_bootstrap": {
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "ci": 0.95,
            "n_samples": len(diff),
        },
        "chg_permutation": {"p_value": p_value, "n_perm": 2000},
        "seeds": seeds,
        "n_samples_per_seed": n_samples,
        "gap_strata": gap_strata_report,  # §48
        # offline demo 自标注（科学诚实性）：得分是合成数据，不是真实测量
        "offline_demo": True,
        "note": (
            "T09 得分为合成数据（offline demo），不可作为真实能力迁移的证据；"
            "design.md §75 要求真实 Test 上 CHG>0 才支持 Runtime Capability Transfer"
        ),
    }
    write_json(run_dir / "metrics.json", metrics)
    md = (
        "# T09 Main Capability\n\n"
        "> ⚠️ **offline demo**：T09 得分为合成数据，不可作为真实能力迁移的证据；"
        "design.md §75 要求真实 Test 上 CHG>0 才支持 Runtime Capability Transfer。\n\n"
        f"- Student: {s_self:.4f}\n"
        f"- Teacher: {s_teach:.4f}\n"
        f"- Teacher gap: {s_teach - s_self:.4f}\n\n"
        "## Per-method (§37)\n"
        "| method | score | std | retention | CHG | TGRR | JCR vs Student |\n"
        "| ------ | ----: | --: | --------: | --: | ---: | -------------: |\n"
        + "\n".join(
            f"| {r['method']} | {r['score']:.4f} | {r['score_std']:.4f} | "
            f"{r['retention']:.4f} | {r['chg']:+.4f} | {r['tgrr']:+.4f} | "
            f"{r['jcr_vs_student']:.4f} |"
            for r in rows
        )
        + f"\n\n## Bootstrap CI on CHG (base+adv − student, n={len(diff)})\n"
        f"- point={point:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}]\n\n"
        f"## Paired Permutation Test (§51, n_perm=2000)\n"
        f"- p-value = {p_value:.4f}\n\n"
        f"## Gap Strata (§48)\n"
        + _gap_strata_md(gap_strata_report)
    )
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    # §7 Gate 2A 完整版：CHG>0 + bootstrap CI 下界>0 + TGRR>0
    base_plus_adv = next(r for r in rows if r["method"] == "base_plus_adv")
    gate2a = (
        "PASS"
        if (
            point > 0
            and lo > 0  # §51 95% CI 支持正向
            and base_plus_adv["tgrr"] > 0
            and p_value < 0.05
        )
        else "FAIL"
    )
    metrics["gate2a"] = gate2a
    write_json(run_dir / "metrics.json", metrics)
    return {"status": gate2a, "metrics": metrics, "summary": md}


def _gap_strata_report(
    scores: dict[str, list[list[float]]],
    decisions: dict[str, list[list[int]]],
    seeds: list[int],
) -> dict[str, dict[str, float]]:
    """§48 按 gap strata（low / medium / high）报告 Base+Adv 的 CHG / TGRR / N。

    离线模拟：用 teacher-student 得分差作为 gap 代理。
    真实实现应从 T07 的 gap_distribution 读入。
    """
    # 把 teacher / student / base_plus_adv 的每 sample 均值算出来
    student_arr = np.mean(scores["student"], axis=0)
    teacher_arr = np.mean(scores["teacher"], axis=0)
    base_plus_adv_arr = np.mean(scores["base_plus_adv"], axis=0)
    gap = teacher_arr - student_arr
    # 三等分
    quantiles = np.quantile(gap, [1 / 3, 2 / 3])
    buckets = {
        "low": gap < quantiles[0],
        "medium": (gap >= quantiles[0]) & (gap < quantiles[1]),
        "high": gap >= quantiles[1],
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        s_self = float(np.mean(student_arr[mask]))
        s_teach = float(np.mean(teacher_arr[mask]))
        s_adv = float(np.mean(base_plus_adv_arr[mask]))
        out[name] = {
            "n": int(mask.sum()),
            "student_score": s_self,
            "teacher_score": s_teach,
            "gap_mean": float(np.mean(gap[mask])),
            "base_plus_adv_chg": s_adv - s_self,
            "base_plus_adv_tgrr": (s_adv - s_self) / max(s_teach - s_self, 1e-6),
        }
    return out


def _gap_strata_md(report: dict[str, dict[str, float]]) -> str:
    lines = ["| strata | n | gap | CHG | TGRR |", "| ------ | -: | --: | --: | ---: |"]
    for name in ("low", "medium", "high"):
        if name not in report:
            continue
        r = report[name]
        if r.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | - | - | - |")
        else:
            lines.append(
                f"| {name} | {r['n']} | {r['gap_mean']:.4f} | "
                f"{r['base_plus_adv_chg']:+.4f} | {r['base_plus_adv_tgrr']:+.4f} |"
            )
    return "\n".join(lines) + "\n"