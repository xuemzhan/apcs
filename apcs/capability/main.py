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


# 每个 kind 的 per-seed 偏移幅度（seed 决定的真实偏移，见 _simulate_scores）。
# 幅度含义 = 跨 seed 波动的量级：
#   - teacher 幅度最大（0.05）：Teacher 强，但不同 seed 表现波动也大；
#   - full_apcs / base_plus_adv 次之（0.02）：handoff 态部分继承 Teacher 波动；
#   - student / text / ridge / base_only 最小（0.01）：弱模型波动较小。
# 目的是让不同 cfg seeds 产生不同 CHG / CI / p，又不至于打乱
# teacher > adv > base > student 的相对顺序（否则 TGRR/CHG 语义失真）。
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

    kind 用于区分不同 RNG 通道（如 "offset:<kind>" 与 "noise:<kind>"），
    保证同一 seed 下各通道的噪声互相独立。
    """
    # crc32 输出 32 位；& 0x7FFFFFFF 取非负，兼容 numpy RNG 的正整数种子要求
    return zlib.crc32(f"{seed}:{kind}:{sample_id}".encode("utf-8")) & 0x7FFFFFFF


def _simulate_scores(seed: int, kind: str) -> float:
    """离线合成各方法得分（teacher > base+adv > base > student）。

    offline demo：得分为合成数据，不是真实实验测量值。
    base 表固定各方法的期望分（§37 相对顺序）；offset 通道（span 见
    _OFFSET_SPAN）给出跨 seed 的真实差异；noise 通道（σ≈0.05）模拟样本内
    方差。因此不同 seed 得不同 CHG / CI / p；固定 cfg seeds 重复运行结果
    完全一致（跨进程可复现）。
    """
    # §37 各方法期望得分：teacher(0.80) > full_apcs(0.70) > base_plus_adv(0.66)
    # > base_only(0.60) > ridge(0.58) > text(0.55) > student(0.50)
    base = {
        "student": 0.50,
        "teacher": 0.80,
        "text": 0.55,
        "ridge": 0.58,
        "base_only": 0.60,
        "base_plus_adv": 0.66,
        "full_apcs": 0.70,
    }[kind]
    # offset：seed 级平移，使不同 seeds 有不同均值（真实实验 = 数据随机抽取的效应）
    offset_rng = np.random.default_rng(_stable_seed(seed, f"offset:{kind}"))
    offset = (offset_rng.random() * 2 - 1) * _OFFSET_SPAN[kind]
    # noise：样本内波动（真实实验 = 模型随机性 / 采样噪声）
    noise_rng = np.random.default_rng(_stable_seed(seed, f"noise:{kind}"))
    # clip 到 [0,1]：得分是比例/准确率语义，不能越界
    return float(np.clip(base + offset + noise_rng.normal(0, 0.05), 0, 1))


def _simulate_decision(seed: int, kind: str, sample_id: str) -> int:
    """离线合成决策 ID（0/1），用于 §45 JCR 计算。

    真实实现中 decision 可以是：
        - judge: prefer A or B
        - ranker: top-1 candidate index
        - multi-choice: option index
    这里统一为伯努利采样：P(kind) 由 base_p 给定（teacher 0.85 > full_apcs
    0.72 > base_plus_adv 0.68 > student 0.55），决策与样本绑定（sample_id 参与
    种子派生），跨进程可复现。
    """
    rng = np.random.default_rng(_stable_seed(seed, kind, sample_id))
    # §45 决策先验：方法越强，给出"正确"决策的概率越高
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
    统计量 = mean(a - b)（配对差值均值）；置换 = 每对 (a_i, b_i) 内以 1/2
    概率交换符号，得到 H0 下的统计量分布。p-value = P(permuted_stat ≥ observed | H0)。
    采用单侧：只关心 handoff 是否显著优于 student（CHG > 0 方向），
    不检验反向优势。分子分母各 +1 做平滑，保证 p 永不为 0（保守估计）。
    """
    rng = np.random.default_rng(rng_seed)
    diff = np.asarray(a) - np.asarray(b)  # 配对差值（handoff − student）
    observed = float(np.mean(diff))       # 观测统计量
    n = len(diff)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)  # 随机翻符号 = 随机交换配对顺序
        perm_stat = float(np.mean(diff * signs))  # H0 下的统计量
        if perm_stat >= observed:                 # 比观测值更极端 → 计入
            count += 1
    return (count + 1) / (n_perm + 1)  # +1 平滑：p ∈ (0, 1]，不为 0


def run_main_capability(cfg: dict[str, Any], run_dir) -> dict[str, Any]:
    """T09 主能力实验入口（§37）。

    结构：
        1. 读取 seeds / n_samples，生成 synthetic teacher-advantage 样本（§16 分层）；
        2. 对 7 种方法逐 (seed, sample) 模拟得分与决策（§37 方法集）；
        3. 跨 seed 聚合 → 逐方法算 score / retention / CHG / TGRR / JCR；
        4. §51 统计：per-sample diff 的 bootstrap CI + paired permutation test；
        5. §48 Gap Strata 报告；§7 Gate 2A 判定；落 metrics.json + summary.md。
    """
    seeds = cfg.get("seeds", [0, 1, 2])  # §51 固定 3 seeds（默认）
    n_samples = int(cfg.get("t09_n_samples", 32))
    samples = synthetic_teacher_advantage_set(n=n_samples)
    # §37 必须比较的 7 种方法（顺序即论文报告顺序）
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
                # 得分/决策的种子都经 _stable_seed 派生：跨进程可复现
                seed_for_sample = _stable_seed(seed, m, s.sample_id)
                score_row.append(_simulate_scores(seed_for_sample, m))
                dec_row.append(_simulate_decision(seed, m, s.sample_id))
            scores[m].append(score_row)
            decisions[m].append(dec_row)

    # 全局聚合（跨 seed 求 mean）：先对 seed 轴平均，再跨 sample 求总均值
    def mean_scores_across_seeds(m: str) -> np.ndarray:
        """返回长度为 n_samples 的数组：每个 sample 在 seeds 上的均值。

        先对 seed 轴平均再算指标，避免把同一样本的多次重复采样当作独立样本
        （旧实现对 3 个 seed-均值做 bootstrap，n=3，CI 几乎退化为单点，见 bug-5）。
        """
        per_seed = np.asarray(scores[m])  # (n_seeds, n_samples)
        return per_seed.mean(axis=0)

    s_self = float(np.mean(mean_scores_across_seeds("student")))   # Score_student_self（基线）
    s_teach = float(np.mean(mean_scores_across_seeds("teacher")))  # Score_teacher（上限参考）
    rows = []
    for m in ["text", "ridge", "base_only", "base_plus_adv", "full_apcs"]:
        s_arr = mean_scores_across_seeds(m)
        s = float(np.mean(s_arr))
        # §45 JCR：以 student 为 reference（handoff 决策与 Student 自 prefill 决策的一致率）
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
                "score_std": float(np.std(s_arr)),  # §51 Std（跨 sample）
                "retention": retention(s, s_self),  # §3.1 替代保真度（比值）
                "chg": chg(s, s_self),              # §3.2 首要科学端点：>0 才支持能力迁移
                "tgrr": tgrr(s, s_self, s_teach),   # §3.3 Teacher gap 恢复率
                "jcr_vs_student": jcr_val,          # §45 行为一致性
            }
        )

    # §51 Bootstrap CI on CHG（per-sample diff，bug-5 修复）
    # 用最强候选 base+adv 与 student 的 per-sample 差值做非参 bootstrap：
    # 对 n=32 个 diff 有放回重采样 2000 次，取 2.5%/97.5% 分位 → CI 下界 >0 才有统计学支持
    student_arr = mean_scores_across_seeds("student")
    base_plus_adv_arr = mean_scores_across_seeds("base_plus_adv")
    diff = base_plus_adv_arr - student_arr  # (n_samples,) 逐样本 CHG
    point, lo, hi = bootstrap_ci(diff.tolist(), n_boot=2000, rng=np.random.default_rng(0))
    # §51 Paired Permutation Test：置换检验的 p（H0: CHG ≤ 0）
    p_value = _paired_permutation_test(base_plus_adv_arr, student_arr, n_perm=2000)

    # §48 Gap strata 报告：用 sample 的 teacher-student gap 分桶
    gap_strata_report = _gap_strata_report(scores, decisions, seeds)

    metrics = {
        "task": "T09",
        "student_score": s_self,
        "teacher_score": s_teach,
        "teacher_gap": s_teach - s_self,  # §16 Teacher 能力差（TGRR 分母）
        "per_method": rows,
        # §51 CHG 的 bootstrap 统计（bug-5：per-sample 重采样，n=样本数而非 seed 数）
        "chg_bootstrap": {
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "ci": 0.95,
            "n_samples": len(diff),
        },
        # §51 置换检验结果（Gate 2A 判定所需的 p<0.05 依据）
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
    # §7 Gate 2A 完整版：CHG>0 + bootstrap CI 下界>0 + TGRR>0 + permutation 显著
    base_plus_adv = next(r for r in rows if r["method"] == "base_plus_adv")
    gate2a = (
        "PASS"
        if (
            point > 0                     # 点估计 CHG > 0
            and lo > 0                    # §51 95% CI 下界 > 0（统计稳健）
            and base_plus_adv["tgrr"] > 0  # 朝 Teacher 方向恢复了部分 gap
            and p_value < 0.05            # 置换检验显著（§51）
        )
        else "FAIL"
    )
    metrics["gate2a"] = gate2a
    write_json(run_dir / "metrics.json", metrics)  # 补写 gate2a 字段（第二次落盘）
    return {"status": gate2a, "metrics": metrics, "summary": md}


def _gap_strata_report(
    scores: dict[str, list[list[float]]],
    decisions: dict[str, list[list[int]]],
    seeds: list[int],
) -> dict[str, dict[str, float]]:
    """§48 按 gap strata（low / medium / high）报告 Base+Adv 的 CHG / TGRR / N。

    离线模拟：用 teacher-student 得分差作为 gap 代理。
    真实实现应从 T07 的 gap_distribution 读入。

    gap 按三分位切分（而不是 T07 的固定阈值 0.10/0.25），保证每桶都有样本，
    便于观察"Teacher 优势越大 → CHG 越大"的分层趋势（§48 主图之一）。
    """
    # 把 teacher / student / base_plus_adv 的每 sample 均值算出来
    student_arr = np.mean(scores["student"], axis=0)
    teacher_arr = np.mean(scores["teacher"], axis=0)
    base_plus_adv_arr = np.mean(scores["base_plus_adv"], axis=0)
    gap = teacher_arr - student_arr  # 每样本的 Teacher−Student 能力差（gap 代理）
    # 三等分：按 gap 分布取 1/3、2/3 分位作为桶边界
    quantiles = np.quantile(gap, [1 / 3, 2 / 3])
    buckets = {
        "low": gap < quantiles[0],
        "medium": (gap >= quantiles[0]) & (gap < quantiles[1]),
        "high": gap >= quantiles[1],
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            out[name] = {"n": 0}  # 空桶：只记 n=0，避免后续除零
            continue
        s_self = float(np.mean(student_arr[mask]))
        s_teach = float(np.mean(teacher_arr[mask]))
        s_adv = float(np.mean(base_plus_adv_arr[mask]))
        out[name] = {
            "n": int(mask.sum()),
            "student_score": s_self,
            "teacher_score": s_teach,
            "gap_mean": float(np.mean(gap[mask])),
            # §3.2 CHG：本桶内 base+adv 相对 student 的能力增益
            "base_plus_adv_chg": s_adv - s_self,
            # §3.3 TGRR：分母 max(gap, 1e-6) 防止 gap≈0 的桶出现除零
            "base_plus_adv_tgrr": (s_adv - s_self) / max(s_teach - s_self, 1e-6),
        }
    return out


def _gap_strata_md(report: dict[str, dict[str, float]]) -> str:
    """把 §48 Gap Strata 报告渲染成 Markdown 表格行。

    空桶（n==0）渲染为占位横线，保证三桶行数固定、表格对齐；
    每行输出 n / gap / CHG / TGRR 四个列。
    """
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