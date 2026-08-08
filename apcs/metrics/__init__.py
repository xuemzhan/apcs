"""评估指标（design.md §3 / §4 / §45 / §48 / §51）。

═══════════════════════════════════════════════════════════════════════════════
§3 论文核心指标：
    3.1 Retention      = Score_handoff / Score_student_self
    3.2 CHG            = Score_handoff - Score_student_self
    3.3 TGRR           = (S_h - S_s) / (S_t - S_s)
    3.4 PCR            = |Mapper_ours| / |Mapper_ridge|

§4 系统收益：
    PSR_A (Scenario A) = 1 - (T_map + T_load + T_query) / T_prefill_S

§45 JCR (Judge Consistency Rate)
    = handoff 与 self-prefill 决策一致次数 / N

§48 Gap Strata 报告：
    按 Teacher-Student gap 把 sample 分到 Low/Medium/High 三桶，
    每桶报告 N / Gap / CHG / TGRR / CI。

§51 统计：
    - bootstrap_ci: 非参 bootstrap 均值/中位数 CI
    - linear_cka: Kornblith et al., 2019
    - principal_angle: 最大 principal angle (弧度)
    - effective_rank: Roy & Vetterli 2007 entropy-based

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def safe_div(a: float, b: float) -> float:
    """安全除法：分母接近 0 时返回 0.0，避免 NaN 传播。"""
    return a / b if abs(b) > 1e-12 else 0.0


def retention(score_handoff: float, score_student: float) -> float:
    """§3.1 Retention = Score_handoff / Score_student_self。

    高 Retention 说明 Handoff State 可近似替代 Student State，
    但不能说明 Teacher Advantage 已迁移（设计文档 §3.1 强调）。
    """
    return safe_div(score_handoff, score_student)


def chg(score_handoff: float, score_student: float) -> float:
    """§3.2 CHG = Score_handoff - Score_student_self。

    论文最重要的科学端点：CHG > 0 才支持 Runtime Capability Transfer。
    """
    return score_handoff - score_student


def tgrr(score_handoff: float, score_student: float, score_teacher: float) -> float:
    """§3.3 TGRR = (S_h - S_s) / (S_t - S_s)。

    表示 Teacher 相对 Student 的能力差距有多少被 Student 通过 Runtime State 恢复。
    分母 ≤ 0 时返回 0（避免 NaN）。
    """
    denom = score_teacher - score_student
    return safe_div(score_handoff - score_student, denom)


def pcr(mapper_ours_params: int, mapper_ridge_params: int) -> float:
    """§3.4 PCR = |Mapper_ours| / |Mapper_ridge|。

    评价 mapper 的部署规模：<1.0 说明比 Ridge 更省参数。
    """
    return safe_div(mapper_ours_params, mapper_ridge_params)


def psr_a(t_map: float, t_load: float, t_query: float, t_prefill_student: float) -> float:
    """§4.1 Scenario A 自然 handoff 的 Prefill Saving Ratio。

    PSR_A = 1 - (T_map + T_load + T_query) / T_prefill_S
    注意：仅在 Scenario A 下有意义（B/C 禁止使用）。
    """
    return 1.0 - safe_div(t_map + t_load + t_query, t_prefill_student)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """§45 KL(p || q) 离散分布版。

    用 eps 平滑避免 log(0)。返回标量。
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def jcr(decisions_a: Iterable[int], decisions_b: Iterable[int]) -> float:
    """§45 Judge Consistency Rate。

    JCR = handoff 与 self-prefill 决策一致次数 / N
    用于 Judge / Ranker / Multi-choice preference 行为稳定性评估。
    """
    a = list(decisions_a)
    b = list(decisions_b)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine 相似度（用于 §32 attn-output cosine / §40 geometry）。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)。

    注意：R² 可为负（当模型预测比均值还差时）。§75 强调不能用 R² 替代 CHG。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - safe_div(ss_res, ss_tot)


def rms_norm(x: np.ndarray) -> float:
    """RMS（root mean square）范数，用于 §24 RMS Calibration。

    RMS = sqrt(mean(x²))
    注意与 L2 norm 不同：L2 = sqrt(sum(x²))
    """
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def effective_rank(x: np.ndarray, eps: float = 1e-12) -> float:
    """§40 / §62 Effective rank via SVD entropy (Roy & Vetterli 2007)。

    eff_rank = exp(H(p))，其中 p_i = σ_i / Σσ，H 是 Shannon 熵。
    各向同性随机矩阵的 eff_rank ≈ min(n, m)。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    s = np.linalg.svd(x, compute_uv=False)
    p = s / (s.sum() + eps)
    p = np.clip(p, eps, None)
    p = p / p.sum()
    ent = -float(np.sum(p * np.log(p)))
    return float(math.exp(ent))


def principal_angle(a: np.ndarray, b: np.ndarray) -> float:
    """§40 两组基的最大 principal angle（弧度）。

    越小说明两个子空间越对齐。§62 Fig.7b 用作 CHG 预测因子。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    prod = qa.T @ qb
    s = np.linalg.svd(prod, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.arccos(s.min()))


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """§40 / §62 Linear CKA（Kornblith et al., 2019）。

    CKA = ||a^T b||_F² / sqrt(||a^T a||_F² · ||b^T b||_F²)
    衡量两组表示的线性对齐程度，对各向同性缩放不变。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)

    num = float(np.linalg.norm(a.T @ b, ord="fro") ** 2)
    den_a = float(np.linalg.norm(a.T @ a, ord="fro") ** 2)
    den_b = float(np.linalg.norm(b.T @ b, ord="fro") ** 2)
    return safe_div(num, math.sqrt(den_a * den_b))


def bootstrap_ci(
    values: Iterable[float],
    stat: str = "mean",
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """§51 非参 bootstrap 均值/中位数 CI。

    返回 (point, ci_low, ci_high)。
    注意：传入 values 应为 paired diff（per-sample），否则 CI 无意义。
    """
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    fn = np.mean if stat == "mean" else np.median
    point = float(fn(arr))
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    samples = fn(arr[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    return point, low, high