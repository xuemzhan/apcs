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

本模块之外的相邻指标（不在本文件，勿在本模块搜索）：
    - mean_cos / mean_kv_cosine：KV 余弦均值，见 apcs.mapper.runner（§32/T04），
      对 K/V 逐层 cosine 取平均；分 K/V 版本为 mean_cos_K / mean_cos_V。
    - token_agreement / mean_token_agreement：解码 token 级一致率，见
      apcs.replay.runner（§29/T01，Gate 0）与 apcs.mapper.runner（§32/T05），
      "Teacher 前向 vs 注入后 Student 解码逐 token 相等"占比，Gate 0 要求 == 1.0。
    - r2_kv / retention_k / retention_v：K/V 分项指标（A8 消融，§22 separate_kv）。
    本模块聚焦"标量指标计算"，token/逐层聚合在调用方 runner 完成后再喂进来。

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def safe_div(a: float, b: float) -> float:
    """安全除法：分母接近 0 时返回 0.0，避免 NaN 传播。

    所有比率型指标（retention/tgrr/pcr/psr_a/cka/r2）共用此函数；
    |b| ≤ 1e-12 时按 0 处理——代价是"分母为 0"的场景静默返回 0，
    调用方需自行判断该返回值是否有物理意义（如 psr_a 全 0 时返回 1.0）。
    """
    return a / b if abs(b) > 1e-12 else 0.0


def retention(score_handoff: float, score_student: float) -> float:
    """§3.1 Retention = Score_handoff / Score_student_self。

    含义：Handoff（APCS 注入态）的得分占 Student 自 Prefill 得分的比例——
    衡量"合成态能否近似替代 Student 原生态"，是 Gate 1（T05）的判定指标：
        retention ≥ 0.90 → PASS；0.80–0.90 → CONDITIONAL；< 0.80 → FAIL。

    高 Retention 说明 Handoff State 可近似替代 Student State，
    但不能说明 Teacher Advantage 已迁移（设计文档 §3.1 强调）。
    注意：分母为 0（student_score=0）时 safe_div 返回 0.0，而非无穷。
    """
    return safe_div(score_handoff, score_student)


def chg(score_handoff: float, score_student: float) -> float:
    """§3.2 CHG = Score_handoff - Score_student_self。

    论文最重要的科学端点：CHG > 0 才支持 Runtime Capability Transfer。
    与 retention 的差异：retention 是"替代保真度"（比值，≥0.9 即可 PASS），
    CHG 是"能力增益"（差值，必须 > 0 且 bootstrap CI 下界 > 0 才算成立，
    见 §51 / Gate 2A）。
    """
    return score_handoff - score_student


def tgrr(score_handoff: float, score_student: float, score_teacher: float) -> float:
    """§3.3 TGRR = (S_h - S_s) / (S_t - S_s)。

    表示 Teacher 相对 Student 的能力差距有多少被 Student 通过 Runtime State 恢复。
    TGRR > 0 表示朝 Teacher 方向回收了部分 gap；≈1 表示完全恢复到 Teacher 水平。
    分母 ≤ 0 时返回 0（避免 NaN；此时 Teacher 并不优于 Student，gap 无定义）。
    """
    denom = score_teacher - score_student  # Teacher−Student 原始能力差距（gap）
    return safe_div(score_handoff - score_student, denom)


def pcr(mapper_ours_params: int, mapper_ridge_params: int) -> float:
    """§3.4 PCR = |Mapper_ours| / |Mapper_ridge|。

    评价 mapper 的部署规模：<1.0 说明比 Ridge 更省参数。
    T06 Lightweight Mapper 扫 rank 档位时逐行算 PCR（row: {variant, rank, pcr, retention}），
    Figure 1 横轴即 PCR。
    """
    return safe_div(mapper_ours_params, mapper_ridge_params)


def psr_a(t_map: float, t_load: float, t_query: float, t_prefill_student: float) -> float:
    """§4.1 Scenario A 自然 handoff 的 Prefill Saving Ratio。

    PSR_A = 1 - (T_map + T_load + T_query) / T_prefill_S
    含义：省掉的 Student Prefill 时间占比。PSR_A → 1 说明迁移开销可忽略。
    注意：仅在 Scenario A 下有意义（B/C 禁止使用，§38）。
    T10 中按 context length 逐档报告 psr_a_p50（p50 为延迟分位点）。
    """
    return 1.0 - safe_div(t_map + t_load + t_query, t_prefill_student)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """§45 KL(p || q) 离散分布版。

    用 eps 平滑避免 log(0)。返回标量。
    输入：p/q 为同长度的离散概率向量（任意非负权重皆可，内部会归一化）。
    输出：KL 散度 ≥ 0；p==q 时为 0（测试保证 < 1e-6）。
    含义：Handoff 态输出分布相对 Student 自 Prefill 的偏离度（行为稳定性，T05/T09）。
    """
    p = np.asarray(p, dtype=np.float64) + eps  # eps 平滑：0 概率参与 log 时不产生 -inf
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()  # 归一化为合法分布
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def jcr(decisions_a: Iterable[int], decisions_b: Iterable[int]) -> float:
    """§45 Judge Consistency Rate。

    JCR = handoff 与 self-prefill 决策一致次数 / N
    用于 Judge / Ranker / Multi-choice preference 行为稳定性评估。
    输入：两条等长（或可截断对齐）的逐 sample 决策序列；
    输出：[0,1] 一致率，任一条为空返回 0.0。
    含义：分数不变时决策行为是否稳定（§48：accuracy 不变但 JCR 掉 → 行为漂移）。
    """
    a = list(decisions_a)
    b = list(decisions_b)
    n = min(len(a), len(b))  # 长度不一致时按短者对齐截断
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine 相似度（用于 §32 attn-output cosine / §40 geometry）。

    输入：任意形状数组（内部 reshape 为 1-D 后计算）。
    输出：[-1, 1]；零向量按 0.0 处理。
    用法：
        - §32 T04：attention output 余弦（attn_output_cosine），衡量映射前后
          attention 输出对齐程度；
        - §40/§62 Fig.7c：attn-output cosine × Retention 散点；
        - T01/T02：RoPE 往返后 logit cosine。
    注意：本函数是"逐对"余弦，逐层均值（mean_cos）由调用方 runner 聚合。
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:  # 任一侧零向量 → 方向未定义，返回 0
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)。

    输入：y_true=真值向量，y_pred=预测向量（同长）。
    输出：R² = 1 − SS_res/SS_tot。
    含义：
        - R²=1：预测完全复现真值（T04 Ridge 基线要求逐位逼近，见 §32 bug-3 聚合等价性测试）；
        - R²=0：预测等同于"恒用均值"；
        - 注意：R² 可为负（当模型预测比均值还差时）。
    §75 强调不能用 R² 替代 CHG——R² 是表示空间的重建保真度，不是下游能力。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))  # 残差平方和（拟合误差）
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))  # 总平方和（相对均值）
    return 1.0 - safe_div(ss_res, ss_tot)


def rms_norm(x: np.ndarray) -> float:
    """RMS（root mean square）范数，用于 §24 RMS Calibration。

    RMS = sqrt(mean(x²))
    注意与 L2 norm 不同：L2 = sqrt(sum(x²))
    T08 Advantage State 训练用它做 K/V 幅值校准：把合成态 RMS 对齐到
    Student 自 Prefill 的 RMS 水平（测试：RMS([3,4]) ≈ 3.536）。
    """
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def effective_rank(x: np.ndarray, eps: float = 1e-12) -> float:
    """§40 / §62 Effective rank via SVD entropy (Roy & Vetterli 2007)。

    eff_rank = exp(H(p))，其中 p_i = σ_i / Σσ，H 是 Shannon 熵。
    各向同性随机矩阵的 eff_rank ≈ min(n, m)。
    含义：KV 表示的实际自由度——秩太低说明表示退化（信息坍缩），
    §62 Fig.7d 用 Student 侧 eff_rank（effective_rank_s）作 CHG 预测因子。
    输入：任意形状数组（1-D 视为单行矩阵）。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    s = np.linalg.svd(x, compute_uv=False)  # 奇异值 σ_i
    p = s / (s.sum() + eps)  # 归一化为概率分布（eps 防除零）
    p = np.clip(p, eps, None)  # 下界裁剪：p→0 时 p·log(p)→0 的极限处理
    p = p / p.sum()  # 裁剪后重新归一化，保持合法分布
    ent = -float(np.sum(p * np.log(p)))  # Shannon 熵 H(p)
    return float(math.exp(ent))


def principal_angle(a: np.ndarray, b: np.ndarray) -> float:
    """§40 两组基的最大 principal angle（弧度）。

    越小说明两个子空间越对齐。§62 Fig.7b 用作 CHG 预测因子。
    输入：a/b 为"样本 × 维度"矩阵（1-D 视为单向量）。
    实现：QR 正交化 → 构造 Gram 矩阵 âᵀb̂ → SVD → 最小奇异值对应最大角度。
    返回：[0, π/2]（完全对齐=0，正交=π/2）。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    # a/b 的每一行是特征空间中的一个基向量/样本，因此要对
    # 转置后的列空间正交化。直接 qr(a) 只会比较“样本坐标空间”，
    # 对 (rank, hidden) PCA 基会得到失真的近零夹角。
    qa, _ = np.linalg.qr(a.T, mode="reduced")
    qb, _ = np.linalg.qr(b.T, mode="reduced")
    prod = qa.T @ qb  # 两个正交基的内积矩阵
    s = np.linalg.svd(prod, compute_uv=False)  # 奇异值 = cos(各 principal angle)
    s = np.clip(s, -1.0, 1.0)  # 数值安全裁剪：防止浮点误差导致 arccos 域外
    return float(np.arccos(s.min()))  # 最小 cos 对应最大角度


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """§40 / §62 Linear CKA（Kornblith et al., 2019）。

    CKA = ||a^T b||_F² / sqrt(||a^T a||_F² · ||b^T b||_F²)
    衡量两组表示的线性对齐程度，对各向同性缩放不变。
    输入：a/b 为"样本 × 特征"矩阵（先对列去均值中心化）。
    输出：[0, 1]；相同表示 → 1（测试 > 0.999），正交表示 → 0。
    §62 Fig.7a 逐层（student_layer × teacher_layer）算 CKA 画 heatmap。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)  # 中心化（消除各特征均值偏移）
    b = b - b.mean(axis=0, keepdims=True)

    num = float(np.linalg.norm(a.T @ b, ord="fro") ** 2)  # 交叉 Gram 的 Frobenius 范数²
    den_a = float(np.linalg.norm(a.T @ a, ord="fro") ** 2)  # 自 Gram 范数²
    den_b = float(np.linalg.norm(b.T @ b, ord="fro") ** 2)
    return safe_div(num, math.sqrt(den_a * den_b))  # 分母 0（全零矩阵）→ 0


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
    输入：values=逐 sample 统计量（如 per-sample CHG），stat ∈ {mean, median}，
          n_boot=重采样次数，ci=置信水平，rng=可复现随机源（默认种子 0）。
    输出：(点估计, CI 下界, CI 上界)。空输入 → (0, 0, 0)。
    流程：有放回抽样 n_boot 组 → 每组算统计量 → 取 (1−ci)/2 与 1−(1−ci)/2 分位数。
    Gate 2A 用 CI 下界 > 0 判定 CHG 显著为正（§7 / §51）。
    """
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    fn = np.mean if stat == "mean" else np.median  # 目标统计量（均值或中位数）
    point = float(fn(arr))  # 原始样本上的点估计
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))  # 有放回重采样索引 (n_boot, n)
    samples = fn(arr[idx], axis=1)  # 每行一次重采样的统计量
    alpha = (1.0 - ci) / 2.0  # 双侧尾部概率
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    return point, low, high


def permutation_p(
    values: Iterable[float],
    n_perm: int = 1000,
    alternative: str = "greater",
    rng: np.random.Generator | None = None,
) -> float:
    """配对符号翻转置换检验（§51 补充）：H0 = 逐样本差值的均值为 0。

    参数：
        values: 逐样本 paired diff（如 per-sample CHG = handoff - student）
        n_perm: 置换次数
        alternative: "greater"（单侧，均值 > 0）、"less"、"two-sided"
        rng: 可复现随机源（默认种子 0）
    返回：
        p 值。空输入 → 1.0。

    方法：对每条 diff 随机赋 ±号（符号翻转保持配对结构），统计置换后的
    均值分布中极端于观测均值的比例。这是配对设计的精确置换检验近似，
    与 bootstrap_ci（§51）共同构成 inject-eval 的显著性证据链。
    """
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 1.0
    obs = float(np.mean(arr))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, arr.size))
    perm_means = (arr[None, :] * signs).mean(axis=1)  # (n_perm,)
    if alternative == "greater":
        extreme = perm_means >= obs
    elif alternative == "less":
        extreme = perm_means <= obs
    else:
        extreme = np.abs(perm_means) >= abs(obs)
    # +1 修正：把观测本身计入分母，避免 p=0 的过度声明（§75 诚实性）
    return float((int(extreme.sum()) + 1) / (n_perm + 1))
