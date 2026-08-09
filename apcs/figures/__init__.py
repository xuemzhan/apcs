"""§56-§62 论文 Figure 渲染（matplotlib）。

═══════════════════════════════════════════════════════════════════════════════
Figure 1: PCR vs Retention (T04-T06)
    - 绘图类型：散点 + 变体标签（rows[].variant），红色虚线 Gate 1 阈值 0.90
Figure 2: Teacher Gap ↔ CHG / TGRR (T07/T09)
    - 绘图类型：并排双柱状图，按 Gap 分层 Low/Medium/High 三桶
Figure 3: CHG–PSR_A Pareto, only Scenario A, 右上象限才有意义 (T09/T10)
    - 绘图类型：单点散点（CHG, PSR_A），过原点十字参考线
Figure 4: Context Length Scaling (T10)
    - 绘图类型：双 Y 轴折线（Cost_B + PSR_A），X 轴 log2 刻度
Figure 5: Multi-turn Stability (T46)
    - 绘图类型：三线折线（CHG / KL / JCR）随轮次变化
Figure 6: Ablation (T47)
    - 绘图类型：柱状图；变体布局为 "ablation\nsetting" 双行标签，
      红线为 Gate 1 阈值，Y 轴固定 [0,1]
Figure 7: Geometry
    Fig.7a: Layer × Layer CKA heatmap        （viridis，逐 (student,teacher) 层对）
    Fig.7b: Principal Angle × CHG scatter
    Fig.7c: Attention-output Cosine × Retention scatter
    Fig.7d: Effective Rank × CHG scatter

输出文件名约定：fig1.png … fig7a.png / fig7bcd.png，全部 dpi=120 + bbox_inches="tight"。
每个 figN_* 函数独立幂等：数据缺失时静默返回（不抛异常、不落文件）。
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path


def _save(fig, path: Path) -> None:
    """保存 figure 到指定路径（自动建父目录）。

    输入：fig=matplotlib figure，path=目标 .png 路径。
    输出：无（副作用：写文件）。dpi=120 + bbox_inches="tight" 保证清晰且裁掉白边。
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # reports/figures/ 不存在则创建
    fig.savefig(path, dpi=120, bbox_inches="tight")


def fig1_pcr_retention(t06_metrics: dict, out_path: Path):
    """Figure 1: PCR vs Retention。

    数据源：t06_metrics["rows"]，每行 {variant, rank, pcr, retention}（T06 扫 rank 档）。
    绘图：横轴 PCR（参数压缩比），纵轴 Retention；每个变体用 annotate 标注
    variant 名（如 "lowrank-8"、"shared-basis-16"，见 §34 变体布局）。
    语义：PCR<1 且 Retention≥0.90（红色虚线）右上侧的点 = 轻量且保真的最优配置。
    """
    import matplotlib.pyplot as plt

    rows = t06_metrics.get("rows", [])
    if not rows:  # 数据缺失：静默返回，不产图
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    pcrs = [r["pcr"] for r in rows]
    rets = [r["retention"] for r in rows]
    ax.scatter(pcrs, rets, s=80)
    for r in rows:
        # 变体布局：每个数据点旁标注其变体名（rank / basis 策略）
        ax.annotate(r["variant"], (r["pcr"], r["retention"]), fontsize=8)
    ax.axhline(0.90, color="red", linestyle="--", label="Gate 1 (0.90)")
    ax.set_xlabel("PCR (Parameter Compression Ratio)")
    ax.set_ylabel("Retention")
    ax.set_title("Figure 1: PCR vs Retention")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)
    plt.close(fig)


def fig2_gap_chg(t07_metrics: dict, t09_metrics: dict, out_path: Path):
    """Figure 2: Teacher Gap ↔ CHG / TGRR。

    数据源：t09_metrics["gap_strata"]，按 Teacher−Student gap 分层的 Low/Medium/High 三桶
    （§48 Gap Strata；每桶含 base_plus_adv_chg / base_plus_adv_tgrr）。
    绘图：并排双柱状图（左 CHG、右 TGRR），颜色 蓝/灰/红 区分三层。
    语义：gap 越大的桶若 CHG/TGRR 越高，说明优势迁移来自 Teacher 领先的样本。
    """
    import matplotlib.pyplot as plt

    strata = t09_metrics.get("gap_strata", {})
    if not strata:
        return
    names = [n for n in ("low", "medium", "high") if n in strata]  # 按 §48 固定层顺序
    chgs = [strata[n].get("base_plus_adv_chg", 0) for n in names]
    tgrrs = [strata[n].get("base_plus_adv_tgrr", 0) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, chgs, color=["#88c", "#aaa", "#c88"])
    axes[0].axhline(0, color="black", linewidth=0.5)  # 零线：CHG<0 的桶即负迁移
    axes[0].set_title("CHG by Gap Stratum")
    axes[0].set_ylabel("CHG")
    axes[1].bar(names, tgrrs, color=["#88c", "#aaa", "#c88"])
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("TGRR by Gap Stratum")
    axes[1].set_ylabel("TGRR")
    _save(fig, out_path)
    plt.close(fig)


def fig3_pareto(t09_metrics: dict, t10_metrics: dict, out_path: Path):
    """Figure 3: CHG–PSR_A Pareto (only Scenario A, 右上象限才有意义)。

    数据源：t09_metrics.chg_bootstrap.point（CHG 点估计）+ t10_metrics.per_context[0].psr_a_p50
    （最短 context 档的 PSR_A p50；仅 Scenario A 定义，§38）。
    绘图：单点散点（红点，标注 "APCS"），过原点十字参考线划分四个象限。
    语义：右上象限（CHG>0 ∧ PSR_A>0）= 能力增强且省时，即论文主张的有利区域；
    其它象限结论要降级（如仅效率、仅能力或都不成立）。
    """
    import matplotlib.pyplot as plt

    chg = t09_metrics.get("chg_bootstrap", {}).get("point", 0)
    psr_a = (
        t10_metrics.get("per_context", [{}])[0].get("psr_a_p50", 0)
        if t10_metrics.get("per_context")
        else 0
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color="black", linewidth=0.5)  # PSR_A=0 分界线
    ax.axvline(0, color="black", linewidth=0.5)  # CHG=0 分界线
    ax.scatter([chg], [psr_a], s=200, color="red")
    ax.annotate(
        "APCS",
        (chg, psr_a),
        textcoords="offset points",
        xytext=(8, 8),
    )
    ax.set_xlabel("CHG")
    ax.set_ylabel("PSR_A")
    ax.set_title("Figure 3: CHG-PSR_A Pareto (Scenario A)")
    ax.grid(alpha=0.3)
    _save(fig, out_path)
    plt.close(fig)


def fig4_context_scaling(t10_metrics: dict, out_path: Path):
    """Figure 4: Context Length Scaling。

    数据源：t10_metrics["per_context"]，每行 {context, cost_b_p50, psr_a_p50}。
    绘图：双 Y 轴——左轴 Cost_B（蓝线，ms），右轴 PSR_A（红线）；X 轴 log2 刻度
    （512→1024→…→16K 等距显示）。
    语义：随 context 增长 Cost_B 是否线性/超线性上升、PSR_A 是否保持正值
    （Scenario B 成本 vs Scenario A 收益的伸缩性判断，§4/§38）。
    """
    import matplotlib.pyplot as plt

    rows = t10_metrics.get("per_context", [])
    if not rows:
        return
    ctxs = [r["context"] for r in rows]
    cost_b = [r["cost_b_p50"] for r in rows]
    psr_a = [r["psr_a_p50"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()  # 共享 X 轴的第二 Y 轴
    ax1.plot(ctxs, cost_b, "b-o", label="Cost_B (ms)")
    ax2.plot(ctxs, psr_a, "r-s", label="PSR_A")
    ax1.set_xlabel("Context length")
    ax1.set_ylabel("Cost_B (ms)", color="b")
    ax2.set_ylabel("PSR_A", color="r")
    ax1.set_title("Figure 4: Context Length Scaling")
    ax1.set_xscale("log", base=2)  # context 长度按 2 的幂分布 → log2 刻度
    _save(fig, out_path)
    plt.close(fig)


def fig5_multiturn(mt_metrics: dict, out_path: Path):
    """Figure 5: Multi-turn Stability。

    数据源：mt_metrics["per_turn"]，每行 {turn, chg_mean, kl_mean, jcr_mean}。
    绘图：三线折线（CHG 蓝、KL 绿、JCR 红）随轮次变化。
    语义：CHG 是否随多轮对话保持、KL 不漂移、JCR 不下降（§46 多轮稳定性，
    T09 行为稳定性延伸到交互场景；任何一条线明显衰减即稳定性风险）。
    """
    import matplotlib.pyplot as plt

    rows = mt_metrics.get("per_turn", [])
    if not rows:
        return
    turns = [r["turn"] for r in rows]
    chg = [r["chg_mean"] for r in rows]
    kl = [r["kl_mean"] for r in rows]
    jcr = [r["jcr_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(turns, chg, "b-o", label="CHG")
    ax.plot(turns, kl, "g-s", label="KL")
    ax.plot(turns, jcr, "r-^", label="JCR")
    ax.set_xlabel("Turns")
    ax.set_title("Figure 5: Multi-turn Stability")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_path)
    plt.close(fig)


def fig6_ablation(abl_metrics: dict, out_path: Path):
    """Figure 6: Ablation 关键项。

    数据源：abl_metrics["rows"]，每行 {ablation, setting, mean_retention}。
    变体布局：柱标签为 "ablation\nsetting" 双行（如 "RMS Calibration\non/off"），
    便于并列展示同一消融项的开关对照；只画 mean_retention 非空的条目。
    语义：去掉某组件后 retention 是否跌破 0.90（红色虚线）→ 该组件的必要性论证（§47）。
    """
    import matplotlib.pyplot as plt

    rows = abl_metrics.get("rows", [])
    rows = [r for r in rows if r.get("mean_retention") is not None]  # 跳过缺失项
    if not rows:
        return
    labels = [f"{r['ablation']}\n{r['setting']}" for r in rows]  # 双行变体标签
    vals = [r["mean_retention"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals)
    ax.axhline(0.90, color="red", linestyle="--", label="Gate 1 (0.90)")
    ax.set_ylim(0, 1)  # retention 固定 [0,1] 便于横向比较
    ax.set_title("Figure 6: Ablation")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")  # 长标签旋转避免重叠
    _save(fig, out_path)
    plt.close(fig)


def fig7a_cka_heatmap(t12_metrics: dict, out_path: Path):
    """Figure 7a: Layer × Layer CKA heatmap。

    数据源：t12_metrics["geometry"]["per_layer"]，每行
    {teacher_layer, student_layer, cka}（T12 §40 逐层对线性 CKA）。
    布局：矩阵 [student_layer × teacher_layer]，viridis 色带 [0,1]。
    语义：主对角带越亮 → Teacher/Student 层表示逐层对齐越好；
    偏移亮带提示最佳层对齐需用 §31 层映射而非逐层同名。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    geo = t12_metrics.get("geometry", {})
    rows = geo.get("per_layer", [])
    if not rows:
        return
    n_t = max(r["teacher_layer"] for r in rows) + 1  # Teacher 层数（矩阵列数）
    n_s = max(r["student_layer"] for r in rows) + 1  # Student 层数（矩阵行数）
    mat = np.zeros((n_s, n_t))
    for r in rows:
        mat[r["student_layer"], r["teacher_layer"]] = r["cka"]  # 未覆盖格保持 0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Teacher layer")
    ax.set_ylabel("Student layer")
    ax.set_title("Figure 7a: Layer × Layer CKA")
    fig.colorbar(im, ax=ax)
    _save(fig, out_path)
    plt.close(fig)


def fig7bcd_scatter(t12_metrics: dict, t05_metrics: dict, t09_metrics: dict, out_dir: Path):
    """Figure 7b/7c/7d: Principal Angle × CHG / Attn-cos × Retention / Eff-rank × CHG。

    数据源：
        per_layer rows → {principal_angle, attn_output_cosine, effective_rank_s}；
        t05_metrics.mean_retention（标量）、t09_metrics.chg_bootstrap.point（标量）。
    绘图：1×3 散点；由于 CHG/Retention 是 run 级标量而几何量是层级的，
    每个散点取标量常数值展开到各层（用于"层几何 × run 能力"的相关性目检，
    §62 Fig.7b–7d 的正式定量分析在 T12 的关联统计中完成）。
    """
    import matplotlib.pyplot as plt

    geo = t12_metrics.get("geometry", {})
    rows = geo.get("per_layer", [])
    if not rows:
        return
    chg = t09_metrics.get("chg_bootstrap", {}).get("point", 0)
    ret = t05_metrics.get("mean_retention", 0)
    pa = [r["principal_angle"] for r in rows]
    ac = [r["attn_output_cosine"] for r in rows]
    er = [r["effective_rank_s"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].scatter(pa, [chg] * len(pa))  # 7b：PA 越小（越对齐）CHG 越高
    axes[0].set_xlabel("Principal Angle")
    axes[0].set_ylabel("CHG")
    axes[0].set_title("Fig.7b: PA × CHG")
    axes[1].scatter(ac, [ret] * len(ac))  # 7c：attn 输出余弦 × Retention
    axes[1].set_xlabel("Attn-out cosine")
    axes[1].set_ylabel("Retention")
    axes[1].set_title("Fig.7c: Attn-cos × Retention")
    axes[2].scatter(er, [chg] * len(er))  # 7d：Student 有效秩 × CHG
    axes[2].set_xlabel("Effective rank (Student)")
    axes[2].set_ylabel("CHG")
    axes[2].set_title("Fig.7d: ER × CHG")
    _save(fig, out_dir / "fig7bcd.png")
    plt.close(fig)


def render_all(artifacts: dict, out_dir: Path):
    """一次性渲染全部 Figure。artifacts 是 {name: metrics_dict}。

    输入：
        artifacts: {t06, t07, t09, t10, t12, t05, multiturn, ablation} 各 task 的 metrics dict
                   （键缺失时对应 figN 函数静默跳过，不会报错）。
        out_dir:  输出目录（默认 reports/figures/），各图文件名见模块 docstring。
    输出：8 张 PNG（fig1–fig6 + fig7a + fig7bcd.png）。
    用法：render_all({"t06": ..., "t09": ...}, Path("reports/figures"))
    """
    fig1_pcr_retention(artifacts.get("t06", {}), out_dir / "fig1.png")
    fig2_gap_chg(
        artifacts.get("t07", {}), artifacts.get("t09", {}), out_dir / "fig2.png"
    )
    fig3_pareto(
        artifacts.get("t09", {}), artifacts.get("t10", {}), out_dir / "fig3.png"
    )
    fig4_context_scaling(artifacts.get("t10", {}), out_dir / "fig4.png")
    fig5_multiturn(artifacts.get("multiturn", {}), out_dir / "fig5.png")
    fig6_ablation(artifacts.get("ablation", {}), out_dir / "fig6.png")
    fig7a_cka_heatmap(artifacts.get("t12", {}), out_dir / "fig7a.png")
    fig7bcd_scatter(
        artifacts.get("t12", {}),
        artifacts.get("t05", {}),
        artifacts.get("t09", {}),
        out_dir,
    )