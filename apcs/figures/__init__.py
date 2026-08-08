"""§56-§62 论文 Figure 渲染（matplotlib）。

═══════════════════════════════════════════════════════════════════════════════
Figure 1: PCR vs Retention (T04-T06)
Figure 2: Teacher Gap ↔ CHG / TGRR (T07/T09)
Figure 3: CHG–PSR_A Pareto, only Scenario A, 右上象限才有意义 (T09/T10)
Figure 4: Context Length Scaling (T10)
Figure 5: Multi-turn Stability (T46)
Figure 6: Ablation (T47)
Figure 7: Geometry
    Fig.7a: Layer × Layer CKA heatmap
    Fig.7b: Principal Angle × CHG scatter
    Fig.7c: Attention-output Cosine × Retention scatter
    Fig.7d: Effective Rank × CHG scatter

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def fig1_pcr_retention(t06_metrics: dict, out_path: Path):
    """Figure 1: PCR vs Retention。"""
    import matplotlib.pyplot as plt

    rows = t06_metrics.get("rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    pcrs = [r["pcr"] for r in rows]
    rets = [r["retention"] for r in rows]
    ax.scatter(pcrs, rets, s=80)
    for r in rows:
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
    """Figure 2: Teacher Gap ↔ CHG / TGRR。"""
    import matplotlib.pyplot as plt

    strata = t09_metrics.get("gap_strata", {})
    if not strata:
        return
    names = [n for n in ("low", "medium", "high") if n in strata]
    chgs = [strata[n].get("base_plus_adv_chg", 0) for n in names]
    tgrrs = [strata[n].get("base_plus_adv_tgrr", 0) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, chgs, color=["#88c", "#aaa", "#c88"])
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title("CHG by Gap Stratum")
    axes[0].set_ylabel("CHG")
    axes[1].bar(names, tgrrs, color=["#88c", "#aaa", "#c88"])
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("TGRR by Gap Stratum")
    axes[1].set_ylabel("TGRR")
    _save(fig, out_path)
    plt.close(fig)


def fig3_pareto(t09_metrics: dict, t10_metrics: dict, out_path: Path):
    """Figure 3: CHG–PSR_A Pareto (only Scenario A, 右上象限才有意义)。"""
    import matplotlib.pyplot as plt

    chg = t09_metrics.get("chg_bootstrap", {}).get("point", 0)
    psr_a = (
        t10_metrics.get("per_context", [{}])[0].get("psr_a_p50", 0)
        if t10_metrics.get("per_context")
        else 0
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
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
    """Figure 4: Context Length Scaling。"""
    import matplotlib.pyplot as plt

    rows = t10_metrics.get("per_context", [])
    if not rows:
        return
    ctxs = [r["context"] for r in rows]
    cost_b = [r["cost_b_p50"] for r in rows]
    psr_a = [r["psr_a_p50"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(ctxs, cost_b, "b-o", label="Cost_B (ms)")
    ax2.plot(ctxs, psr_a, "r-s", label="PSR_A")
    ax1.set_xlabel("Context length")
    ax1.set_ylabel("Cost_B (ms)", color="b")
    ax2.set_ylabel("PSR_A", color="r")
    ax1.set_title("Figure 4: Context Length Scaling")
    ax1.set_xscale("log", base=2)
    _save(fig, out_path)
    plt.close(fig)


def fig5_multiturn(mt_metrics: dict, out_path: Path):
    """Figure 5: Multi-turn Stability。"""
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
    """Figure 6: Ablation 关键项。"""
    import matplotlib.pyplot as plt

    rows = abl_metrics.get("rows", [])
    rows = [r for r in rows if r.get("mean_retention") is not None]
    if not rows:
        return
    labels = [f"{r['ablation']}\n{r['setting']}" for r in rows]
    vals = [r["mean_retention"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals)
    ax.axhline(0.90, color="red", linestyle="--", label="Gate 1 (0.90)")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 6: Ablation")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    _save(fig, out_path)
    plt.close(fig)


def fig7a_cka_heatmap(t12_metrics: dict, out_path: Path):
    """Figure 7a: Layer × Layer CKA heatmap。"""
    import matplotlib.pyplot as plt
    import numpy as np

    geo = t12_metrics.get("geometry", {})
    rows = geo.get("per_layer", [])
    if not rows:
        return
    n_t = max(r["teacher_layer"] for r in rows) + 1
    n_s = max(r["student_layer"] for r in rows) + 1
    mat = np.zeros((n_s, n_t))
    for r in rows:
        mat[r["student_layer"], r["teacher_layer"]] = r["cka"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Teacher layer")
    ax.set_ylabel("Student layer")
    ax.set_title("Figure 7a: Layer × Layer CKA")
    fig.colorbar(im, ax=ax)
    _save(fig, out_path)
    plt.close(fig)


def fig7bcd_scatter(t12_metrics: dict, t05_metrics: dict, t09_metrics: dict, out_dir: Path):
    """Figure 7b/7c/7d: Principal Angle × CHG / Attn-cos × Retention / Eff-rank × CHG。"""
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
    axes[0].scatter(pa, [chg] * len(pa))
    axes[0].set_xlabel("Principal Angle")
    axes[0].set_ylabel("CHG")
    axes[0].set_title("Fig.7b: PA × CHG")
    axes[1].scatter(ac, [ret] * len(ac))
    axes[1].set_xlabel("Attn-out cosine")
    axes[1].set_ylabel("Retention")
    axes[1].set_title("Fig.7c: Attn-cos × Retention")
    axes[2].scatter(er, [chg] * len(er))
    axes[2].set_xlabel("Effective rank (Student)")
    axes[2].set_ylabel("CHG")
    axes[2].set_title("Fig.7d: ER × CHG")
    _save(fig, out_dir / "fig7bcd.png")
    plt.close(fig)


def render_all(artifacts: dict, out_dir: Path):
    """一次性渲染全部 Figure。artifacts 是 {name: metrics_dict}。"""
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