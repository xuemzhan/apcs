"""Fig 3: Oracle probes.

Left:  gold-letter probability as translated teacher content replaces the
       student's own cache, for two content sources (RAT and affine). Filled
       markers mark mixture fractions whose paired CHG interval excludes zero;
       open markers mark fractions that are not separated from the student.
Right: per-octant window injection (affine source) as gold-probability CHG with
       95% CIs, so the reader can see that one octant separates from the
       student and that two top-layer octants are mildly positive with
       intervals spanning zero.
"""
from paper_plot_style import *
import json

data = json.load(open(DATA, encoding="utf-8"))

MIX = [
    ("v14-4b-to-1.7b-rat-c30-probes-20260829-175933",
     "RAT source", COLORS[0], "-", "o"),
    ("v15-4b-to-1.7b-affine-c30-probes-s42-20260829-234406",
     "affine source", COLORS[3], "--", "s"),
]
OCT = "v15-4b-to-1.7b-affine-c30-probes-octant-s42-20260912-070436"

fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7))

# ---- left: teacher-fraction blending, two content sources -------------------
ax = axes[0]
alphas = [0.0, 0.25, 0.50, 0.75]
keys = ["ridge_self_kv", "ridge_mix_a25", "ridge_mix_a50", "ridge_mix_a75"]
for run, name, col, ls, mk in MIX:
    ms = data[run]["method_stats"]
    cg = data[run]["chg_gold_all"]
    gold = [ms[k]["gold_prob_mean"] for k in keys]
    sig = [False] + [(cg[k]["ci_low"] > 0) or (cg[k]["ci_high"] < 0)
                     for k in keys[1:]]
    ax.plot(alphas, gold, ls, color=col, lw=1.1, label=name)
    for a, g, s in zip(alphas, gold, sig):
        ax.plot([a], [g], marker=mk, ms=4.5, color=col, linestyle="none",
                markerfacecolor=col if s else "white", markeredgecolor=col,
                zorder=3)
ref = data[MIX[0][0]]["method_stats"]
ax.axhline(ref["teacher"]["gold_prob_mean"], color="black", ls=":", lw=0.9,
           label="teacher (full prefill)")
ax.axhline(ref["ridge_self_kv"]["gold_prob_mean"], color="gray", ls="-.",
           lw=0.8, label="student self-prefill")
ax.set_xlabel(r"Teacher fraction $\alpha$ in the blended cache")
ax.set_ylabel("Gold-letter probability")
ax.set_xticks(alphas)
ax.set_xlim(-0.06, 0.81)
ax.set_ylim(0.20, 0.75)
ax.legend(frameon=False, fontsize=6, loc="lower left")
ax.text(0.98, 0.95, "filled marker: 95% CI excludes 0",
        transform=ax.transAxes, fontsize=6, ha="right", va="top")

# ---- right: per-octant window injection as CHG with CIs ---------------------
ax = axes[1]
ms = data[OCT]["chg_gold_all"]
keys = [f"ridge_win_oct{i}" for i in range(8)]
mean = [ms[k]["mean"] for k in keys]
lo = [ms[k]["ci_low"] for k in keys]
hi = [ms[k]["ci_high"] for k in keys]
cols = [COLORS[3] if (l > 0 or h < 0) else COLORS[0]
        for l, h in zip(lo, hi)]
ax.bar(range(8), mean, color=cols, alpha=0.85, width=0.68)
ax.errorbar(range(8), mean,
            yerr=[[m - l for m, l in zip(mean, lo)],
                  [h - m for m, h in zip(mean, hi)]],
            fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(8))
ax.set_xticklabels([str(i) for i in range(8)], fontsize=7)
ax.set_xlabel("student-layer octant (0 = bottom)")
ax.set_ylabel("Gold-prob. CHG vs self-prefill")
ax.set_ylim(-0.45, 0.11)
ax.text(0.02, 0.96, "red: 95% CI excludes 0", transform=ax.transAxes,
        fontsize=6, va="top")
ax.annotate("+0.020", xy=(5, mean[5]), xytext=(4.3, 0.055), fontsize=6,
            arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("+0.006", xy=(7, mean[7]), xytext=(6.2, 0.045), fontsize=6,
            arrowprops=dict(arrowstyle="-", lw=0.6))

fig.tight_layout()
save_fig(fig, 'fig3_oracle_probes', out_dir='figures')
