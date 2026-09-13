"""Fig 3: Oracle probes — fraction blending and per-octant window injection."""
from paper_plot_style import *
import json

data = json.load(open(DATA))
mix_run = "v14-4b-to-1.7b-rat-c30-probes-20260829-175933"
oct_run = "v15-4b-to-1.7b-affine-c30-probes-octant-s42-20260912-070436"
ms = data[mix_run]["method_stats"]
mo = data[oct_run]["method_stats"]

fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.6))

ax = axes[0]
alphas = [0.0, 0.25, 0.50, 0.75]
keys = ["ridge_self_kv", "ridge_mix_a25", "ridge_mix_a50", "ridge_mix_a75"]
gold = [ms[k]["gold_prob_mean"] for k in keys]
acc = [ms[k]["accuracy"] for k in keys]
ax.plot(alphas, gold, 'o-', color=COLORS[0], label='Gold prob.')
ax.plot(alphas, acc, 's--', color=COLORS[1], label='Accuracy')
ax.axhline(ms["teacher"]["gold_prob_mean"], color=COLORS[2], linestyle=':',
           label='Teacher (full)')
ax.set_xlabel(r"Teacher fraction $\alpha$ in cache")
ax.set_ylabel("Score")
ax.set_xticks(alphas)
ax.legend(frameon=False, fontsize=6.5)

ax = axes[1]
keys = [f"ridge_win_oct{i}" for i in range(8)]
vals = [mo[k]["gold_prob_mean"] for k in keys]
cols = [COLORS[3] if v >= 0.49 else COLORS[0] for v in vals]
ax.bar(range(len(keys)), vals, color=cols, alpha=0.85, width=0.7)
ax.axhline(mo["ridge_self_kv"]["gold_prob_mean"], color='black',
           linewidth=0.8, linestyle='--', label='self-kv')
ax.set_xticks(range(len(keys)))
ax.set_xticklabels([str(i) for i in range(8)], fontsize=7)
ax.set_xlabel("Student-layer octant (0 = bottom)")
ax.set_ylabel("Gold prob.")
ax.set_ylim(0, 0.58)
ax.legend(frameon=False, fontsize=6.5, loc='lower left')
fig.tight_layout()
save_fig(fig, 'fig3_oracle_probes', out_dir='figures')
