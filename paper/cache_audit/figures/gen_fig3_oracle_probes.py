"""Fig 3: Oracle probes — mix_alpha monotone decay + window injection bars."""
from paper_plot_style import *
import json

data = json.load(open(DATA))
run = "v14-4b-to-1.7b-rat-c30-probes-20260829-175933"
ms = data[run]["method_stats"]

alphas = [0.0, 0.25, 0.50, 0.75]
keys = ["ridge_self_kv", "ridge_mix_a25", "ridge_mix_a50", "ridge_mix_a75"]
gold = [ms[k]["gold_prob_mean"] for k in keys]
acc = [ms[k]["accuracy"] for k in keys]

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
ax = axes[0]
ax.plot(alphas, gold, 'o-', color=COLORS[0], label='Gold prob.')
ax.plot(alphas, acc, 's--', color=COLORS[1], label='Accuracy')
ax.axhline(ms["teacher"]["gold_prob_mean"], color=COLORS[2], linestyle=':',
           label='Teacher (full)')
ax.set_xlabel(r"Teacher fraction $\alpha$ in cache")
ax.set_ylabel("Score")
ax.set_xticks(alphas)
ax.legend(frameon=False, fontsize=6.5)

ax = axes[1]
wins = ["ridge_win_low", "ridge_win_mid", "ridge_win_high", "ridge_kv_both", "ridge_self_kv"]
labels = ["low\n1/3", "mid\n1/3", "high\n1/3", "full\n(repl.)", "self\n(0%)"]
vals = [ms[k]["gold_prob_mean"] for k in wins]
cols = [COLORS[3] if v >= 0.49 else COLORS[0] for v in vals]
bars = ax.bar(range(len(wins)), vals, color=cols, alpha=0.85, width=0.6)
ax.axhline(ms["ridge_self_kv"]["gold_prob_mean"], color='black',
           linewidth=0.8, linestyle='--')
ax.set_xticks(range(len(wins)))
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("Gold prob.")
ax.set_ylim(0, 0.58)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
            ha='center', va='bottom', fontsize=6)
fig.tight_layout()
save_fig(fig, 'fig3_oracle_probes', out_dir='figures')
