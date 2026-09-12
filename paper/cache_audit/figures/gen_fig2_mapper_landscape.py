"""Fig 2: H2 mapper landscape — gold CHG with 95% CI across mapper families."""
from paper_plot_style import *
import json

data = json.load(open(DATA))

# run-name -> (label, mapper family, calib)
SELECT_17B = [
    ("v12-4b-1.7b-ridge-20260829-144617",       "Ridge (per-head)", "30"),
    ("v12-4b-1.7b-affine-20260829-142010",      "Affine (per-head)", "30"),
    ("v15-4b-to-1.7b-jointmlp-c30-s42-20260912-072431", "Joint MLP", "30"),
    ("v13-4b-to-1.7b-affine-c200-20260829-152704", "Affine (per-head)", "200"),
    ("v13-4b-to-1.7b-affine-c200-lam1e-02-20260829-154654", "Affine $\\lambda$=1e-2", "200"),
    ("v13-4b-to-1.7b-affine-c200-lam1e-01-20260829-155201", "Affine $\\lambda$=1e-1", "200"),
    ("v13-4b-to-1.7b-affine_layer-c200-20260829-153202", "Affine (per-layer)", "200"),
    ("v13-4b-to-1.7b-taskaware-c200-20260829-155727", "Task-aware", "200"),
    ("v14-4b-to-1.7b-rat-c30-20260829-174525",  "RAT", "30"),
    ("v14-4b-to-1.7b-rat-c200-20260829-174926", "RAT", "200"),
]

rows = []
for run, label, calib in SELECT_17B:
    e = data[run]
    g = e["chg_gold"]
    rows.append((label, calib, g["mean"], g["ci_low"], g["ci_high"]))

fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.4))
xs = range(len(rows))
for i, (label, calib, mean, lo, hi) in enumerate(rows):
    color = COLORS[3] if mean > 0 else COLORS[0]
    ax.bar(i, mean, color=color, alpha=0.85, width=0.62)
    ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt='none',
                ecolor='black', elinewidth=1.0, capsize=3)
ax.axhline(0, color='black', linewidth=0.8)
labels = [f"{lab}\n(c={c})" for lab, c, *_ in rows]
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.5)
ax.set_ylabel("Gold-prob. CHG\n(handoff $-$ student)")
ax.set_ylim(-0.40, 0.08)
ax.annotate("self-kv control: $+0.0001$\n$[-0.0006, +0.0009]$",
            xy=(0.02, 0.92), xycoords='axes fraction', fontsize=7)
fig.tight_layout()
save_fig(fig, 'fig2_mapper_landscape', out_dir='figures')
