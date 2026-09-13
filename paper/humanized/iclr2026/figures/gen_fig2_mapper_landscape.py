"""Fig 2: H2 mapper landscape — gold CHG with 95% CI across mapper families."""
from paper_plot_style import *
import json

data = json.load(open(DATA, encoding="utf-8"))

# run-name -> (label, mapper family, calib)
SELECT_17B = [
    ("v12-4b-1.7b-ridge-20260829-144617",       "Ridge (per-head)", "30"),
    ("v12-4b-1.7b-affine-20260829-142010",      "Affine (per-head)", "30"),
    ("v13-4b-to-1.7b-affine-c200-20260829-152704", "Affine (per-head)", "200"),
    ("v13-4b-to-1.7b-affine_layer-c200-20260829-153202", "Affine (per-layer)", "200"),
    ("v13-4b-to-1.7b-taskaware-c200-20260829-155727", "Task-aware", "200"),
    ("v14-4b-to-1.7b-rat-c30-20260829-174525",  "RAT", "30"),
    ("v14-4b-to-1.7b-rat-c200-20260829-174926", "RAT", "200"),
]

# Gradient-trained families: one bar per family/budget, spanning independent
# trainings (hatched) instead of a bootstrap CI.
RANGE_17B = [
    ("MLP (per-head)", "30", [
        "v15-4b-to-1.7b-mlp-c30-s42-20260829-195607",
        "v15-4b-to-1.7b-mlp-c30-s42-repro-20260912-115929",
        "v15-4b-to-1.7b-mlp-c30-seed0-20260912-145147",
        "v15-4b-to-1.7b-mlp-c30-seed1-20260912-150059",
        "v15-4b-to-1.7b-mlp-c30-seed2-20260912-151111",
    ]),
    ("MLP (per-head)", "200", [
        "v15-4b-to-1.7b-mlp-c200-s42-20260829-200557",
        "v15-4b-to-1.7b-mlp-c200-s42-repro-20260912-120751",
        "v15-4b-to-1.7b-mlp-c200-seed0-20260912-152031",
        "v15-4b-to-1.7b-mlp-c200-seed1-20260912-153003",
        "v15-4b-to-1.7b-mlp-c200-seed2-20260912-154233",
    ]),
    ("Joint MLP", "30", [
        "v15-4b-to-1.7b-jointmlp-c30-s42-20260912-072431",
        "v15-4b-to-1.7b-jointmlp-c30-seed0-20260912-164750",
        "v15-4b-to-1.7b-jointmlp-c30-seed1-20260912-164949",
        "v15-4b-to-1.7b-jointmlp-c30-seed2-20260912-165141",
    ]),
]

rows = []
for run, label, calib in SELECT_17B:
    e = data[run]
    g = e["chg_gold"]
    rows.append((label, calib, g["mean"], g["ci_low"], g["ci_high"], False))
for label, calib, runs in RANGE_17B:
    means = [data[r]["chg_gold"]["mean"] for r in runs]
    rows.append((label, calib, sum(means) / len(means), min(means), max(means), True))

fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.4))
xs = range(len(rows))
for i, (label, calib, mean, lo, hi, is_range) in enumerate(rows):
    color = COLORS[3] if mean > 0 else COLORS[0]
    ax.bar(i, mean, color=color, alpha=0.85, width=0.62,
           hatch='//' if is_range else None,
           edgecolor='black' if is_range else None, linewidth=0.4)
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
ax.annotate("hatched: independent-training range",
            xy=(0.02, 0.84), xycoords='axes fraction', fontsize=7)
fig.tight_layout()
save_fig(fig, 'fig2_mapper_landscape', out_dir='figures')
