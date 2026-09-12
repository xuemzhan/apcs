"""Fig 4: perplexity vs accuracy across audit-protocol runs (log-x).

Only runs that belong to the audit protocol are plotted; the superseded v1.1
runs (confidence-based metric, in-sample calibration) are excluded, matching
the exclusion stated in the paper.
"""
from paper_plot_style import *
import json

data = json.load(open(DATA, encoding="utf-8"))

AUDIT_PREFIXES = ("v12-", "v13-", "v14-", "v15-",
                  "qwen3-affine-c30-tail-sanity")
METHODS = {
    "ridge_self_kv": "reference",
    "student": "reference",
    "ridge_native": "native",
    "ridge_kv_both": "translated",
    "ridge_k_only": "translated",
    "ridge_v_only": "translated",
    "ridge_mix_a25": "probe",
    "ridge_mix_a50": "probe",
    "ridge_mix_a75": "probe",
    "ridge_win_low": "probe",
    "ridge_win_mid": "probe",
    "ridge_win_high": "probe",
}

pts = []
for run, entry in data.items():
    if not run.startswith(AUDIT_PREFIXES):
        continue
    ms = entry.get("method_stats") or {}
    for meth, tag in METHODS.items():
        v = ms.get(meth) or {}
        ppl, acc = v.get("ppl_mean"), v.get("accuracy")
        if ppl and acc and 0 < ppl < 1e8:
            pts.append((ppl, acc, tag))

fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.1))
styles = {
    "reference": (COLORS[2], "o", "student self-prefill / identity control"),
    "native": (COLORS[3], "^", "native teacher content"),
    "translated": (COLORS[0], "o", "translated cache"),
    "probe": (COLORS[1], "s", "probe mixture / window"),
}
seen = set()
for ppl, acc, tag in pts:
    c, m, lab = styles[tag]
    ax.scatter(ppl, acc, c=[c], marker=m, s=20, alpha=0.7,
               label=lab if tag not in seen else None)
    seen.add(tag)

ax.set_xscale("log")
ax.set_xlabel("Perplexity on the scoring suffix (log scale)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.05, 0.62)
ax.axvline(21.2, color="gray", ls=":", lw=0.8)
ax.text(21.2, 0.60, " identity-control perplexity", fontsize=6,
        color="gray", rotation=90, va="top")
ax.annotate("near-native fluency, still 23 accuracy points\n"
            "below the student (PPL 23.7, acc 0.267 vs 0.500)",
            xy=(23.7, 0.267), xytext=(120, 0.10), fontsize=6.5,
            arrowprops=dict(arrowstyle="->", lw=0.7))
ax.legend(frameon=False, fontsize=6, loc="upper right")
fig.tight_layout()
save_fig(fig, 'fig4_ppl_acc_decoupling', out_dir='figures')
