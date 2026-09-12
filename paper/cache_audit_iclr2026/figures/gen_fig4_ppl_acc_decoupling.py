"""Fig 4: PPL vs accuracy decoupling across all methods and runs (log-x)."""
from paper_plot_style import *
import json

data = json.load(open(DATA))
pts = []
for run, e in data.items():
    ms = e.get("method_stats") or {}
    for meth in ("ridge_self_kv", "ridge_kv_both", "ridge_native", "student",
                 "ridge_k_only", "ridge_v_only", "ridge_mix_a25",
                 "ridge_win_low", "ridge_win_mid", "ridge_win_high"):
        v = ms.get(meth) or {}
        ppl, acc = v.get("ppl_mean"), v.get("accuracy")
        if ppl and acc and 0 < ppl < 1e8:
            tag = "self-kv" if meth == "ridge_self_kv" else (
                  "student" if meth == "student" else (
                  "native" if meth == "ridge_native" else "translated"))
            pts.append((ppl, acc, tag))

fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.0))
styles = {"self-kv": (COLORS[2], 'o', "Student self-kv"),
          "student": (COLORS[2], '*', "Student prefill"),
          "native": (COLORS[3], '^', "Native injection"),
          "translated": (COLORS[0], 'o', "Translated caches")}
seen = set()
for ppl, acc, tag in pts:
    c, m, lab = styles[tag]
    ax.scatter(ppl, acc, c=[c], marker=m, s=22, alpha=0.75,
               label=lab if tag not in seen else None)
    seen.add(tag)
ax.set_xscale('log')
ax.set_xlabel("Perplexity on scoring suffix (log scale)")
ax.set_ylabel("Accuracy")
ax.annotate("fluent $\\neq$ capable:\nPPL 23.7, acc 0.267",
            xy=(23.7, 0.267), xytext=(400, 0.40), fontsize=7,
            arrowprops=dict(arrowstyle='->', lw=0.7))
ax.legend(frameon=False, fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
save_fig(fig, 'fig4_ppl_acc_decoupling', out_dir='figures')
