"""Table 1: main results grid (mapper families x pairs) -> LaTeX."""
import json

data = json.load(open('figures/data/all_results.json'))

def g(run, field="chg_gold", meth="ridge_kv_both"):
    e = data[run]
    if field == "chg_gold":
        key = {"ridge_kv_both": "chg_gold", "ridge_native": "chg_gold_native",
               "ridge_self_kv": "chg_gold_selfkv"}[meth]
        d = e.get(key) or {}
        return d.get("mean"), d.get("ci_low"), d.get("ci_high")
    v = (e.get("method_stats") or {}).get(meth) or {}
    return v.get("accuracy"), v.get("gold_prob_mean"), v.get("ppl_mean")

def fmt_chg(t):
    if not t or t[0] is None:
        return "--"
    m, lo, hi = t
    def s(x):
        return f"+{x:.3f}" if x >= 0 else f"-{abs(x):.3f}"
    return f"${m:+.3f}$ [${s(lo)}, {s(hi)}$]" if lo is not None else f"${m:+.3f}$"

def fmt_acc(run, meth="ridge_kv_both"):
    a, gp, ppl = g(run, "method_stats", meth)
    if a is None:
        return "--"
    ppl_s = f"{ppl:.1f}" if ppl and ppl < 1e6 else (f"{ppl:.0e}" if ppl else "--")
    return f"{a:.3f} / {gp:.3f} / {ppl_s}"

rows = [
    ("Identity control (self-kv), 1.7B", "v12-4b-1.7b-ridge-20260829-144617", "ridge_self_kv"),
    ("Student self-prefill, 1.7B", "v12-4b-1.7b-ridge-20260829-144617", "student"),
    ("Teacher (full prefill), 1.7B", "v12-4b-1.7b-ridge-20260829-144617", "teacher"),
    ("Native average, 1.7B", "v12-4b-1.7b-ridge-20260829-144617", "ridge_native"),
    ("Ridge per-head, c=30, 1.7B", "v12-4b-1.7b-ridge-20260829-144617", "ridge_kv_both"),
    ("Affine per-head, c=30, 1.7B", "v12-4b-1.7b-affine-20260829-142010", "ridge_kv_both"),
    ("Affine per-head, c=200, 1.7B", "v13-4b-to-1.7b-affine-c200-20260829-152704", "ridge_kv_both"),
    ("Affine per-layer, c=200, 1.7B", "v13-4b-to-1.7b-affine_layer-c200-20260829-153202", "ridge_kv_both"),
    ("Task-aware, c=200, 1.7B", "v13-4b-to-1.7b-taskaware-c200-20260829-155727", "ridge_kv_both"),
    ("RAT, c=30, 1.7B", "v14-4b-to-1.7b-rat-c30-20260829-174525", "ridge_kv_both"),
    ("RAT, c=200, 1.7B", "v14-4b-to-1.7b-rat-c200-20260829-174926", "ridge_kv_both"),
    ("RAT no-anchor rank-32, c=30, 1.7B", "v14-4b-to-1.7b-rat-c30-noanchor-20260829-180839", "ridge_kv_both"),
    ("Affine per-head, c=200, 0.6B", "v13-4b-to-0.6b-affine-c200-20260829-154145", "ridge_kv_both"),
    ("RAT, c=200, 0.6B", "v14-4b-to-0.6b-rat-c200-20260829-175436", "ridge_kv_both"),
]

lines = [r"\caption{Main audit results. Acc/gold/PPL: answer accuracy,",
         r"gold-letter probability, and perplexity of the scoring suffix for",
         r"the translated cache (kv-both). CHG is the gold-probability change",
         r"over the student's own prefill with 95\% bootstrap CIs. Pairs:",
         r"Qwen3-4B$\to$1.7B unless noted. All runs use disjoint",
         r"calibration/evaluation splits and the unified prompt format.}",
         r"\label{tab:main}",
         r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Configuration & Acc / Gold / PPL & Gold CHG [95\% CI] \\",
         r"\midrule"]
for label, run, meth in rows:
    e = data[run]
    v = e["method_stats"].get(meth) or {}
    a, gp, ppl = v.get("accuracy"), v.get("gold_prob_mean"), v.get("ppl_mean")
    if meth in ("ridge_kv_both", "ridge_native", "ridge_self_kv"):
        chg = fmt_chg(g(run, meth=meth))
    else:
        chg = "--"
    ppl_s = "--"
    if ppl is not None:
        ppl_s = f"{ppl:.1f}" if ppl < 1e6 else f"{ppl:.0e}"
    lines.append(f"{label} & {a:.3f} / {gp:.3f} / {ppl_s} & {chg} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("figures/tab1_main.tex", "w").write("\n".join(lines) + "\n")
print("tab1 written")
