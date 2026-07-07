#!/usr/bin/env python3
"""Figure 1: attribution Hit@1 per corpus x method. Regenerates fig1_summary.svg.

Numbers are copied from docs/results/*.md (see each entry). Palette is the
CVD-validated Okabe-Ito set (validate_palette.js: ALL CHECKS PASS, worst
adjacent CVD dE 37.2). Deterministic baselines have no error bar; learned
models carry mean +/- std over seeds. mjs (n=1) is excluded — its metric is
vacuous (single free); see the note's Section 6.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# method identity -> fixed hue (never cycled)
METHODS = ["recency", "best domain rule", "bi-LSTM", "transformer (ours)", "oracle rule"]
COLOR = {
    "recency": "#0072B2",
    "best domain rule": "#E69F00",
    "bi-LSTM": "#009E73",
    "transformer (ours)": "#D55E00",
    "oracle rule": "#CC79A7",
}
# corpus -> {method: (mean, std or None; None -> deterministic, no bar). np.nan -> not run}
DATA = {
    # Transformer bars use bias 4 on every corpus (consistent config). The
    # UAF transformer is pooled over 3 realizations (0.818); v1/v2 as noted.
    "UAF\n(real ASan)": {
        "recency": (0.378, None), "best domain rule": (1.000, None),
        "bi-LSTM": (np.nan, None), "transformer (ours)": (0.818, 0.210),
        "oracle rule": (1.000, None),
    },
    "distal v1\n(adjacent)": {
        "recency": (0.000, None), "best domain rule": (0.289, None),
        "bi-LSTM": (0.842, 0.117), "transformer (ours)": (0.537, 0.118),
        "oracle rule": (1.000, None),
    },
    "distal v2\n(gapped, pooled)": {
        "recency": (0.000, None), "best domain rule": (0.355, 0.050),
        "bi-LSTM": (0.856, 0.113), "transformer (ours)": (0.807, 0.152),
        "oracle rule": (1.000, None),
    },
}

corpora = list(DATA)
n_m = len(METHODS)
bw = 0.15
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, corpus in enumerate(corpora):
    for j, m in enumerate(METHODS):
        mean, std = DATA[corpus][m]
        if np.isnan(mean):
            continue
        x = i + (j - (n_m - 1) / 2) * bw
        ax.bar(x, mean, bw * 0.92, color=COLOR[m], zorder=3,
               yerr=std, ecolor="#333333", capsize=2.5 if std else 0,
               error_kw={"lw": 1})
        ax.text(x, mean + (std or 0) + 0.02, f"{mean:.2f}", ha="center",
                va="bottom", fontsize=7, color="#222222", rotation=0)

ax.set_xticks(range(len(corpora)))
ax.set_xticklabels(corpora, fontsize=9)
ax.set_ylabel("Attribution Hit@1", fontsize=10)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eeeeee", zorder=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in METHODS]
ax.legend(handles, METHODS, ncol=5, fontsize=8, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.set_title("The learned model wins no benchmark: below the oracle everywhere, "
             "below or tied with a plain LSTM on the synthetic corpora",
             fontsize=9.5, color="#444444", pad=24)
fig.tight_layout()
out = __file__.rsplit("/", 1)[0] + "/fig1_summary.svg"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".svg", ".png"), dpi=150, bbox_inches="tight")
print("wrote", out)
