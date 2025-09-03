import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


INPUT = Path("reasoning_wide.csv")


dfw = pd.read_csv(INPUT)

types = ["Deductive", "Inductive", "Abductive", "Analogical"]
corpora = ["original", "global", "local"]

# -------------
# Long format
# -------------
rows = []
for _, row in dfw.iterrows():
    base = {"Year_submission": row["Year_submission"], "ID": row["ID"]}
    for corpus in corpora:
        for t in types:
            col = f"{t}_per_1000_words_{corpus}"
            rate = row.get(col, np.nan)
            rows.append({**base, "corpus": corpus, "type": t, "rate_per_1000": rate})
dfl = pd.DataFrame(rows)

# -------------
# Per-essay metrics: total rate, richness, Shannon H'
# -------------
def per_essay_metrics_safe(g):
    out = []
    for corpus in corpora:
        r = g[g["corpus"]==corpus].set_index("type")["rate_per_1000"].reindex(types).fillna(0.0)
        total = r.sum()
        richness = int((r > 0).sum())
        if total > 0:
            p = r / total
            ppos = p[p > 0]
            H = -(ppos * np.log(ppos)).sum()
        else:
            H = 0.0
        out.append({"corpus": corpus,
                    "total_rate_per_1000": float(total),
                    "richness_0to4": int(richness),
                    "shannon_H": float(H)})
    return pd.DataFrame(out)

per_essay = (dfl.groupby(["Year_submission", "ID"])
               .apply(per_essay_metrics_safe)
               .reset_index()
               .drop(columns=["level_2"]))

# -------------
# Paired deltas vs Original
# -------------
metrics = ["total_rate_per_1000", "richness_0to4", "shannon_H"]
wide = per_essay.pivot(index=["Year_submission","ID"], columns="corpus", values=metrics)
wide.columns = [f"{m}_{c}" for m, c in wide.columns]
wide = wide.reset_index()

for m in metrics:
    wide[f"delta_{m}_local_minus_original"] = wide[f"{m}_local"] - wide[f"{m}_original"]
    wide[f"delta_{m}_global_minus_original"] = wide[f"{m}_global"] - wide[f"{m}_original"]

# -------------
# Summaries
# -------------
summary_type = (dfl.groupby(["corpus","type"])["rate_per_1000"]
                  .agg(mean="mean", median="median", std="std", count="count")
                  .reset_index())

summary_metrics = (per_essay.groupby("corpus")[["total_rate_per_1000","richness_0to4","shannon_H"]]
                   .agg(["mean","median","std","count"]))
summary_metrics.columns = ['_'.join(col).strip() for col in summary_metrics.columns.values]
summary_metrics = summary_metrics.reset_index()

# -------------
# Save outputs
# -------------
Path("outputs").mkdir(exist_ok=True)
dfl.to_csv("outputs/reasoning_long.csv", index=False)
per_essay.to_csv("outputs/reasoning_per_essay_metrics.csv", index=False)
wide.to_csv("outputs/reasoning_per_essay_deltas.csv", index=False)
summary_type.to_csv("outputs/reasoning_summary_by_type.csv", index=False)
summary_metrics.to_csv("outputs/reasoning_summary_metrics.csv", index=False)

# -------------
# Figures (matplotlib, no custom colors)
# -------------
## ---- Figure 1 ----
g = (
    dfl.groupby(["type", "corpus"])["rate_per_1000"]
       .agg(mean="mean", sd="std", n="count")
       .reset_index()
)
g["se"]   = g["sd"] / np.sqrt(g["n"])
g["ci95"] = 1.96 * g["se"]

types   = ["Deductive", "Inductive", "Abductive", "Analogical"]
corpora = ["original", "local", "global"]  # make sure these match dfl['corpus']
labels  = {"original": "Original", "local": "Local", "global": "Global"}
colors  = {"original": "#86be91", "local": "#509e90", "global": "#1f5b86"}

import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.edgecolor": "#222", "axes.linewidth": 0.8, "savefig.dpi": 300
})

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(types))
width = 0.26
offsets = [-width, 0.0, width]

for k, corp in enumerate(corpora):
    gg = (
        g[g["corpus"].str.lower() == corp]
         .set_index("type")
         .reindex(types)
    )
    means = gg["mean"].to_numpy()
    errs  = gg["ci95"].to_numpy()

    bars = ax.bar(
        x + offsets[k], means, width,
        yerr=errs, capsize=3,
        color=colors[corp], edgecolor="#333", linewidth=0.6,
        label=labels[corp]
    )
    for b, val in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylabel("Mean per 1,000 words")
ax.set_title("Reasoning by Corpus (means with 95% CIs)")
ax.yaxis.grid(True, linestyle=":", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
plt.savefig("outputs/fig_reasoning_rate_by_type_corpus_CI.png", bbox_inches="tight", dpi=300)
plt.close()

# Figure 2: Boxplot Shannon H' by corpus
plt.figure(figsize=(6,5))
data_H = [per_essay[per_essay["corpus"]==c]["shannon_H"].values for c in corpora]
plt.boxplot(data_H, labels=corpora, showmeans=True)
plt.ylabel("Shannon Diversity H'")
plt.title("Reasoning Diversity (Shannon H') by Corpus")
plt.tight_layout()
plt.savefig("outputs/fig_shannon_by_corpus.png", dpi=300)
plt.close()

# Figure 3: Boxplot Richness by corpus
plt.figure(figsize=(6,5))
data_R = [per_essay[per_essay["corpus"]==c]["richness_0to4"].values for c in corpora]
plt.boxplot(data_R, labels=corpora, showmeans=True)
plt.ylabel("Richness (0–4 distinct types)")
plt.title("Reasoning richness by corpus")
plt.tight_layout()
plt.savefig("outputs/fig_richness_by_corpus.png", dpi=300)
plt.close()

print("Done. CSVs written to ./outputs and figures saved.")
