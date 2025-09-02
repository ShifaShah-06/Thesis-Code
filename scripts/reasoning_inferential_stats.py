import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

WIDE_PATH =  r"C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\Reasoning Pipeline Output\reasoning_wide.csv"
METRICS = ["Deductive", "Inductive", "Abductive", "Analogical"]
CORPUS_ORDER = ["original", "local", "global"]

df = pd.read_csv(WIDE_PATH)

# ---- DESCRIPTIVE STATS ----
desc_rows = []
for m in METRICS:
    for c in CORPUS_ORDER:
        col = f"{m}_per_1000_words_{c}"
        vals = df[col].dropna()
        desc_rows.append({
            "Metric": m,
            "Corpus": c.capitalize(),
            "N": len(vals),
            "Mean": np.mean(vals),
            "SD": np.std(vals, ddof=1),
            "Median": np.median(vals),
            "Min": np.min(vals),
            "Max": np.max(vals),
        })
desc_stats = pd.DataFrame(desc_rows)
desc_stats.to_csv("reasoning_descriptive_stats.csv", index=False)
print("\n=== DESCRIPTIVE STATS SAVED: reasoning_descriptive_stats.csv ===\n")
print(desc_stats)

# ---- INFERENTIAL STATS ----
def cohen_dz(x, y):
    """Cohen's dz for paired samples."""
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def run_stats(metric, a, b):
    col_a = f"{metric}_per_1000_words_{a}"
    col_b = f"{metric}_per_1000_words_{b}"
    x = df[col_a].astype(float)
    y = df[col_b].astype(float)
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values
    n = len(x)
    mean_a, mean_b = np.mean(x), np.mean(y)
    mean_diff = mean_a - mean_b
    # t-test
    t, p = ttest_rel(x, y)
    dz = cohen_dz(x, y)
    # Wilcoxon
    try:
        w_stat, w_p = wilcoxon(x, y)
    except ValueError:  # all differences zero or <10 pairs
        w_stat, w_p = np.nan, np.nan
    return {
        "Metric": metric,
        "Comparison": f"{a.capitalize()} vs {b.capitalize()}",
        "N": n,
        "Mean_A": mean_a,
        "Mean_B": mean_b,
        "MeanDiff(A-B)": mean_diff,
        "t": t,
        "p": p,
        "dz": dz,
        "sig(p<.05)": p < 0.05,
        "W": w_stat,
        "Wp": w_p,
        "sig_W(p<.05)": w_p < 0.05 if not np.isnan(w_p) else np.nan
    }

stats_rows = []
for m in METRICS:
    for a in ["local", "global"]:
        stats_rows.append(run_stats(m, a, "original"))
stats_df = pd.DataFrame(stats_rows)

# ---- PRINT SUMMARY TABLE ----
print("\n=== INFERENTIAL STATS (T-test + Wilcoxon) ===\n")
print(stats_df[["Metric", "Comparison", "N", "Mean_A", "Mean_B", "MeanDiff(A-B)", "t", "p", "dz", "sig(p<.05)", "W", "Wp", "sig_W(p<.05)"]].round(3).to_string(index=False))

# ---- SAVE FULL STATS ----
stats_df.to_csv("reasoning_inferential_stats.csv", index=False)
print("\n=== FULL INFERENTIAL STATS SAVED: reasoning_inferential_stats.csv ===")

