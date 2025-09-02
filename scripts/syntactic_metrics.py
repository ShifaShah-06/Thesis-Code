import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


INPUT_CSV = Path(r"C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\syntactic_metrics_wide.csv")
OUT_DIR   = Path(r"C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\syntactic_stats_plots")

METRICS   = ["depth", "clausal_density", "nominalisations_per_1000"]
CORPORA   = {"original": "_original", "global": "_global", "local": "_local"}
AI_MODELS = ["global", "local"]    # comparisons vs Original
N_BINS    = 40

# --------------------------
def paired_cohens_d(x, y):
    diff = np.asarray(y) - np.asarray(x)
    diff = diff[~np.isnan(diff)]
    if diff.size < 2:
        return np.nan
    sd = np.nanstd(diff, ddof=1)
    return float(np.nanmean(diff) / sd) if sd > 0 else np.nan

def save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Wrote: {path}")

def stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

def print_table(df: pd.DataFrame, title: str):
    print("\n" + title)
    print("-" * len(title))
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))

def violin_plot(values_list, labels, title, path):
    plt.figure(figsize=(8,6))
    parts = plt.violinplot(values_list, showmeans=True, showextrema=True, showmedians=False)
    plt.xticks(range(1, len(labels)+1), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def delta_histogram(deltas, title, path):
    plt.figure(figsize=(8,6))
    plt.hist(deltas, bins=N_BINS, alpha=0.8)
    plt.axvline(0, linestyle="--")
    d = deltas[~np.isnan(deltas)]
    if d.size > 1:
        m = float(np.mean(d))
        sd = float(np.std(d, ddof=1))
        se = sd / np.sqrt(d.size)
        tcrit = stats.t.ppf(0.975, df=d.size-1)
        lo, hi = m - tcrit*se, m + tcrit*se
        plt.axvline(m, linestyle="-")
        plt.axvspan(lo, hi, alpha=0.15)
    plt.xlabel("Delta (AI − Original)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def paired_scatter(x, y, xlabel, ylabel, title, path):
    plt.figure(figsize=(6,6))
    plt.plot(x, y, 'o', alpha=0.5)
    mn = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    mx = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    plt.plot([mn, mx], [mn, mx], '--')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    OUT_DIR.mkdir(exist_ok=True)

    # 1) Descriptives
    desc_rows = []
    for m in METRICS:
        for corpus, suf in CORPORA.items():
            col = f"{m}{suf}"
            if col in df.columns:
                s = df[col]
                desc_rows.append({
                    "metric": m,
                    "corpus": corpus,
                    "n": int(s.notna().sum()),
                    "mean": float(s.mean(skipna=True)),
                    "sd": float(s.std(skipna=True)),
                    "median": float(s.median(skipna=True)),
                    "min": float(s.min(skipna=True)),
                    "max": float(s.max(skipna=True)),
                })
    descriptives = pd.DataFrame(desc_rows)
    save_csv(descriptives, OUT_DIR / "syntactic_descriptives.csv")
    print_table(descriptives.round(3), "Descriptive statistics")

    # 2) Paired tests + effect sizes + CI
    paired_rows = []
    for m in METRICS:
        orig_col = f"{m}{CORPORA['original']}"
        if orig_col not in df.columns:
            continue
        for mdl in AI_MODELS:
            mdl_col = f"{m}{CORPORA[mdl]}"
            if mdl_col not in df.columns:
                continue
            mask = df[orig_col].notna() & df[mdl_col].notna()
            x = df.loc[mask, orig_col].to_numpy()
            y = df.loc[mask, mdl_col].to_numpy()
            if x.size < 2:
                continue
            t_stat, t_p = stats.ttest_rel(y, x, nan_policy='omit')
            try:
                w_stat, w_p = stats.wilcoxon(y, x, zero_method='wilcox', alternative='two-sided', correction=False)
            except Exception:
                w_stat, w_p = (np.nan, np.nan)
            d = paired_cohens_d(x, y)
            delta = y - x
            n = delta.size
            mean_delta = float(np.nanmean(delta))
            sd_delta = float(np.nanstd(delta, ddof=1)) if n > 1 else np.nan
            se = sd_delta / np.sqrt(n) if n > 1 else np.nan
            if n > 1:
                tcrit = stats.t.ppf(0.975, df=n-1)
                ci_low = mean_delta - tcrit * se
                ci_high = mean_delta + tcrit * se
            else:
                ci_low = ci_high = np.nan

            # Direction counts + sign test
            d_nonan = delta[~np.isnan(delta)]
            n_neg = int((d_nonan < 0).sum())
            n_pos = int((d_nonan > 0).sum())
            n_eff = n_neg + n_pos
            if n_eff > 0:
                k = min(n_neg, n_pos)
                p_sign = stats.binomtest(k, n_eff, 0.5, alternative="greater").pvalue * 2
                p_sign = min(1.0, p_sign)
            else:
                p_sign = np.nan

            paired_rows.append({
                "metric": m,
                "comparison": f"{mdl} - original",
                "n_pairs": int(n),
                "mean_delta": mean_delta,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "t_stat": float(t_stat), "t_p": float(t_p),
                "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
                "cohens_d_paired": float(d),
                "prop_negative": n_neg / n if n else np.nan,
                "prop_positive": n_pos / n if n else np.nan,
                "sign_test_p": float(p_sign) if not np.isnan(p_sign) else np.nan
            })

    paired_tests = pd.DataFrame(paired_rows)
    save_csv(paired_tests, OUT_DIR / "syntactic_paired_tests.csv")

    # Pretty console print of paired tests
    paired_print = paired_tests.copy()
    paired_print["t (p)"] = paired_print.apply(lambda r: f"{r['t_stat']:.2f} ({r['t_p']:.3g}){stars(r['t_p'])}", axis=1)
    paired_print["Wilcoxon p"] = paired_print["wilcoxon_p"].apply(lambda p: f"{p:.3g}{stars(p)}" if not np.isnan(p) else "")
    paired_print["Cohen d"] = paired_print["cohens_d_paired"].round(3)
    paired_print["(95% CI)"] = paired_print.apply(lambda r: f"{r['mean_delta']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]", axis=1)
    paired_print["% ↓"] = (paired_print["prop_negative"]*100).round(1)
    paired_print["% ↑"] = (paired_print["prop_positive"]*100).round(1)
    cols_show = ["metric","comparison","n_pairs","(95% CI)","t (p)","Wilcoxon p","Cohen d","% ↓","% ↑"]
    print_table(paired_print[cols_show], "Paired tests (AI − Original)")

    # 3) Variance tests (Levene) + console
    var_rows = []
    for m in METRICS:
        ocol = f"{m}{CORPORA['original']}"
        if ocol not in df.columns:
            continue
        o = df[ocol].dropna().to_numpy()
        if o.size < 2:
            continue
        for mdl in AI_MODELS:
            acol = f"{m}{CORPORA[mdl]}"
            if acol not in df.columns:
                continue
            a = df[acol].dropna().to_numpy()
            if a.size < 2:
                continue
            W, p = stats.levene(o, a, center='median')
            var_rows.append({
                "metric": m,
                "groups": f"original vs {mdl}",
                "levene_W": float(W),
                "levene_p": float(p),
                "sd_original": float(np.nanstd(o, ddof=1)),
                f"sd_{mdl}": float(np.nanstd(a, ddof=1)),
            })
    variance_tests = pd.DataFrame(var_rows)
    save_csv(variance_tests, OUT_DIR / "syntactic_variance_tests.csv")

    var_print = variance_tests.copy()
    var_print["SD ratio (AI/Orig)"] = var_print.apply(
        lambda r: (r[f"sd_{r['groups'].split()[-1]}"] / r["sd_original"]) if f"sd_{r['groups'].split()[-1]}" in r else np.nan,
        axis=1,
    )
    print_table(var_print.round(3), "Variance tests (Levene, center=median)")

    # 4) Save per-essay deltas (also used for histograms)
    delta_rows = []
    for m in METRICS:
        ocol = f"{m}{CORPORA['original']}"
        if ocol not in df.columns:
            continue
        for mdl in AI_MODELS:
            acol = f"{m}{CORPORA[mdl]}"
            if acol not in df.columns:
                continue
            mask = df[ocol].notna() & df[acol].notna()
            sub = df.loc[mask, ["Year_submission","ID", ocol, acol]].copy()
            sub["model"] = mdl
            sub["metric"] = m
            sub["delta"] = sub[acol] - sub[ocol]
            delta_rows.append(sub[["Year_submission","ID","metric","model","delta"]])
    deltas = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame(columns=["Year_submission","ID","metric","model","delta"])
    save_csv(deltas, OUT_DIR / "syntactic_deltas.csv")

    # 5) PLOTS — Scatter, Violin, Delta histograms
    for m in METRICS:
        xcol = f"{m}{CORPORA['original']}"
        if xcol not in df.columns:
            continue
        x_all = df[xcol].to_numpy()
        for mdl in AI_MODELS:
            ycol = f"{m}{CORPORA[mdl]}"
            if ycol not in df.columns:
                continue
            y_all = df[ycol].to_numpy()
            mask = ~np.isnan(x_all) & ~np.isnan(y_all)
            if np.sum(mask) > 1:
                paired_scatter(
                    x_all[mask], y_all[mask],
                    xlabel="Original", ylabel=mdl.upper(),
                    title=f"{m}: Original vs {mdl.upper()}",
                    path=OUT_DIR / f"{m}_scatter_{mdl}.png"
                )

    # Violin: distribution per corpus
    for m in METRICS:
        vals = []
        labels = []
        for corp in ["original","global","local"]:
            col = f"{m}{CORPORA[corp]}"
            if col in df.columns:
                v = df[col].dropna().to_numpy()
                if v.size > 0:
                    vals.append(v); labels.append(corp.replace("_","-").upper())
        if len(vals) >= 2:
            violin_plot(vals, labels, f"{m}: Distribution per corpus", OUT_DIR / f"{m}_violin.png")

    # Delta histograms
    for m in METRICS:
        for mdl in AI_MODELS:
            dsub = deltas[(deltas["metric"]==m) & (deltas["model"]==mdl)]
            if not dsub.empty:
                delta_histogram(dsub["delta"].to_numpy(), f"{m}: Δ ( {mdl.upper()} − Original )", OUT_DIR / f"{m}_delta_hist_{mdl}.png")

    print("\n[Done] Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
