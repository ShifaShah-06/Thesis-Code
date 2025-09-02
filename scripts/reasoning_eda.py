"""
Exploratory Data Analysis (EDA) for the reasoning_wide.csv dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_data(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    return df


def basic_overview(df: pd.DataFrame) -> None:

    print("DataFrame shape (rows, columns):", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())


def save_summary_statistics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:

    numeric_cols = [col for col in df.columns if col not in ['Year_submission', 'ID']]
    summary_stats = df[numeric_cols].describe().T
    summary_path = output_dir / 'summary_stats.csv'
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    return summary_stats


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:

    numeric_cols = [col for col in df.columns if col not in ['Year_submission', 'ID']]
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of Reasoning Metrics')
    plt.tight_layout()
    heatmap_path = output_dir / 'correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:

    numeric_cols = [col for col in df.columns if col not in ['Year_submission', 'ID']]
    num_cols = len(numeric_cols)
    cols = 4  # number of columns in subplot grid
    rows = int(np.ceil(num_cols / cols))

    # Histograms
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    for idx, col in enumerate(numeric_cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        sns.histplot(df[col], kde=True, ax=ax, color='skyblue', bins=30)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
    # Remove empty subplots
    for idx in range(num_cols, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])
    plt.tight_layout()
    hist_path = output_dir / 'histograms.png'
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Histograms saved to: {hist_path}")

    # Boxplots
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    for idx, col in enumerate(numeric_cols):
        r = idx // cols
        c = idx % cols
        ax = axes2[r, c]
        sns.boxplot(x=df[col], ax=ax, color='lightgreen')
        ax.set_title(f'Boxplot of {col}')
        ax.set_xlabel(col)
    for idx in range(num_cols, rows * cols):
        r = idx // cols
        c = idx % cols
        fig2.delaxes(axes2[r][c])
    plt.tight_layout()
    boxplot_path = output_dir / 'boxplots.png'
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Boxplots saved to: {boxplot_path}")


def plot_mean_trends(df: pd.DataFrame, output_dir: Path) -> None:

    reasoning_types = ['Deductive', 'Inductive', 'Abductive', 'Analogical']
    categories = ['original', 'global', 'local']
    # Compute mean per year
    metrics_cols = [col for col in df.columns if col not in ['Year_submission', 'ID']]
    mean_by_year = df.groupby('Year_submission')[metrics_cols].mean()
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, rtype in enumerate(reasoning_types):
        ax = axes[i]
        for cat in categories:
            col_name = f'{rtype}_per_1000_words_{cat}'
            ax.plot(mean_by_year.index, mean_by_year[col_name], marker='o', label=cat.capitalize())
        ax.set_title(f'Mean {rtype} Reasoning per 1000 Words Over Years')
        ax.set_xlabel('Year of Submission')
        ax.set_ylabel('Mean per 1000 Words')
        ax.legend()
    plt.tight_layout()
    trends_path = output_dir / 'mean_trends.png'
    plt.savefig(trends_path, dpi=300)
    plt.close()
    print(f"Mean trends plot saved to: {trends_path}")


def plot_global_local_ratio(df: pd.DataFrame, output_dir: Path) -> None:

    reasoning_types = ['Deductive', 'Inductive', 'Abductive', 'Analogical']
    ratio_cols = {}
    # Calculate ratios
    for rtype in reasoning_types:
        ratio_col = f'{rtype}_global_to_local_ratio'
        ratio_cols[rtype] = ratio_col
        df[ratio_col] = df[f'{rtype}_per_1000_words_global'] / df[f'{rtype}_per_1000_words_local']
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, rtype in enumerate(reasoning_types):
        col = ratio_cols[rtype]
        ax = axes[i]
        sns.boxplot(x=df[col], ax=ax, color='orchid')
        ax.set_title(f'Global to Local Ratio for {rtype}')
        ax.set_xlabel('Ratio (Global / Local)')
    plt.tight_layout()
    ratio_path = output_dir / 'global_local_ratio_boxplots.png'
    plt.savefig(ratio_path, dpi=300)
    plt.close()
    print(f"Global-to-local ratio boxplots saved to: {ratio_path}")


def main():
    # Path to CSV file relative to this script
    csv_path = Path(r'C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\Reasoning Pipeline Output\reasoning_wide.csv')
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Please ensure the file is in the working directory."
        )
    output_dir = Path(r'C:\Users\shifa\OneDrive\Desktop\Thesis\Corpora\Reasoning Pipeline Output')
    # Load data
    df = load_data(csv_path)
    # Print basic overview
    basic_overview(df)
    # Generate and save summary statistics
    save_summary_statistics(df, output_dir)
    # Visualisations
    plot_correlation_heatmap(df, output_dir)
    plot_distributions(df, output_dir)
    plot_mean_trends(df, output_dir)
    plot_global_local_ratio(df, output_dir)
    print("\nEDA completed. Plots and summary statistics have been saved.")


if __name__ == '__main__':
    main()