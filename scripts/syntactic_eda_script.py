"""
Exploratory Data Analysis (EDA) for the syntactic_metrics_wide.csv dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the syntactic metrics dataset from a CSV file."""
    return pd.read_csv(csv_path)


def basic_overview(df: pd.DataFrame) -> None:
    """Print basic information about the DataFrame."""
    print("DataFrame shape (rows, columns):", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())


def save_summary_statistics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute and save summary statistics for numeric columns."""
    numeric_cols = [col for col in df.columns if col not in ['ID', 'Year_submission']]
    summary_stats = df[numeric_cols].describe().T
    summary_path = output_dir / 'summary_stats2.csv'
    summary_stats.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    return summary_stats


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save a heatmap of the correlation matrix for numeric columns."""
    numeric_cols = [col for col in df.columns if col not in ['ID', 'Year_submission']]
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of Syntactic Metrics')
    plt.tight_layout()
    heatmap_path = output_dir / 'correlation_heatmap2.png'
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Create histograms and boxplots for each numeric variable."""
    numeric_cols = [col for col in df.columns if col not in ['ID', 'Year_submission']]
    num_cols = len(numeric_cols)
    cols = 3
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
    for idx in range(num_cols, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])
    plt.tight_layout()
    hist_path = output_dir / 'histograms2.png'
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
    boxplot_path = output_dir / 'boxplots2.png'
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Boxplots saved to: {boxplot_path}")


def plot_mean_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean trends per year for each syntactic metric across categories."""
    # Identify metric names and categories
    metrics = ['clausal_density', 'depth', 'nominalisations_per_1000']
    categories = ['original', 'local', 'global']
    numeric_cols = [col for col in df.columns if col not in ['ID', 'Year_submission']]
    mean_by_year = df.groupby('Year_submission')[numeric_cols].mean()
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 6, 4))
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for cat in categories:
            col_name = f'{metric}_{cat}'
            ax.plot(mean_by_year.index, mean_by_year[col_name], marker='o', label=cat.capitalize())
        ax.set_title(f'Mean {metric.replace("_", " ").title()} Over Years')
        ax.set_xlabel('Year of Submission')
        ax.set_ylabel('Mean')
        ax.legend()
    plt.tight_layout()
    trends_path = output_dir / 'mean_trends2.png'
    plt.savefig(trends_path, dpi=300)
    plt.close()
    print(f"Mean trends plot saved to: {trends_path}")


def plot_ratio_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot boxplots of the global/local ratio for each metric."""
    metrics = ['clausal_density', 'depth', 'nominalisations_per_1000']
    ratio_cols = {}
    for metric in metrics:
        ratio_col = f'{metric}_global_to_local_ratio'
        ratio_cols[metric] = ratio_col
        df[ratio_col] = df[f'{metric}_global'] / df[f'{metric}_local']
    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 6, 4))
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.boxplot(x=df[ratio_cols[metric]], ax=ax, color='orchid')
        ax.set_title(f'Global to Local Ratio for {metric.replace("_", " ").title()}')
        ax.set_xlabel('Ratio (Global / Local)')
    plt.tight_layout()
    ratio_path = output_dir / 'ratio_boxplots2.png'
    plt.savefig(ratio_path, dpi=300)
    plt.close()
    print(f"Ratio boxplots saved to: {ratio_path}")


def main() -> None:
    csv_path = Path('syntactic_metrics_wide.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Ensure the file is available.")
    output_dir = Path('.')
    df = load_data(csv_path)
    basic_overview(df)
    save_summary_statistics(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_distributions(df, output_dir)
    plot_mean_trends(df, output_dir)
    plot_ratio_boxplots(df, output_dir)
    print("\nEDA for syntactic metrics completed. All results saved.")


if __name__ == '__main__':
    main()