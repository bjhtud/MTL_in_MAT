import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_multi_seed"
OUTPUT_DIR = BASE_DIR / "analysis_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILES = {
    'direct': RESULTS_DIR / 'direct_results.csv',
    'feature': RESULTS_DIR / 'feature_results.csv',
    'complete': RESULTS_DIR / 'complete_results.csv',
}

SUMMARY_PATH = OUTPUT_DIR / 'summary_stats.csv'
TOP10_OVERALL_PATH = OUTPUT_DIR / 'top10_overall.csv'
TOP10_GROUP_PATH = OUTPUT_DIR / 'top10_{group}.csv'


def load_results():
    data = {}
    for key, path in CSV_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV file: {path}")
        df = pd.read_csv(path)
        data[key] = df
    return data


def summarize_results(data):
    records = []
    for group, df in data.items():
        grouped = df.groupby('model')[['r2', 'mse', 'mae']].agg(['mean', 'std']).reset_index()
        grouped.columns = ['model', 'r2_mean', 'r2_std', 'mse_mean', 'mse_std', 'mae_mean', 'mae_std']
        grouped['group'] = group
        records.append(grouped)
    summary = pd.concat(records, ignore_index=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved summary statistics to {SUMMARY_PATH}")
    return summary


def select_top_models(summary, top_k=10):
    ranking_cols = ['r2_mean', 'mse_mean', 'mae_mean']
    summary_sorted = summary.sort_values(
        by=['r2_mean', 'mse_mean', 'mae_mean'],
        ascending=[False, True, True]
    )

    top_overall = summary_sorted.head(top_k)
    top_overall.to_csv(TOP10_OVERALL_PATH, index=False)
    print(f"Saved overall Top {top_k} combinations to {TOP10_OVERALL_PATH}")

    for group, group_df in summary_sorted.groupby('group'):
        top_group = group_df.head(top_k)
        out_path = OUTPUT_DIR / f"top10_{group}.csv"
        top_group.to_csv(out_path, index=False)
        print(f"Saved Top {top_k} combinations for '{group}' to {out_path}")

def plot_metric_summary(summary):
    melted = summary.melt(
        id_vars=['group', 'model'],
        value_vars=['r2_mean', 'mse_mean', 'mae_mean'],
        var_name='metric',
        value_name='value'
    )
    g = sns.catplot(
        data=melted,
        x='model',
        y='value',
        hue='group',
        col='metric',
        kind='bar',
        height=4,
        aspect=1.5,
        col_wrap=1,
    )
    g.set_xticklabels(rotation=90)
    g.tight_layout()
    plot_path = OUTPUT_DIR / 'metrics_barplot.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved metrics bar plot to {plot_path}")

def plot_boxplots(data):
    for metric in ['r2', 'mse', 'mae']:
        combined = []
        for group, df in data.items():
            tmp = df[['model', metric]].copy()
            tmp['group'] = group
            combined.append(tmp)
        combined_df = pd.concat(combined, ignore_index=True)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=combined_df, x='model', y=metric, hue='group')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f'{metric}_boxplot.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved {metric} boxplot to {plot_path}")

def main():
    data = load_results()
    summary = summarize_results(data)
    select_top_models(summary, top_k=10)
    plot_metric_summary(summary)
    plot_boxplots(data)


if __name__ == '__main__':
    main()
