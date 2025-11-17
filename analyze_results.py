import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_multi_seed"
OUTPUT_DIR = BASE_DIR / "analysis_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILES = {
    "direct": RESULTS_DIR / "direct_results.csv",
    "feature": RESULTS_DIR / "feature_results.csv",
    "complete": RESULTS_DIR / "complete_results.csv",
}

RANK_SUFFIX = {
    "r2": "r2",
    "mse": "mse",
    "rmse": "rmse",
    "mae": "mae",
}

METRIC_ORDER = {
    "r2": False,   # descending
    "mse": True,   # ascending
    "rmse": True,  # ascending
    "mae": True,   # ascending
}


def load_results():
    data = {}
    for key, path in CSV_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV file: {path}")
        data[key] = pd.read_csv(path)
    return data


def aggregate_over_seeds(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Average metrics across seeds for each model (collapse different seeds within the same group)."""
    aggregated = {}
    for group, df in data.items():
        metrics_cols = [c for c in ["r2", "mse", "rmse", "mae"] if c in df.columns]
        if not metrics_cols:
            continue
        grouped = df.groupby("model", as_index=False)[metrics_cols].mean()
        if "seed" in df.columns:
            grouped["seed_count"] = df.groupby("model")["seed"].nunique().values
        aggregated[group] = grouped
    return aggregated


def rank_and_save_per_group(data: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    ranked: dict[str, dict[str, pd.DataFrame]] = {}
    for group, df in data.items():
        ranked[group] = {}
        for metric, suffix in RANK_SUFFIX.items():
            asc = METRIC_ORDER[metric]
            sorted_df = df.sort_values(by=metric, ascending=asc).reset_index(drop=True)
            out_path = OUTPUT_DIR / f"{group}_{suffix}.csv"
            sorted_df.to_csv(out_path, index=False)
            ranked[group][metric] = sorted_df
            print(f"Saved ranking by {metric} for {group} -> {out_path}")
    return ranked


def plot_top10_per_group(ranked: dict[str, dict[str, pd.DataFrame]]):
    for group, metrics in ranked.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for ax, metric in zip(axes, ["r2", "mse", "rmse", "mae"]):
            top10 = metrics.get(metric)
            if top10 is None or top10.empty:
                continue
            subset = top10.head(10)
            ax.barh(subset["model"], subset[metric], color="steelblue")
            ax.set_title(f"{group} - Top 10 by {metric}")
            ax.invert_yaxis()
            ax.grid(True, axis="x", linestyle="--", alpha=0.5)
            ax.set_xlabel(metric)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f"top10_{group}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved top10 plots for {group} to {plot_path}")

def save_best_per_group_metric(ranked: dict[str, dict[str, pd.DataFrame]]):
    records = []
    for group, metrics in ranked.items():
        for metric, df in metrics.items():
            best = df.iloc[0].copy()
            best["group"] = group
            best["metric"] = metric
            records.append(best)
    if records:
        out_df = pd.DataFrame(records)
        out_path = OUTPUT_DIR / "best_per_group_metric.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved best-per-group-per-metric summary to {out_path}")


def main():
    raw = load_results()
    data = aggregate_over_seeds(raw)
    ranked = rank_and_save_per_group(data)
    plot_top10_per_group(ranked)
    save_best_per_group_metric(ranked)


if __name__ == "__main__":
    main()
