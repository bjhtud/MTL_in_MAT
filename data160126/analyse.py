import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

DATASETS = {
    "cp": ("cp/x_columns.txt", "cp/y_columns.txt"),
    "FEA": ("FEA/x_columns.txt", "FEA/y_columns.txt"),
    "pan": ("me/pan/pan_x_columns.txt", "me/pan/pan_y_columns.txt"),
    "trc": ("me/trc/trc_x_columns.txt", "me/trc/trc_y_columns.txt"),
}


def data_choice(data_idx):

    if data_idx not in DATASETS:
        raise ValueError(f"Unknown dataset key: {data_idx}")
    x_path, y_path = DATASETS[data_idx]
    with open(x_path, "r") as f:
        x_col = [line.strip() for line in f if line.strip()]
    with open(y_path, "r") as f:
        y_col = [line.strip() for line in f if line.strip()]
    return x_col, y_col


def data_loader(csv_path, dataset_key):

    x_cols, y_cols = data_choice(dataset_key)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in x_cols + y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[x_cols].to_numpy()
    y = df[y_cols].to_numpy()
    return X, y


def data_path(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    dataset_key = folder.name
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    return dataset_key, csv_files
    

def main():
    # ====== config ======
    test_count = 100
    train_sizes = [20, 50, 100, 200, 300, 400]
    seed = 42
    
    # ====================
    model_name = "./me/trc"
    output_csv_dir = Path(model_name) / "csv"
    dataset_key, csv_files = data_path(model_name)
    
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    r2_by_size = {n: [] for n in train_sizes}
    target_r2_by_size = {}
    num_targets = None

    for csv_path in csv_files:

        X, y = data_loader(csv_path, dataset_key)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if num_targets is None:
            num_targets = y.shape[1]
            target_r2_by_size = {
                target_idx: {n: [] for n in train_sizes}
                for target_idx in range(num_targets)
            }
        elif y.shape[1] != num_targets:
            raise ValueError(
                f"Inconsistent target count in {csv_path}: {y.shape[1]} vs {num_targets}"
            )
        if test_count >= len(X):
            raise ValueError(
                f"test-count {test_count} must be smaller than total samples {len(X)}"
            )

        train_pool = len(X) - test_count
        if max(train_sizes) > train_pool:
            raise ValueError(
                f"Max train size {max(train_sizes)} exceeds available train pool {train_pool}"
            )

        rng = np.random.default_rng(seed)
        X_test = X[-test_count:]
        y_test = y[-test_count:]
        train_indices_map = {
            n_train: rng.choice(train_pool, size=n_train, replace=False)
            for n_train in train_sizes
        }

        results = []
        for n_train in train_sizes:
            train_indices = train_indices_map[n_train]
            X_train = X[train_indices]
            y_train = y[train_indices]
            print(f'training with x {X_train.shape}, y {y_train.shape}')
            model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                objective="reg:squarederror",
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            results.append((n_train, r2))
            r2_by_size[n_train].append(r2)

        results.sort(key=lambda x: x[0])
        train_sizes_sorted = [r[0] for r in results]
        r2_scores = [r[1] for r in results]
        
        result_df = pd.DataFrame(results, columns=["train_size", "r2"])
        result_df.to_csv(output_csv_dir / f"{csv_path.stem}_r2.csv", index=False)

        print(json.dumps({"results": results, "csv": str(csv_path)}))

        if num_targets > 1:
            for target_idx in range(num_targets):
                target_results = []
                y_test_col = y_test[:, target_idx]
                for n_train in train_sizes:
                    train_indices = train_indices_map[n_train]
                    X_train = X[train_indices]
                    y_train = y[train_indices, target_idx]
                    model = XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        objective="reg:squarederror",
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test_col, preds)
                    target_results.append((n_train, r2))
                    target_r2_by_size[target_idx][n_train].append(r2)

                target_results.sort(key=lambda x: x[0])
                target_df = pd.DataFrame(target_results, columns=["train_size", "r2"])
                target_df.to_csv(
                    output_csv_dir / f"{csv_path.stem}_r2_target{target_idx}.csv",
                    index=False,
                )
                print(
                    json.dumps(
                        {
                            "target_idx": target_idx,
                            "results": target_results,
                            "csv": str(csv_path),
                        }
                    )
                )

    avg_results = []
    for n_train in train_sizes:
        scores = r2_by_size[n_train]
        avg_results.append((n_train, float(np.mean(scores))))

    train_sizes_sorted = [r[0] for r in avg_results]
    r2_scores = [r[1] for r in avg_results]
    output_plot = f"{model_name}/r2.png"
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_sorted, r2_scores, marker="o")
    plt.title("Average R2 vs Train Size")
    plt.xlabel("Train sample count")
    plt.ylabel("R2 score (mean of CSVs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)

    avg_df = pd.DataFrame(avg_results, columns=["train_size", "r2_mean"])
    avg_df.to_csv(output_csv_dir / "avg_r2.csv", index=False)

    print(json.dumps({"avg_results": avg_results, "plot": str(Path(output_plot).resolve())}))

    if num_targets and num_targets > 1:
        for target_idx in range(num_targets):
            target_avg_results = []
            for n_train in train_sizes:
                scores = target_r2_by_size[target_idx][n_train]
                target_avg_results.append((n_train, float(np.mean(scores))))

            target_sizes = [r[0] for r in target_avg_results]
            target_scores = [r[1] for r in target_avg_results]
            target_plot = Path(model_name) / f"r2_target{target_idx}.png"
            plt.figure(figsize=(8, 5))
            plt.plot(target_sizes, target_scores, marker="o")
            plt.title(f"Average R2 vs Train Size (target {target_idx})")
            plt.xlabel("Train sample count")
            plt.ylabel("R2 score (mean of CSVs)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(target_plot, dpi=150)

            target_df = pd.DataFrame(
                target_avg_results, columns=["train_size", "r2_mean"]
            )
            target_df.to_csv(
                output_csv_dir / f"avg_r2_target{target_idx}.csv", index=False
            )
            print(
                json.dumps(
                    {
                        "target_idx": target_idx,
                        "avg_results": target_avg_results,
                        "plot": str(target_plot.resolve()),
                    }
                )
            )

if __name__ == "__main__":
    main()
