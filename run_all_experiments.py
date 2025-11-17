import sys
import warnings
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count

from uhpc.class_method import (
    DirectMissingModels,
    FeatureMissingModels,
    CompleteDataModels,
    scale_features,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'uhpc'))

def load_data():
    for candidate in [PROJECT_ROOT / 'uhpc' / 'UHPC.xlsx', PROJECT_ROOT / 'UHPC.xlsx']:
        if candidate.exists():
            df = pd.read_excel(candidate)
            break
    else:
        raise FileNotFoundError("UHPC.xlsx 不存在，请确认文件位置")

    complete_mask = df.notna().all(axis=1)
    train_df = df.loc[~complete_mask].reset_index(drop=True)
    test_df = df.loc[complete_mask].reset_index(drop=True)

    X_train_raw = train_df.iloc[:, :24]
    y_train = train_df.iloc[:, 24:]
    X_test_raw = test_df.iloc[:, :24]
    y_test = test_df.iloc[:, 24:]

    X_train, X_test = scale_features(X_train_raw, X_test_raw)
    return X_train, y_train, X_test, y_test


def run_for_seed(seed):
    X_train, y_train, X_test, y_test = load_data()
    direct = DirectMissingModels(X_train, y_train, X_test, y_test, seed=seed)
    feature = FeatureMissingModels(X_train, y_train, X_test, y_test, seed=seed)
    complete = CompleteDataModels(X_train, y_train, X_test, y_test, seed=seed)

    results = {
        'direct': direct.run().assign(seed=seed),
        'feature': feature.run().assign(seed=seed),
        'complete': complete.run().assign(seed=seed)
    }
    return results

def main():
    seeds = [10, 20, 30, 40, 50]
    aggregated = {
        'direct': [],
        'feature': [],
        'complete': []
    }

    n_proc = min(len(seeds), cpu_count())

    with Pool(processes=n_proc) as pool:

        results_list = pool.map(run_for_seed, seeds)

    for res in results_list:
        for key in aggregated:
            aggregated[key].append(res[key])

    output_dir = PROJECT_ROOT / 'results_multi_seed'
    output_dir.mkdir(exist_ok=True)

    for key, frames in aggregated.items():
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_dir / f'{key}_results.csv', index=False)
        print(f"Saved {key} results to {output_dir / f'{key}_results.csv'}")

if __name__ == '__main__':
    main()
