import sys
import warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

from uhpc.class_method import (
    DirectMissingModels,
    FeatureMissingModels,
    CompleteDataModels,
    scale_features,
    _slug,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'uhpc'))

_log_handle = None
_err_handle = None
_warn_handle = None


def _setup_logging(log_path: Path, err_path: Path, warn_path: Path):
    """Initializer for Pool: redirect stdout/stderr and warnings inside each worker."""
    global _log_handle, _err_handle, _warn_handle
    _log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    _err_handle = open(err_path, "a", encoding="utf-8", buffering=1)
    _warn_handle = open(warn_path, "a", encoding="utf-8", buffering=1)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        _warn_handle.write(warnings.formatwarning(message, category, filename, lineno, line))
        _warn_handle.flush()

    warnings.showwarning = _showwarning
    warnings.filterwarnings("default")
    sys.stdout = _log_handle
    sys.stderr = _err_handle

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
        'complete': complete.run().assign(seed=seed),
        'direct_label': direct.last_label_metrics.assign(seed=seed) if not direct.last_label_metrics.empty else pd.DataFrame(),
        'feature_label': feature.last_label_metrics.assign(seed=seed) if not feature.last_label_metrics.empty else pd.DataFrame(),
        'complete_label': complete.last_label_metrics.assign(seed=seed) if not complete.last_label_metrics.empty else pd.DataFrame(),
    }
    return results

def main():
    seeds = [40]
    aggregated = {
        'direct': [],
        'feature': [],
        'complete': []
    }
    aggregated_label = {
        'direct_label': [],
        'feature_label': [],
        'complete_label': [],
    }

    n_proc = min(len(seeds), cpu_count())

    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = log_dir / f'run_{timestamp}.log'
    warn_file = log_dir / f'warn_{timestamp}.log'
    err_file = log_dir / f'error_{timestamp}.log'

    with log_file.open('w', encoding='utf-8') as log_f, \
         warn_file.open('w', encoding='utf-8') as warn_f, \
         err_file.open('w', encoding='utf-8') as err_f:

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            warn_f.write(warnings.formatwarning(message, category, filename, lineno, line))
            warn_f.flush()

        warnings.showwarning = _showwarning
        warnings.filterwarnings("default")

        with redirect_stdout(log_f), redirect_stderr(err_f):
            with Pool(
                processes=n_proc,
                initializer=_setup_logging,
                initargs=(log_file, err_file, warn_file),
            ) as pool:
                results_list = pool.map(run_for_seed, seeds)

            for res in results_list:
                for key in aggregated:
                    aggregated[key].append(res[key])
                for key in aggregated_label:
                    if not res[key].empty:
                        aggregated_label[key].append(res[key])

            output_dir = PROJECT_ROOT / 'results_multi_seed'
            output_dir.mkdir(exist_ok=True)

            for key, frames in aggregated.items():
                combined = pd.concat(frames, ignore_index=True)
                combined.to_csv(output_dir / f'{key}_results.csv', index=False)
                print(f"Saved {key} results to {output_dir / f'{key}_results.csv'}")

            # 保存按标签拆分的结果：文件名 = 原规则 + _{label}
            for key, frames in aggregated_label.items():
                if not frames:
                    continue
                combined_label = pd.concat(frames, ignore_index=True)
                if combined_label.empty or 'label' not in combined_label:
                    continue
                base = key.replace('_label', '')
                for label_name, df_label in combined_label.groupby('label'):
                    safe_label = _slug(label_name)
                    out_path = output_dir / f'{base}_results_{safe_label}.csv'
                    df_label.to_csv(out_path, index=False)
                    print(f"Saved {base} label `{label_name}` results to {out_path}")

if __name__ == '__main__':
    main()
