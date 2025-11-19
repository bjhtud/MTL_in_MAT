import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from uhpc.Input_space_expansion import erc, sst
from uhpc.Class import BaselineImputer, Subset
from uhpc.method.CatBoost import catboost_fit_predict
from uhpc.method.GBDT import gbdt_fit_predict
from uhpc.method.HistGradientBoostingRegressor import hgbr_fit_predict
from uhpc.method.HMlasso import hmlasso_fit_predict
from uhpc.method.LightGBM import ligbm_fit_predict
from uhpc.method.MTExtraTress import mtet_fit_predict
from uhpc.method.MultitaskGP import model_fit_predict as multitask_gp_predict
from uhpc.method.XGBoost import xgboost_fit_predict

IMPUTATION_METHODS = [
    ('MissForest', 'missforest'),
    ('RFE-MissForest', 'RFE_mf'),
    ('Hyperimpute', 'hyperimpute'),
    ('KNN based method (MatImputer)', 'MatImputer'),
    ('GAIN', 'gain'),
    ('Sinkhorn', 'sinkhorn'),
    ('MICE (Iterativeimputer)', 'MICE'),
    ('KNN插补', 'KNN'),
    ('MIDA', 'MIDA'),
    ('MIWAE', 'miwae'),
    ('SoftImpute', 'softimpute'),
    ('低秩矩阵', 'lm'),
    ('堆叠', 'stacking'),
    ('VAE', 'vae'),
    ('vanilla', 'vanilla'), 
    ('vae_miwae', 'vae_miwae'), 
    ('h_vae', 'h_vae'),
    ('hver', 'hver'),
    ('hmc_vae', 'hmc_vae'),
    ('hh_vaem', 'hh_vaem'),
    ('gtmcc', 'gtmcc'),
    ('subset', 'subset')
    ]

def _slug(text: str) -> str:
    allowed = []
    for ch in str(text):
        if ch.isalnum():
            allowed.append(ch.lower())
        else:
            allowed.append("_")
    slug = "".join(allowed)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")

def _as_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame().copy()
    return pd.DataFrame(data).copy()


def _match_template(df_like, template: pd.DataFrame) -> pd.DataFrame:
    df = _as_frame(df_like)
    if df.shape[1] != template.shape[1]:
        raise ValueError('列数量与模板不一致，无法对齐。')
    df = df.reset_index(drop=True)
    if df.shape[0] != template.shape[0]:
        raise ValueError('行数量与模板不一致，无法对齐。')
    df.index = template.index
    df.columns = template.columns
    return df


def _select_model_names(requested, registry):
    if requested is None:
        return list(registry.keys())
    unknown = sorted(set(requested) - set(registry.keys()))
    if unknown:
        raise ValueError(f'未知模型: {unknown}. 可选项: {list(registry.keys())}')
    return requested


def _valid_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    y_true = _as_frame(y_true)
    y_pred = _match_template(y_pred, y_true)
    mask = ~y_true.isna().any(axis=1)
    if not mask.any():
        raise ValueError('目标值全为 NaN，无法计算指标。')
    y_true_valid = y_true.loc[mask]
    y_pred_valid = y_pred.loc[mask]
    return {
        'r2': r2_score(y_true_valid, y_pred_valid, multioutput='uniform_average'), #'raw_values'
        'mse': mean_squared_error(y_true_valid, y_pred_valid, multioutput='uniform_average'),
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid, multioutput='uniform_average')),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid, multioutput='uniform_average'),
    }

def _valid_metrics_label(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    y_true = _as_frame(y_true)
    y_pred = _match_template(y_pred, y_true)
    mask = ~y_true.isna().any(axis=1)
    if not mask.any():
        raise ValueError('目标值全为 NaN，无法计算指标。')
    y_true_valid = y_true.loc[mask]
    y_pred_valid = y_pred.loc[mask]
    return {
        'r2': r2_score(y_true_valid, y_pred_valid, multioutput='raw_values'), #'raw_values'
        'mse': mean_squared_error(y_true_valid, y_pred_valid, multioutput='raw_values'),
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid, multioutput='raw_values')),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid, multioutput='raw_values'),
    }

def _scale_with_missing(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scaled = X_train.copy()
    test_scaled = X_test.copy()

    for col in X_train.columns:
        col_train = X_train[col]
        valid = col_train.dropna()
        if valid.empty: # .empty 全是缺失
            continue

        mean = valid.mean()
        std = valid.std(ddof=0) # 按列求标准差

        if std == 0 or np.isnan(std):
            train_scaled[col] = col_train - mean
            test_scaled[col] = X_test[col] - mean
            continue

        train_scaled[col] = (col_train - mean) / std
        test_scaled[col] = (X_test[col] - mean) / std

    return train_scaled, test_scaled

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    try:
        scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_test_scaled
    except ValueError:
        # 如果数据中有缺失值，则传入 _scale_with_missing 进行处理
        return _scale_with_missing(X_train, X_test)


class BaseDataset:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        seed: int = 42,
    ):
        self.seed = seed
        np.random.seed(self.seed)

        # 数据标准化
        self.X_train, self.X_test = scale_features(_as_frame(X_train), _as_frame(X_test)) 
        self.y_train = _as_frame(y_train)
        self.y_test = _as_frame(y_test)

        # 统一列名为字符串，避免列名类型混用导致下游模型报错
        self.X_train.columns = self.X_train.columns.map(str)
        self.X_test.columns = self.X_test.columns.map(str)
        self.y_train.columns = self.y_train.columns.map(str)
        self.y_test.columns = self.y_test.columns.map(str)
        self.last_label_metrics: pd.DataFrame = pd.DataFrame()

        # 磁盘缓存目录（用于容错与重复使用）
        root = Path(__file__).resolve().parent.parent
        self._cache_root = root / "run_results"
        self._impute_cache_dir = self._cache_root / "impute"
        self._regress_cache_dir = self._cache_root / "regress"
        self._impute_cache_dir.mkdir(parents=True, exist_ok=True)
        self._regress_cache_dir.mkdir(parents=True, exist_ok=True)

        # 对标签插补结果做缓存，适用于只能自动插补特征的模型
        self._label_impute_cache: dict[str, pd.DataFrame] = {}
        # 对完整插补结果做缓存，适用于需要同时插补特征和标签的模型
        self._full_impute_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    def _impute_cache_path(self, kind: str, method: str) -> Path:
        return self._impute_cache_dir / f"{kind}_seed{self.seed}_{_slug(method)}.pkl"

    def _save_impute_cache(self, path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(obj, path)

    def _load_impute_cache(self, path: Path):
        return pd.read_pickle(path)

    def _regress_cache_path(self, dataset_tag: str, imputer_label: str, model_name: str) -> Path:
        fname = f"{dataset_tag}_seed{self.seed}_{_slug(imputer_label)}_{_slug(model_name)}.csv"
        return self._regress_cache_dir / fname

    def _save_prediction_cache(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def _load_prediction_cache(self, path: Path, template: pd.DataFrame) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        return _match_template(df, template)

    def _score_predictions(self, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
        records = []
        label_records = []
        for name, pred in predictions.items():
            df_pred = _match_template(pred, self.y_test)
            metrics = _valid_metrics(self.y_test, df_pred)
            metrics['model'] = name
            records.append(metrics)
            try:
                label_metrics = _valid_metrics_label(self.y_test, df_pred)
                for idx, col in enumerate(self.y_test.columns):
                    label_records.append({
                        'model': name,
                        'label': col,
                        'r2': label_metrics['r2'][idx],
                        'mse': label_metrics['mse'][idx],
                        'rmse': label_metrics['rmse'][idx],
                        'mae': label_metrics['mae'][idx],
                    })
            except Exception as exc:
                print(f'[WARN] per-label metrics failed for {name}: {exc}')

        self.last_label_metrics = pd.DataFrame(label_records)
        return pd.DataFrame(records)

    def _score_predictions_label(self, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
        label_records = []
        for name, pred in predictions.items():
            df_pred = _match_template(pred, self.y_test)
            metrics = _valid_metrics_label(self.y_test, df_pred)
            for idx, col in enumerate(self.y_test.columns):
                label_records.append({
                    'model': name,
                    'label': col,
                    'r2': metrics['r2'][idx],
                    'mse': metrics['mse'][idx],
                    'rmse': metrics['rmse'][idx],
                    'mae': metrics['mae'][idx],
                })
        return pd.DataFrame(label_records)

    def _impute_labels(self, method: str) -> pd.DataFrame:
        cache_path = self._impute_cache_path("labels", method)
        if cache_path.exists():
            return self._load_impute_cache(cache_path)

        _, y_imp = BaselineImputer(random_state=self.seed).impute(self.X_train, self.y_train, method=method)
        y_imp = _match_template(y_imp, self.y_train)
        self._save_impute_cache(cache_path, y_imp)
        return y_imp

    def _get_imputed_labels(self, method: str) -> pd.DataFrame:
        if method not in self._label_impute_cache:
            self._label_impute_cache[method] = self._impute_labels(method)
        return self._label_impute_cache[method]

    def _impute_all(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cache_path = self._impute_cache_path("full", method)
        if cache_path.exists():
            cached = self._load_impute_cache(cache_path)
            return cached["X_train"], cached["y_train"], cached["X_test"]

        x_imp_train, y_imp_train = BaselineImputer(random_state=self.seed).impute(self.X_train, self.y_train, method=method)
        x_imp_test, _ = BaselineImputer(random_state=self.seed).impute(self.X_test, self.y_test, method=method)
        X_train_filled = _match_template(x_imp_train, self.X_train)
        y_train_filled = _match_template(y_imp_train, self.y_train)
        X_test_filled = _match_template(x_imp_test, self.X_test)
        payload = {"X_train": X_train_filled, "y_train": y_train_filled, "X_test": X_test_filled}
        self._save_impute_cache(cache_path, payload)
        return X_train_filled, y_train_filled, X_test_filled

    def _get_full_imputed_data(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if method not in self._full_impute_cache:
            self._full_impute_cache[method] = self._impute_all(method)
        return self._full_impute_cache[method]

    @staticmethod
    def _format_combo_name(model_name: str, imputer_label: str) -> str:
        def _slug(text: str) -> str:
            allowed = []
            for ch in text:
                if ch.isalnum():
                    allowed.append(ch.lower())
                else:
                    allowed.append('_')
            slug = ''.join(allowed)
            while '__' in slug:
                slug = slug.replace('__', '_')
            return slug.strip('_')

        return f'{_slug(model_name)}__{_slug(imputer_label)}'

class DirectMissingModels(BaseDataset):
    """
    Models that can handle missing features and labels directly without imputation.
    """
    MODEL_REGISTRY = {
        'catboost': lambda obj: catboost_fit_predict(obj.X_train, obj.y_train, obj.X_test),
        'mt_extra_trees': lambda obj: mtet_fit_predict(obj.X_train, obj.y_train, obj.X_test, obj.y_test),
    }

    def _run_subset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        runner = Subset(
            model=RandomForestRegressor,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            f_num=self.X_train.shape[1],
            random_state=self.seed,
        )
        label_results, overall_results = runner._get_output()

        records = []
        label_records = []
        for item in label_results:
            label_name = item.get('label', '')
            label_records.append({
                'model': self._format_combo_name('subset', label_name if label_name else 'label'),
                'label': label_name,
                'r2': item['r2'],
                'mse': item['mse'],
                'rmse': item['rmse'],
                'mae': item['mae'],
            })
        for item in overall_results:
            records.append({
                'model': self._format_combo_name('subset', 'overall'),
                'r2': item['r2'],
                'mse': item['mse'],
                'rmse': item['rmse'],
                'mae': item['mae'],
            })

        return pd.DataFrame(records), pd.DataFrame(label_records)

    def run(self, model_names: list[str] | None = None) -> pd.DataFrame:
        self.last_label_metrics = pd.DataFrame()
        registry = dict(self.MODEL_REGISTRY)
        registry['subset'] = None
        names = _select_model_names(model_names, registry)
        predictions = {}
        extra_metrics: list[pd.DataFrame] = []
        label_frames: list[pd.DataFrame] = []
        for name in names:
            if name == 'subset':
                try:
                    subset_metrics, subset_label_metrics = self._run_subset()
                except Exception as exc:
                    print(f'[WARN] subset failed: {exc}')
                    continue
                if not subset_metrics.empty:
                    extra_metrics.append(subset_metrics)
                if not subset_label_metrics.empty:
                    label_frames.append(subset_label_metrics)
                continue
            cache_path = self._regress_cache_path("direct", "none", name)
            if cache_path.exists():
                pred = self._load_prediction_cache(cache_path, self.y_test)
            else:
                print(f"[seed {self.seed}] model={name}")
                pred = self.MODEL_REGISTRY[name](self)
                self._save_prediction_cache(cache_path, pred)
            predictions[name] = pred

        frames = []
        if predictions:
            frames.append(self._score_predictions(predictions))
            if not self.last_label_metrics.empty:
                label_frames.append(self.last_label_metrics)
        else:
            self.last_label_metrics = pd.DataFrame()
        frames.extend(extra_metrics)
        if label_frames:
            self.last_label_metrics = pd.concat(label_frames, ignore_index=True)
        else:
            self.last_label_metrics = pd.DataFrame()
        if frames:
            return pd.concat(frames, ignore_index=True)
        
        return pd.DataFrame(columns=['r2', 'mse', 'rmse', 'mae', 'model'])
    

class FeatureMissingModels(BaseDataset):
    '''
    类别二：模型可以处理特征缺失，但不能处理标签缺失。
    先对标签做插补，再将原始特征（含缺失）与插补标签一同训练。
    '''
    def __init__(self, *args, impute_methods: list[tuple[str, str | None]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.impute_methods = impute_methods or IMPUTATION_METHODS

    def _run_hmlasso(self, y_filled: pd.DataFrame) -> pd.DataFrame:
        preds = []
        for col in y_filled.columns:
            pred_col = hmlasso_fit_predict(self.X_train, y_filled[col], self.X_test)
            preds.append(pd.Series(pred_col, index=self.X_test.index, name=col))
        return pd.concat(preds, axis=1)

    def _iterative_predictions(self, y_filled: pd.DataFrame) -> dict[str, pd.DataFrame]:
        x_train = self.X_train.to_numpy()
        y_train = y_filled.to_numpy()
        x_test = self.X_test.to_numpy()

        base_model = xgb.XGBRegressor(random_state=self.seed)

        model_sst = sst(model=base_model, cv=2, seed=self.seed, direct=False, verbose=False)
        model_sst.fit(x_train, y_train)
        pred_sst = model_sst.predict(x_test)

        model_erc = erc(model=base_model, cv=2, chain=3, seed=self.seed, direct=False, verbose=False)
        model_erc.fit(x_train, y_train)
        pred_erc = model_erc.predict(x_test)

        return {
            'sst_xgb': pd.DataFrame(pred_sst, index=self.X_test.index, columns=y_filled.columns),
            'erc_xgb': pd.DataFrame(pred_erc, index=self.X_test.index, columns=y_filled.columns),
        }

    def run(
        self,
        model_names: list[str] | None = None,
        include_iterative: bool = True,
        impute_methods: list[tuple[str, str | None]] | None = None,
    ) -> pd.DataFrame:
        self.last_label_metrics = pd.DataFrame()
        registry = {
            'lightgbm': lambda y_filled: ligbm_fit_predict(self.X_train, y_filled, self.X_test, self.y_test),
            'hist_gbr': lambda y_filled: hgbr_fit_predict(self.X_train, y_filled, self.X_test),
            'xgboost': lambda y_filled: xgboost_fit_predict(self.X_train, y_filled, self.X_test),
            'hmlasso': lambda y_filled: self._run_hmlasso(y_filled),
        }

        names = _select_model_names(model_names, registry)
        predictions = {}
        method_seq = impute_methods or self.impute_methods

        for display_name, method_key in method_seq:
            if method_key is None:
                print(f'[WARN] imputer `{display_name}` not implemented, skip')
                continue

            try:
                y_filled = self._get_imputed_labels(method_key)
            except Exception as exc:
                print(f'[WARN] imputer `{display_name}` failed: {exc}')
                continue
            if y_filled.isna().any().any():
                print(f'[WARN] imputer `{display_name}` left NaNs, skip prediction')
                continue

            for name in names:
                cache_path = self._regress_cache_path("feature", method_key, name)
                if cache_path.exists():
                    pred = self._load_prediction_cache(cache_path, self.y_test)
                else:
                    print(f"[seed {self.seed}] imputer={display_name} ({method_key}) model={name}")
                    pred = registry[name](y_filled)
                    self._save_prediction_cache(cache_path, pred)
                combo = self._format_combo_name(name, display_name)
                predictions[combo] = pred

            if include_iterative:
                iterative_preds = self._iterative_predictions(y_filled)
                for iter_name, pred in iterative_preds.items():
                    cache_path = self._regress_cache_path("feature", method_key, iter_name)
                    if cache_path.exists():
                        pred_cached = self._load_prediction_cache(cache_path, self.y_test)
                    else:
                        print(f"[seed {self.seed}] imputer={display_name} ({method_key}) model={iter_name}")
                        pred_cached = pred
                        self._save_prediction_cache(cache_path, pred_cached)
                    combo = self._format_combo_name(iter_name, display_name)
                    predictions[combo] = pred_cached

        if predictions:
            return self._score_predictions(predictions)
        self.last_label_metrics = pd.DataFrame()
        return pd.DataFrame(columns=['r2', 'mse', 'rmse', 'mae', 'model'])

class CompleteDataModels(BaseDataset):
    '''
    Models that require complete features and labels; impute X and y first.
    '''

    def __init__(self, *args, impute_methods: list[tuple[str, str | None]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.impute_methods = impute_methods or IMPUTATION_METHODS

    def _multioutput_gp(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        base = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=self.seed)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return pd.DataFrame(pred, index=X_test.index, columns=y_train.columns)

    def _multioutput_svr(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        base = SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return pd.DataFrame(pred, index=X_test.index, columns=y_train.columns)

    def run(
        self,
        model_names: list[str] | None = None,
        include_multitask_gp: bool = True,
        impute_methods: list[tuple[str, str | None]] | None = None,
    ) -> pd.DataFrame:
        self.last_label_metrics = pd.DataFrame()
        registry = {
            'random_forest': lambda Xtr, ytr, Xte: pd.DataFrame(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=self.seed,
                    n_jobs=-1,
                ).fit(Xtr, ytr).predict(Xte),
                index=Xte.index,
                columns=ytr.columns,
            ),
            'ridge': lambda Xtr, ytr, Xte: pd.DataFrame(
                Ridge(alpha=1.0, random_state=self.seed).fit(Xtr, ytr).predict(Xte),
                index=Xte.index,
                columns=ytr.columns,
            ),
            'multitask_lasso': lambda Xtr, ytr, Xte: pd.DataFrame(
                MultiTaskLasso(alpha=0.001, random_state=self.seed, max_iter=5000).fit(Xtr, ytr).predict(Xte),
                index=Xte.index,
                columns=ytr.columns,
            ),
            'multitask_elasticnet': lambda Xtr, ytr, Xte: pd.DataFrame(
                MultiTaskElasticNet(alpha=0.001, l1_ratio=0.5, random_state=self.seed, max_iter=5000).fit(Xtr, ytr).predict(Xte),
                index=Xte.index,
                columns=ytr.columns,
            ),
            'multioutput_gp': lambda Xtr, ytr, Xte: self._multioutput_gp(Xtr, ytr, Xte),
            'multioutput_svr': lambda Xtr, ytr, Xte: self._multioutput_svr(Xtr, ytr, Xte),
            'gbdt': lambda Xtr, ytr, Xte: gbdt_fit_predict(Xtr, ytr, Xte),
        }

        if include_multitask_gp:
            registry['multitask_gp'] = lambda Xtr, ytr, Xte: multitask_gp_predict(Xtr, ytr, Xte)

        names = _select_model_names(model_names, registry)
        predictions = {}
        method_seq = impute_methods or self.impute_methods

        for display_name, method_key in method_seq:
            if method_key == 'subset':
                print(f'[WARN] imputer `{display_name}` not supported for CompleteDataModels, skip')
                continue
            if method_key is None:
                print(f'[WARN] imputer `{display_name}` not implemented, skip')
                continue
            try:
                X_train_filled, y_train_filled, X_test_filled = self._get_full_imputed_data(method_key)
                if (X_train_filled.isna().any().any()
                    or y_train_filled.isna().any().any()
                    or X_test_filled.isna().any().any()):

                    print(f"[WARN] imputer `{display_name}` produced NaN, skip models")
                    continue
            except Exception as exc:
                print(f'[WARN] imputer `{display_name}` failed: {exc}')
                continue

            for name in names:
                cache_path = self._regress_cache_path("complete", method_key, name)
                if cache_path.exists():
                    pred = self._load_prediction_cache(cache_path, self.y_test)
                else:
                    print(f"[seed {self.seed}] imputer={display_name} ({method_key}) model={name}")
                    pred = registry[name](X_train_filled, y_train_filled, X_test_filled)
                    self._save_prediction_cache(cache_path, pred)
                combo = self._format_combo_name(name, display_name)
                predictions[combo] = pred

        if predictions:
            return self._score_predictions(predictions)
        self.last_label_metrics = pd.DataFrame()
        return pd.DataFrame(columns=['r2', 'mse', 'rmse', 'mae', 'model'])
