import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from Input_space_expansion.Multi_target import erc, sst
from uhpc.Class import BaselineImputer
from uhpc.method.CatBoost import catboost_fit_predict
from uhpc.method.GBDT import gbdt_fit_predict
from uhpc.method.HistGradientBoostingRegressor import hgbr_fit_predict
from uhpc.method.HMlasso import hmlasso_fit_predict
from uhpc.method.LightGBM import ligbm_fit_predict
from uhpc.method.MTExtraTress import mtet_fit_predict
from uhpc.method.MultitaskGP import model_fit_predict as multitask_gp_predict
from uhpc.method.XGBoost import xgboost_fit_predict


def _as_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame().copy()
    return pd.DataFrame(data).copy()


def _match_template(df_like, template: pd.DataFrame) -> pd.DataFrame:
    df = _as_frame(df_like)
    if df.shape[1] != template.shape[1]:
        raise ValueError("列数量与模板不一致，无法对齐。")
    df = df.reset_index(drop=True)
    if df.shape[0] != template.shape[0]:
        raise ValueError("行数量与模板不一致，无法对齐。")
    df.index = template.index
    df.columns = template.columns
    return df


def _select_model_names(requested, registry):
    if requested is None:
        return list(registry.keys())
    unknown = sorted(set(requested) - set(registry.keys()))
    if unknown:
        raise ValueError(f"未知模型: {unknown}. 可选项: {list(registry.keys())}")
    return requested


def _valid_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    y_true = _as_frame(y_true)
    y_pred = _match_template(y_pred, y_true)
    mask = ~y_true.isna().any(axis=1)
    if not mask.any():
        raise ValueError("目标值全为 NaN，无法计算指标。")
    y_true_valid = y_true.loc[mask]
    y_pred_valid = y_pred.loc[mask]
    return {
        "r2": r2_score(y_true_valid, y_pred_valid, multioutput="uniform_average"),
        "mse": mean_squared_error(y_true_valid, y_pred_valid),
        "mae": mean_absolute_error(y_true_valid, y_pred_valid),
    }


def _scale_with_missing(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scaled = X_train.copy()
    test_scaled = X_test.copy()

    for col in X_train.columns:
        col_train = X_train[col]
        valid = col_train.dropna()
        if valid.empty:
            continue

        mean = valid.mean()
        std = valid.std(ddof=0)

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

        self.X_train, self.X_test = scale_features(_as_frame(X_train), _as_frame(X_test))
        self.y_train = _as_frame(y_train)
        self.y_test = _as_frame(y_test)

    def _score_predictions(self, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
        records = []
        for name, pred in predictions.items():
            df_pred = _match_template(pred, self.y_test)
            metrics = _valid_metrics(self.y_test, df_pred)
            metrics["model"] = name
            records.append(metrics)
        return pd.DataFrame(records)

    def _impute_labels(self, method: str) -> pd.DataFrame:
        _, y_imp = BaselineImputer(random_state=self.seed).impute(self.X_train, self.y_train, method=method)
        y_imp = _match_template(y_imp, self.y_train)
        return y_imp

    def _impute_all(self, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x_imp_train, y_imp_train = BaselineImputer(random_state=self.seed).impute(self.X_train, self.y_train, method=method)
        x_imp_test, _ = BaselineImputer(random_state=self.seed).impute(self.X_test, self.y_test, method=method)
        X_train_filled = _match_template(x_imp_train, self.X_train)
        y_train_filled = _match_template(y_imp_train, self.y_train)
        X_test_filled = _match_template(x_imp_test, self.X_test)
        return X_train_filled, y_train_filled, X_test_filled


class DirectMissingModels(BaseDataset):
    """
    类别一：模型能够同时处理特征和标签的缺失，直接输入原始 DataFrame。
    """

    MODEL_REGISTRY = {
        "catboost": lambda obj: catboost_fit_predict(obj.X_train, obj.y_train, obj.X_test),
        "mt_extra_trees": lambda obj: mtet_fit_predict(obj.X_train, obj.y_train, obj.X_test, obj.y_test),
    }

    def run(self, model_names: list[str] | None = None) -> pd.DataFrame:
        names = _select_model_names(model_names, self.MODEL_REGISTRY)
        predictions = {}
        for name in names:
            predictions[name] = self.MODEL_REGISTRY[name](self)
        return self._score_predictions(predictions)


class FeatureMissingModels(BaseDataset):
    """
    类别二：模型可以处理特征缺失，但不能处理标签缺失。
    先对标签做插补，再将原始特征（含缺失）与插补标签一同训练。
    """

    def __init__(self, *args, label_impute_method: str = "missforest", **kwargs):
        super().__init__(*args, **kwargs)
        self.label_impute_method = label_impute_method

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
            "sst_xgb": pd.DataFrame(pred_sst, index=self.X_test.index, columns=y_filled.columns),
            "erc_xgb": pd.DataFrame(pred_erc, index=self.X_test.index, columns=y_filled.columns),
        }

    def run(self, model_names: list[str] | None = None, include_iterative: bool = True) -> pd.DataFrame:
        y_filled = self._impute_labels(self.label_impute_method)

        registry = {
            "lightgbm": lambda: ligbm_fit_predict(self.X_train, y_filled, self.X_test, self.y_test),
            "hist_gbr": lambda: hgbr_fit_predict(self.X_train, y_filled, self.X_test),
            "xgboost": lambda: xgboost_fit_predict(self.X_train, y_filled, self.X_test),
            "hmlasso": lambda: self._run_hmlasso(y_filled),
        }

        names = _select_model_names(model_names, registry)
        predictions = {}
        for name in names:
            predictions[f"{name}__{self.label_impute_method}"] = registry[name]()

        if include_iterative:
            iterative_preds = self._iterative_predictions(y_filled)
            for name, pred in iterative_preds.items():
                predictions[f"{name}__{self.label_impute_method}"] = pred

        return self._score_predictions(predictions)


class CompleteDataModels(BaseDataset):
    """
    类别三：模型要求特征和标签均完整，需要先同时对 X、y 做插补。
    """

    def __init__(self, *args, impute_method: str = "missforest", **kwargs):
        super().__init__(*args, **kwargs)
        self.impute_method = impute_method

    def _multioutput_gp(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        base = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=self.seed)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return pd.DataFrame(pred, index=X_test.index, columns=y_train.columns)

    def _multioutput_svr(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        base = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return pd.DataFrame(pred, index=X_test.index, columns=y_train.columns)

    def run(self, model_names: list[str] | None = None, include_multitask_gp: bool = True) -> pd.DataFrame:
        X_train_filled, y_train_filled, X_test_filled = self._impute_all(self.impute_method)

        registry = {
            "random_forest": lambda: pd.DataFrame(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    random_state=self.seed,
                    n_jobs=-1,
                ).fit(X_train_filled, y_train_filled).predict(X_test_filled),
                index=X_test_filled.index,
                columns=y_train_filled.columns,
            ),
            "ridge": lambda: pd.DataFrame(
                Ridge(alpha=1.0, random_state=self.seed).fit(X_train_filled, y_train_filled).predict(X_test_filled),
                index=X_test_filled.index,
                columns=y_train_filled.columns,
            ),
            "multitask_lasso": lambda: pd.DataFrame(
                MultiTaskLasso(alpha=0.001, random_state=self.seed, max_iter=5000).fit(X_train_filled, y_train_filled).predict(X_test_filled),
                index=X_test_filled.index,
                columns=y_train_filled.columns,
            ),
            "multitask_elasticnet": lambda: pd.DataFrame(
                MultiTaskElasticNet(alpha=0.001, l1_ratio=0.5, random_state=self.seed, max_iter=5000).fit(X_train_filled, y_train_filled).predict(X_test_filled),
                index=X_test_filled.index,
                columns=y_train_filled.columns,
            ),
            "multioutput_gp": lambda: self._multioutput_gp(X_train_filled, y_train_filled, X_test_filled),
            "multioutput_svr": lambda: self._multioutput_svr(X_train_filled, y_train_filled, X_test_filled),
            "gbdt": lambda: gbdt_fit_predict(X_train_filled, y_train_filled, X_test_filled),
        }

        if include_multitask_gp:
            registry["multitask_gp"] = lambda: multitask_gp_predict(X_train_filled, y_train_filled, X_test_filled)

        names = _select_model_names(model_names, registry)
        predictions = {}
        for name in names:
            predictions[f"{name}__{self.impute_method}"] = registry[name]()

        return self._score_predictions(predictions)
