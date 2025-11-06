import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Literal, List, Dict, Tuple

ImputeType = Literal['zero', 'mean', 'knn', 'drop']

def _imputer(impute: ImputeType):
    if impute == 'zero':
        return SimpleImputer(strategy='constant', fill_value=0)
    elif impute == 'mean':
        return SimpleImputer(strategy='mean')
    elif impute == 'knn':
        return KNNImputer(n_neighbors=5, weights='distance')
    elif impute == 'drop':
        return None

def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'rmse': float(rmse)
    }
def _avg(per_target: Dict[str, Dict[str, float]]):
    keys = list(next(iter(per_target.values())).keys())
    return {f'avg_{k}': float(np.mean([per_target[t][k] for t in per_target])) for k in keys}

def prepare_features(
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        impute: ImputeType
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if impute == 'drop':
        mask_tr = ~x_train.isna().any(axis=1)
        Xtr = x_train.loc[mask_tr].to_numpy()
        ytr = y_train[mask_tr].copy()
        if Xtr.shape[0] == 0:
            raise ValueError('Xtr and Xte must have at least one sample.')
        return Xtr, ytr

    else:
        imputer = _imputer(impute)
        Xtr = imputer.fit_transform(x_train.values)
        return Xtr, y_train.copy()

def train_eval(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        order: List[str],
        impute: ImputeType = 'mean',
        base_model = xgb,
        random_state: int = 42,
        ):

    base_model = base_model.XGBRegressor(random_state=random_state)

    Xtr, ytr= prepare_features(X_train, y_train, impute)
    y_train_aligned = ytr
    y_cols = list(y_train.columns)

    models_1, preds1_train, preds1_test = {}, {}, {}
    for yk in y_cols:
        y = ytr[yk].to_numpy().astype(float)
        has_y = ~np.isnan(y)
        m1 = clone(base_model)
        m1.fit(Xtr[has_y], y[has_y])
        models_1[yk] = m1
        preds1_train[yk] = m1.predict(Xtr)
        preds1_test[yk] = m1.predict(X_test)

    models_2, preds2_test = {}, {}
    for yk in y_cols:
        others = [c for c in y_cols if c != yk]

        other_tr = []
        for oj in others:
            tcol = y_train_aligned[oj].to_numpy().astype(float)
            fill = np.where(~np.isnan(tcol), tcol, preds1_train[oj]).reshape(-1,1)
            other_tr.append(fill)
        Xtr_aug = np.hstack([Xtr] + other_tr) if other_tr else Xtr

        y_tgt = y_train_aligned[yk].to_numpy().astype(float)
        has_tgt = ~np.isnan(y_tgt)

        m2 = clone(base_model)
        m2.fit(Xtr_aug[has_tgt], y_tgt[has_tgt])
        models_2[yk] = m2

        other_te = [preds1_test[oj].reshape(-1,1) for oj in others]
        Xte_aug = np.hstack([X_test] + other_te) if other_te else X_test
        preds2_test[yk] = m2.predict(Xte_aug)

    per_target = {
        yk: _metrics(y_test[yk].to_numpy(), preds2_test[yk]) for yk in y_cols
    }
    macro_avg = _avg(per_target)
    per_target_ordered = {yk: per_target[yk] for yk in order}
    return per_target_ordered, macro_avg
