'''
LightGBM：use_missing 参数可以处理缺失值。
'''
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

def _make_masks(train_y, test_y):
    # 单任务：Series；多任务：DataFrame
    if isinstance(train_y, pd.DataFrame):
        train_mask = ~train_y.isna().any(axis=1)
    else:
        train_mask = ~pd.Series(train_y).isna()

    if isinstance(test_y, pd.DataFrame):
        valid_mask = ~test_y.isna().any(axis=1)
    else:
        valid_mask = ~pd.Series(test_y).isna()
    return train_mask, valid_mask

def ligbm_fit_predict(train_x: pd.DataFrame,
                      train_y: pd.Series | pd.DataFrame,
                      test_x: pd.DataFrame,
                      test_y: pd.Series | pd.DataFrame):
    # 确保是 DataFrame / Series
    Xtr = pd.DataFrame(train_x)
    Xte = pd.DataFrame(test_x)
    ytr_in = train_y
    yte_in = test_y

    # 掩码（按行过滤缺失标签）
    train_mask, valid_mask = _make_masks(ytr_in, yte_in)
    X_tr = Xtr.loc[train_mask]
    X_val = Xte.loc[valid_mask]
    # y 同步过滤
    if isinstance(ytr_in, pd.DataFrame):
        y_tr = ytr_in.loc[train_mask]
    else:
        y_tr = pd.Series(ytr_in).loc[train_mask]
    if isinstance(yte_in, pd.DataFrame):
        y_val = yte_in.loc[valid_mask]
    else:
        y_val = pd.Series(yte_in).loc[valid_mask]

    # 经验性参数，缓解“无有效特征”警告
    base_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.0,
        reg_alpha=0.0,
        n_jobs=-1,
        # 关键：放宽前置过滤与叶/桶约束
        feature_pre_filter=False,
        min_data_in_leaf=20,
        min_data_in_bin=1,
        verbosity=-1
    )

    # 如果验证集有效样本太少，就不用 early_stopping
    use_es = len(X_val) >= 20

    # ---------- 单输出 ----------
    if isinstance(y_tr, pd.Series):
        reg = lgb.LGBMRegressor(**base_params)

        if use_es:
            reg.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
            )
        else:
            reg.fit(X_tr, y_tr)

        y_pred = reg.predict(Xte)
        # 形状/类型对齐
        if isinstance(yte_in, pd.Series):
            return pd.Series(y_pred, index=Xte.index, name=yte_in.name)
        else:  # 单列 DataFrame
            return pd.DataFrame(y_pred, index=Xte.index, columns=yte_in.columns)

    # ---------- 多输出 ----------
    # LGBMRegressor 不支持多列标签；用 MultiOutputRegressor 包装
    reg_base = lgb.LGBMRegressor(**base_params)
    reg = MultiOutputRegressor(reg_base)

    if use_es:
        # MultiOutputRegressor 不支持 callbacks 透传到每个子模型的 eval_set
        # 这里简化：不做 early_stopping；如果你想做，可按列循环手写训练
        reg.fit(X_tr, y_tr)
    else:
        reg.fit(X_tr, y_tr)

    Y_pred = reg.predict(Xte)  # ndarray (m, T)
    # 对齐类型/索引/列名
    return pd.DataFrame(Y_pred, index=Xte.index, columns=ytr_in.columns)

