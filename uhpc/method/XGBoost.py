'''
XGBoost：可以直接处理缺失值。
'''
import xgboost
import pandas as pd

def xgboost_fit_predict(train_x, train_y, test_x):
    mask = ~train_y.isna().any(axis=1)

    train_x = train_x.loc[mask]
    train_y = train_y.loc[mask]

    model = xgboost.XGBRegressor()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    pred_y = pd.DataFrame(pred_y, index=test_x.index, columns=train_y.columns)

    return pred_y