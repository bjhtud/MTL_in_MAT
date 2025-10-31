'''
HistGradientBoostingRegressor：可以直接处理缺失值。
'''
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd

def hgbr_fit_predict(train_x: pd.DataFrame, y_df: pd.DataFrame, test_x: pd.DataFrame) -> pd.DataFrame:

    mask = ~y_df.isna().any(axis=1)

    train_x = train_x.loc[mask]
    train_y = y_df.loc[mask]

    base = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, max_iter=300, random_state=0)
    reg = MultiOutputRegressor(base)
    reg.fit(train_x, train_y)
    pred = reg.predict(test_x)              # ndarray (m, T)
    return pd.DataFrame(pred, index=test_x.index, columns=y_df.columns)
