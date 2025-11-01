import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def filled_result(df_full: pd.DataFrame, df_filled: pd.DataFrame, df_with_null: pd.DataFrame, miss_col: str) -> tuple:
    """
    Calculate the result of filling a column
    :param df_full: dataframe with no null values
    :param df_filled: dataframe with filled values
    :param df_with_null:  dataframe with null values
    :param miss_col: the missing column name
    :return:
    """
    miss_index = df_with_null[df_with_null[miss_col].isnull()].index.tolist()
    predict_data = df_filled.loc[miss_index, miss_col].values
    real_data = df_full.loc[miss_index, miss_col].values
    # mae = np.mean(np.abs(predict_data - real_data))
    mae = mean_absolute_error(real_data, predict_data)
    # rmse = np.sqrt(np.mean(np.square(predict_data - real_data)))
    rmse = root_mean_squared_error(real_data, predict_data)
    # r2 = 1 - np.sum(np.square(predict_data - real_data)) / np.sum(np.square(real_data - np.mean(real_data)))
    r2 = r2_score(real_data, predict_data)
    return mae, rmse, r2

def evaluate_wnd(imputed: pd.DataFrame, ground: pd.DataFrame) -> pd.DataFrame:
    res = 0
    for col in range(ground.shape[1]):
        res += wasserstein_distance(
            np.asarray(ground)[:, col], np.asarray(imputed)[:, col]
        )
    return res
# def evaluate_wnd(imputed: pd.DataFrame, ground: pd.DataFrame) -> float:
#     num_ground = ground.select_dtypes(include=[np.number]).columns
#     num_imputed = set(imputed.select_dtypes(include=[np.number]).columns)
#     cols = [c for c in num_ground if c in num_imputed]
#     if len(cols) == 0:
#         return float('nan')
#     wds = []
#     for c in cols:
#         g = ground[c].dropna().to_numpy()
#         i = imputed[c].dropna().to_numpy()
#         if g.size == 0 or i.size == 0:
#             continue
#         m = np.mean(g)
#         s = np.std(g, ddof=0)
#         if not np.isfinite(s) or s == 0.0:
#             s = 1.0
#         g_std = (g - m) / s
#         i_std = (i - m) / s
#         wds.append(float(wasserstein_distance(g_std, i_std)))
#     if len(wds) == 0:
#         return float('nan')
#     return float(np.mean(wds))
