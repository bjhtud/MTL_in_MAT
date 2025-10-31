'''
CatBoost (MultiRMSE / MultiRMSEWithMissingValues) 显式建模标签/任务间相似性
标签部分缺失，特征缺失
'''
from catboost import CatBoostRegressor
import pandas as pd


def catboost_fit_predict(train_x, train_y, test_x):
    reg_multi = CatBoostRegressor(
        loss_function="MultiRMSEWithMissingValues",  # 关键
        nan_mode="Min",
        # 注意：该损失用于优化，但官方文档标注 GPU 不支持
        verbose=False
    )
    reg_multi.fit(train_x, train_y)  # 会忽略 Y 中的 NaN 条目
    pred_y = reg_multi.predict(test_x)
    pred_y = pd.DataFrame(pred_y, index=test_x.index, columns=train_y.columns)
    return pred_y