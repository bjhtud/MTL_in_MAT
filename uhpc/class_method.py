import pandas as pd
import numpy as np
from uhpc.method.LightGBM import ligbm_fit_predict
from uhpc.method.HistGradientBoostingRegressor import hgbr_fit_predict
from uhpc.method.HMlasso import hmlasso_fit_predict
from uhpc.method.GBDT import gbdt_fit_predict
from uhpc.method.CatBoost import catboost_fit_predict
from uhpc.method.XGBoost import xgboost_fit_predict
from uhpc.method.MTExtraTress import mtet_fit_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class BaseDataset():
    def __init__(self,
                 X_train:pd.DataFrame,
                 y_train:pd.DataFrame,
                 X_test:pd.DataFrame,
                 y_test:pd.DataFrame,
                 seed:int = 42):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.seed = seed

        np.random.seed(self.seed)

class ModelInter(BaseDataset): # 可以自动处理缺失的模型
    def __init__(self,
                 X_train:pd.DataFrame,
                 y_train:pd.DataFrame,
                 X_test:pd.DataFrame,
                 y_test:pd.DataFrame,
                 task_name:str,
                 seed:int = 42):

        super().__init__(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed)

        self.task_name = task_name

    def _impute(self):
        '''
        *LightGBM：use_missing    参数可以处理缺失值。
        *HistGradientBoostingRegressor：可以直接处理缺失值。
        *HMlasso
        *梯度提升决策树（GBDT）框架
        *CatBoost(MultiRMSE / MultiRMSEWithMissingValues) 显式建模标签 / 任务间相似性
        *XGBoost：可以直接处理缺失值。
        *MT - ExtraTrees(Tree - Based Ensemble Multi - Task Learning Method for Classification and Regression)
        '''
        # 数据准备
        train_x = self.X_train
        train_y = self.y_train
        test_x = self.X_test
        test_y = self.y_test
        task_name = self.task_name

        # 预测y
        lightgbm_pred_y = ligbm_fit_predict(train_x, train_y, test_x, test_y)
        hgbr_pred_y = hgbr_fit_predict(train_x, train_y, test_x) # 不支持标签缺失
        hmlasso_pred_y = hmlasso_fit_predict(train_x, train_y, test_x, task_name)
        gbdt_pred_y = gbdt_fit_predict(train_x, train_y, test_x, task_name)
        catboost_pred_y = catboost_fit_predict(train_x, train_y, test_x)
        xgboost_pred_y = xgboost_fit_predict(train_x, train_y, test_x)
        mtet_pred_y = mtet_fit_predict(train_x, train_y, test_x, test_y,task_name)

        # 计算误差
        lightgbm_r2 = r2_score(test_y, lightgbm_pred_y[task_name])
        lightgbm_mse = mean_squared_error(test_y, lightgbm_pred_y[task_name])
        lightgbm_mae = mean_absolute_error(test_y, lightgbm_pred_y[task_name])

        hgbr_r2 = r2_score(test_y, hgbr_pred_y[task_name])
        hgbr_mse = mean_squared_error(test_y, hgbr_pred_y[task_name])
        hgbr_mae = mean_absolute_error(test_y, hgbr_pred_y[task_name])

        hmlasso_r2 = r2_score(test_y, hmlasso_pred_y)
        hmlasso_mse = mean_squared_error(test_y, hmlasso_pred_y)
        hmlasso_mae = mean_absolute_error(test_y, hmlasso_pred_y)

        gbdt_r2 = r2_score(test_y, gbdt_pred_y)
        gbdt_mse = mean_squared_error(test_y, gbdt_pred_y)
        gbdt_mae = mean_absolute_error(test_y, gbdt_pred_y)

        catboost_r2 = r2_score(test_y, catboost_pred_y[task_name])
        catboost_mse = mean_squared_error(test_y, catboost_pred_y[task_name])
        catboost_mae = mean_absolute_error(test_y, catboost_pred_y[task_name])

        xgboost_r2 = r2_score(test_y, xgboost_pred_y[task_name])
        xgboost_mse = mean_squared_error(test_y, xgboost_pred_y[task_name])
        xgboost_mae = mean_absolute_error(test_y, xgboost_pred_y[task_name])

        mtet_r2 = r2_score(test_y, mtet_pred_y[task_name])
        mtet_mse = mean_squared_error(test_y, mtet_pred_y[task_name])
        mtet_mae = mean_absolute_error(test_y, mtet_pred_y[task_name])

        print(f'LightGBM R2: {lightgbm_r2:.3f}, MSE: {lightgbm_mse:.3f}, MAE: {lightgbm_mae:.3f}')
        print(f'HistGradientBoostingRegressor R2: {hgbr_r2:.3f}, MSE: {hgbr_mse:.3f}, MAE: {hgbr_mae:.3f}')
        print(f'HMlasso R2: {hmlasso_r2:.3f}, MSE: {hmlasso_mse:.3f}, MAE: {hmlasso_mae:.3f}')
        print(f'GBDT R2: {gbdt_r2:.3f}, MSE: {gbdt_mse:.3f}, MAE: {gbdt_mae:.3f}')
        print(f'CatBoost R2: {catboost_r2:.3f}, MSE: {catboost_mse:.3f}, MAE: {catboost_mae:.3f}')
        print(f'XGBoost R2: {xgboost_r2:.3f}, MSE: {xgboost_mse:.3f}, MAE: {xgboost_mae:.3f}')
        print(f'MT-ExtraTrees R2: {mtet_r2:.3f}, MSE: {mtet_mse:.3f}, MAE: {mtet_mae:.3f}')

class IterativeInter(BaseDataset): #迭代和堆叠
    '''
    * 堆叠(Multi-target regression via input space expansion: treating targets as inputs)
    * 回归链(同上，还有：leveraging multi-task learning regressor chains for small and sparse tabular data in materials design)
    '''
    def __init__(self):
        super().__init__()


from uhpc.method.MultitaskGP import model_fit_predict

class GPReg(BaseDataset): # 传统机器学习，基于贝叶斯推理
    '''
    * Multitask GP Regression (缺失特征：用 Uncertain Inputs——对缺失列设高斯分布 μ±σ，内核对该维做解析积分。GPyTorch里有完整例子。)
    '''
    def __init__(self,
                 X_train:pd.DataFrame,
                 y_train:pd.DataFrame,
                 X_test:pd.DataFrame,
                 y_test:pd.DataFrame,
                 task_name:str,
                 seed:int = 42):
        super().__init__(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed)
        self.task_name = task_name

    def _model(self):
        train_x = self.X_train
        train_y = self.y_train
        test_x = self.X_test
        test_y = self.y_test
        task_name = self.task_name
        pred_y = model_fit_predict(train_x, train_y, test_x)
        r2 = r2_score(test_y, pred_y[task_name])
        mse = mean_squared_error(test_y, pred_y[task_name])
        mae = mean_absolute_error(test_y, pred_y[task_name])
        print(f'Multitask GP Regression R2: {r2:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}')

from Class import BaselineImputer
class int_multi(BaseDataset): # 先插补再回归
    def __init__(self,
                 X_train:pd.DataFrame,
                 y_train:pd.DataFrame,
                 X_test:pd.DataFrame,
                 y_test:pd.DataFrame,
                 seed:int = 42):
        super().__init__(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed)
    def _imputer(self):
        X = self.X_train
        y = self.y_train
        x_imputed, y_imputed = BaselineImputer(random_state=self.seed).impute(X, y)
        return x_imputed, y_imputed

    def fit_pre(self):
        '''
        - 多任务学习模型(其实就是多元回归)：
        - 多输出随机森林
        - Ridge Regression 基于正则化
        - Lasso Regression(MT Lasso) 基于正则化
        - Elastic Net(MT Elastic Net) 基于正则化
        - 多输出GP
        - 多输出SVR (跟GP差不多)
        - autosklearn 支持多元回归
        '''
        x = self.X_train
        y = self.y_train
        test_x = self.X_test
        test_y = self.y_test
        x_imputed, y_imputed = self._imputer()


class SubSet(): # 子集模型
    def __init__(self):
        super().__init__()

