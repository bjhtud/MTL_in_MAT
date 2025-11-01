import pandas as pd
import numpy as np
from pandas import DataFrame

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


def _to_numpy(a):
    # DataFrame/Series -> numpy；已经是ndarray则保持
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return a.to_numpy()
    return np.asarray(a)
import xgboost as xgb
from Input_space_expansion.Multi_target import sst, erc
class IterativeInter(BaseDataset): #迭代和堆叠
    '''
    * 堆叠(Multi-target regression via input space expansion: treating targets as inputs)
    * 回归链(同上，还有：leveraging multi-task learning regressor chains for small and sparse tabular data in materials design)
    '''
    def __init__(self,
                 X_train:pd.DataFrame,
                 y_train:pd.DataFrame,
                 X_test:pd.DataFrame,
                 y_test:pd.DataFrame,
                 seed:int = 42):

        super().__init__(X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test, seed=seed)

    def fit_predict(self,task_name):
        x_train = self.X_train
        y_train = self.y_train
        x_test = self.X_test
        y_test = self.y_test
        list_of_target_names = y_test.columns.tolist()
        base_model = xgb.XGBRegressor()

        mask = ~y_train.isna().any(axis=1)

        x_train = x_train.loc[mask]
        y_train = y_train.loc[mask]

        x_train = _to_numpy(x_train)
        y_train = _to_numpy(y_train)
        x_test = _to_numpy(x_test)
        y_test = _to_numpy(y_test)

        model_sst = sst(model=base_model,
                    cv=2,
                    seed=1,
                    direct=False,
                    verbose=True
                    )

        j = list_of_target_names.index(task_name)
        model_sst.fit(x_train, y_train)
        RMSE_sst = model_sst.score(x_test, y_test[:, j])
        RRMSE_sst = model_sst.rrmse(x_test, y_test[:, j])

        model_erc = erc(model=base_model,
                    cv=2,
                    chain=3,
                    seed=1,
                    direct=False,
                    verbose=True,
                    )

        model_erc.fit(x_train, y_train)
        RMSE_erc = model_erc.score(x_test, y_test[:, j])
        RRMSE_erc = model_erc.rrmse(x_test, y_test[:, j])

        print("\n", "RMSE_SST: \n",RMSE_sst)
        print("\n", "RRMSE_SST: \n",RRMSE_sst)
        print("\n", "RMSE_erc: \n",RMSE_erc)
        print("\n", "RRMSE_erc: \n",RRMSE_erc)


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
from sklearn.ensemble import RandomForestRegressor
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
        imp_methods = ['missforest', #'RFE_mf',
                       'hyperimpute', #'MatImputer',
                       'gain', 'sinkhorn', #'MICE',
                       'KNN',
                       'MIDA', 'miwae',
                       #'softimpute',
                       'gtmcc', 'lm', 'InterativeImputer',
                       #'vae'
                       ]
        imputed_data = []
        for imp_method in imp_methods:
            print(f'Imputing method: {imp_method}\n')
            x_imputed, y_imputed = BaselineImputer(random_state=self.seed).impute(X, y, method=imp_method)

            x_imputed = (pd.DataFrame(x_imputed, index=X.index, columns=X.columns)
                         if not isinstance(x_imputed, pd.DataFrame) else x_imputed)
            y_imputed = (pd.DataFrame(y_imputed, index=y.index, columns=y.columns)
                         if not isinstance(y_imputed, pd.DataFrame) else y_imputed)

            imputed_data.append({'X': x_imputed,
                            'y': y_imputed,
                            'method': imp_method})

        return imputed_data, imp_methods

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
        X_test = self.X_test

        imputed_data, imp_methods = self._imputer()

        pred_data = []

        for data in imputed_data:
            X_train = data['X']
            y_train = data['y']
            method = data['method']

            # 1) 多输出随机森林（原生支持多输出）
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=0,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            pred_data.append({'method': method,
                              'y_pred': y_pred_rf,
                              'model': 'RandomForestRegressor'})

            # 2) Ridge Regression（原生支持多输出）
            from sklearn.linear_model import Ridge

            ridge = Ridge(alpha=1.0, random_state=0)
            ridge.fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            pred_data.append({'method': method,
                              'y_pred': y_pred_ridge,
                              'model': 'Ridge'})

            # 3) MultiTask Lasso（多任务套件，适合多输出）
            from sklearn.linear_model import MultiTaskLasso

            mt_lasso = MultiTaskLasso(alpha=0.001, random_state=0, max_iter=5000)
            mt_lasso.fit(X_train, y_train)
            y_pred_mt = mt_lasso.predict(X_test)
            pred_data.append({'method': method,
                              'y_pred': y_pred_mt,
                              'model': 'MultiTaskLasso'})

            # 4) MultiTask Elastic Net（多任务套件，适合多输出）
            from sklearn.linear_model import MultiTaskElasticNet

            mt_en = MultiTaskElasticNet(alpha=0.001, l1_ratio=0.5, random_state=0, max_iter=5000)
            mt_en.fit(X_train, y_train)
            y_pred_mt_en = mt_en.predict(X_test)
            pred_data.append({'method': method,
                              'y_pred': y_pred_mt_en,
                              'model': 'MultiTaskElasticNet'})

            # 5) 多输出高斯过程回归（用 MultiOutputRegressor 包一层）
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
            gpr_base = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=0)
            gpr = MultiOutputRegressor(gpr_base, n_jobs=-1)
            gpr.fit(X_train, y_train)
            y_pred_gpr = gpr.predict(X_test)
            pred_data.append({'method': method,
                              'y_pred': y_pred_gpr,
                              'model': 'MultiOutputRegressor'})

            # 6) 多输出 SVR（用 MultiOutputRegressor 包一层）
            from sklearn.svm import SVR
            from sklearn.multioutput import MultiOutputRegressor

            svr_base = SVR(C=10.0, epsilon=0.1, kernel='rbf', gamma='scale')
            svr = MultiOutputRegressor(svr_base, n_jobs=-1)
            svr.fit(X_train, y_train)
            y_pred_svr = svr.predict(X_test)
            pred_data.append({'method': method,
                              'y_pred': y_pred_svr,
                              'model': 'MultiOutputRegressor'})

            # 7) auto-sklearn（用 MultiOutputRegressor 包一层以适配多输出）
            # pip install auto-sklearn
            #from autosklearn.regression import AutoSklearnRegressor
            #from sklearn.multioutput import MultiOutputRegressor

            #auto_base = AutoSklearnRegressor(
            #    time_left_for_this_task=120,  # 总搜索时间（秒）
            #    per_run_time_limit=30,  # 单模型训练上限（秒）
            #    ensemble_size=25,
            #    seed=0
            #)
            #auto = MultiOutputRegressor(auto_base, n_jobs=-1)
            #auto.fit(X_train, y_train)
            #y_pred = auto.predict(X_test)
        return pred_data

class SubSet(): # 子集模型
    def __init__(self):
        super().__init__()

