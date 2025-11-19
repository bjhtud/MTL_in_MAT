import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.experimental import enable_iterative_imputer   # 为了能够正常使用 InterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.linalg import svd, norm, inv
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from itertools import combinations

try:
    from hyperimpute.plugins.imputers import Imputers
except ImportError:
    Imputers = None

from .method.MatImpute import MatImputer
from .RFE_missforest import *
from .KNN_RFE_missforest import *
from .MIDA import Autoencoder


class BaselineImputer:
    """
    整合所有的基础基线函数
    """
    def __init__(self,
                 n_clusters: int = 3,
                 n_components: int = 5,
                 svd_max_iter: int = 100,
                 svd_tol: float = 1e-4,
                 random_state: int = 42):

        self.n_clusters = n_clusters
        self.n_components = n_components
        self.svd_max_iter = svd_max_iter
        self.svd_tol = svd_tol
        # 随机种子
        self.random_state = random_state
        np.random.seed(self.random_state)

    def impute(self,
               X: pd.DataFrame,
               y: pd.DataFrame,
               method: str,
               **kwargs) -> (pd.DataFrame, pd.DataFrame):

        # 为了不修改原始数据，先复制一份并重置索引
        X = X.copy().reset_index(drop=True)

        if method == 'lm':
            return self._low_rank_impute(X, y)
        elif method == 'KNN':
            return self._KNN_impute(X, y)
        elif method == "MatImputer":
            return self._mat_imputer(X, y)
        elif method in ['vae', 'vanilla', 'vae_miwae', 'h_vae', 'hver', 'hmc_vae', 'hh_vaem']:
            return self._vae(X, y, variant=method, **kwargs)
        elif method == 'gtmcc':
            X = X.fillna(0)
            S = rbf_kernel(X, gamma=0.5)
            np.fill_diagonal(S, 0)
            return self._gtmcc_impute(X, y, S)
        elif method in ['hyperimpute', 'missforest', 'gain', 'sinkhorn', 'miwae', 'miracle', 'ice', 'em', 'median', 'mean', 'MICE', 'softimpute']:
            return self._hyperimpute_methods(X, y, method, **kwargs)
        elif method == 'RFE_mf':
            return self._RFE_mf(X, y)
        elif method == 'MIDA':
            return self._MIDA(X, y)
        elif method == 'stacking':
            return self._stacking_impute(X, y, **kwargs)
        elif method == 'vae':
            return self._vae(X, y)
        else:
            raise ValueError(f'Unknown method: {method}')

    def _mat_imputer(self,
                    X: pd.DataFrame,
                    y: pd.DataFrame):
        Xy = pd.concat([X, y], axis=1)
        mat_imputer = MatImputer(random_state=self.random_state)
        imputed_Xy = mat_imputer.transform(Xy)
        k = y.shape[1]
        x_imputed = pd.DataFrame(imputed_Xy.iloc[:,:-k], index=X.index, columns=X.columns)
        y_imputed = pd.DataFrame(imputed_Xy.iloc[:,-k:], index=y.index, columns=y.columns)
        return x_imputed, y_imputed

    def _MIDA(self,
              x: pd.DataFrame,
              y: pd.DataFrame):
        x.columns = x.columns.map(str)
        y.columns = y.columns.map(str)
        xy = pd.concat([x, y], axis=1)
        dim = xy.shape[1]
        ae = Autoencoder(dim=dim)
        ae.fit(xy)
        xy_imputed = ae.transform(xy)
        k = y.shape[1]
        x_imputed = pd.DataFrame(xy_imputed.iloc[:,:-k], index=x.index, columns=x.columns)
        y_imputed = pd.DataFrame(xy_imputed.iloc[:,-k:], index=y.index, columns=y.columns)
        return x_imputed, y_imputed

    def _RFE_mf(self,
                   X: pd.DataFrame,
                   y: pd.DataFrame):
        X.columns = X.columns.map(str)
        y.columns = y.columns.map(str)
        xy = pd.concat([X, y], axis=1)
        if not xy.isna().any().any():
            return X.copy(), y.copy()
        mf = RFE_MissForest(
            rfe_n_estimators=100,
            rfe_cv=5,
            rfe_step=1,
            random_state=self.random_state,
            feature_selection=True,
            verbose=0
        )

        mf.fit(xy)
        xy_imputed = mf.transform(xy)
        k = y.shape[1]
        x_imputed = pd.DataFrame(xy_imputed.iloc[:,:-k], index=X.index, columns=X.columns)
        y_imputed = pd.DataFrame(xy_imputed.iloc[:,-k:], index=y.index, columns=y.columns)
        return x_imputed, y_imputed

    def _low_rank_impute(self,
                         X: pd.DataFrame,
                         y: pd.DataFrame):
        """
        对 X 和 y 分别做低秩矩阵分解填充：
        - 若矩阵只有一列，则直接用均值填充；
        - 否则，用 TruncatedSVD 做低秩分解（n_components），重构后将原来缺失的位置替换。
        """
        xy = pd.concat([X, y], axis=1)
        xy_imputed = self._low_rank_impute_single(xy)
        k = y.shape[1]
        x_imputed = pd.DataFrame(xy_imputed.iloc[:,:-k], index=X.index, columns=X.columns)
        y_imputed = pd.DataFrame(xy_imputed.iloc[:,-k:], index=y.index, columns=y.columns)

        return x_imputed, y_imputed

    def _low_rank_impute_single(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df.isna()
        if df.shape[1] == 1:
            return df.fillna(df.mean())
        filled = df.fillna(0).values
        # 调整 n_compinents 不能超过列数 -1
        n_comp = min(self.n_components, df.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp,
                           n_iter=self.svd_max_iter,
                           tol=self.svd_tol,
                           random_state=self.random_state)
        W = svd.fit_transform(filled)   # (n_samples, n_comp)
        H = svd.components_             # (n_comp, n_features)
        reconstructed = np.dot(W, H)    # 重构矩阵，形状同 filled
        # 只替换原来的缺失位置
        filled[mask.values] = reconstructed[mask.values]
        return pd.DataFrame(filled, columns=df.columns, index=df.index)

    def _vae(self,
             X: pd.DataFrame,
             y: pd.DataFrame,
             variant: str = "vae",
             **kwargs):
        from uhpc.vae import VAEImputer

        df = pd.concat([X, y], axis=1)
        imputer = VAEImputer(variant=variant, **kwargs)
        df_filled = imputer.fit_transform(df)
        X_filled = df_filled.iloc[:, :X.shape[1]].reset_index(drop=True)
        y_filled = df_filled.iloc[:, X.shape[1]:].reset_index(drop=True)
        X_imputed = pd.DataFrame(
            X_filled.values,
            columns=X.columns,
            index=X.index
        )
        y_imputed = pd.DataFrame(
            y_filled.values,
            columns=y.columns,
            index=y.index
        )
        return X_imputed, y_imputed

    def _KNN_impute(self,
                    X: pd.DataFrame,
                    y: pd.DataFrame,
                    n_neighbors: int = 5):
        """
        最基础的 KNNImputer：直接在拼接的 X、y 上 fit_transform，不做超参搜索。
        """
        X.columns = X.columns.map(str)
        y.columns = y.columns.map(str)
        Xy = pd.concat([X, y], axis=1)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        arr = imputer.fit_transform(Xy)
        k = y.shape[1]
        X_cols = Xy.columns[:-k]
        y_cols = Xy.columns[-k:]
        X_imputed = pd.DataFrame(arr[:, :-k], columns=X_cols, index=X.index)
        y_imputed = pd.DataFrame(arr[:, -k:], columns=y_cols, index=y.index)
        return X_imputed, y_imputed

    def _stacking_impute(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        n_rounds: int = 3,
        base_model_factory=None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        堆叠插补：利用其他任务标签作为输入，迭代训练回归模型填充缺失标签。
        """
        n_rounds = max(1, int(n_rounds))

        X_model = X.copy()
        X_model.columns = X_model.columns.astype(str)
        for col in X_model.columns:
            series = X_model[col]
            mean_val = series.mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            X_model[col] = series.fillna(mean_val)

        y_filled = y.copy()
        y_filled.columns = y_filled.columns.astype(str)
        y_filled.columns = y_filled.columns.astype(str)
        y_filled = y_filled.fillna(y_filled.mean())
        original_missing = y.isna()
        order = list(y_filled.columns)

        def _build_model():
            if base_model_factory is not None:
                if callable(base_model_factory):
                    return base_model_factory()
                return base_model_factory
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
            )

        for _ in range(n_rounds):
            for col in order:
                mask = ~y[col].isna()
                if mask.sum() < 5:
                    continue
                features = pd.concat([X_model, y_filled.drop(columns=col)], axis=1)
                model = _build_model()
                model.fit(features.loc[mask], y.loc[mask, col])
                preds = pd.Series(model.predict(features), index=y.index)
                missing_mask = original_missing[col]
                y_filled.loc[missing_mask, col] = preds.loc[missing_mask]

        return X.copy(), y_filled

    def _interative_impute(self, X: pd.DataFrame, y: pd.DataFrame):
        X.columns = X.columns.map(str)
        y.columns = y.columns.map(str)
        Xy_Interative = pd.concat([X, y], axis=1)
        imp_Interative = IterativeImputer(max_iter=100, random_state=self.random_state)
        imputed_Interative = imp_Interative.fit_transform(Xy_Interative)
        cols = Xy_Interative.columns.tolist()
        k = y.shape[1]
        X_cols = cols[:-k]
        y_cols = cols[-k:]
        X_imputed = pd.DataFrame(imputed_Interative[:, :-k], columns=X_cols, index=X.index)
        y_imputed = pd.DataFrame(imputed_Interative[:, -k:], columns=y_cols, index=y.index)
        return X_imputed, y_imputed

    def _gtmcc_impute(self,
                      X: pd.DataFrame,
                      y: pd.DataFrame,
                      S: np.ndarray,
                      gtmcc_k: int = 400,
                      gtmcc_beta: float = 0.8,
                      gtmcc_gam: float = 10,
                      gtmcc_lam1: float = 2,
                      gtmcc_lam2: float = 1,
                      gtmcc_nn: int = 10):
        # 构建 Z：行为样本，列为y 后接X
        Y_filled = y.fillna(0).values
        X_filled = X.fillna(0).values
        Z = np.hstack((Y_filled, X_filled))
        # 计算拉普拉斯矩阵
        L = self._get_lap(S, gtmcc_nn)
        # 找出含有 NaN 的样本行
        mask_Z = np.isnan(np.hstack((y.values, X.values)))
        test_idx = np.where(mask_Z.any(axis=1))[0]
        c = y.shape[1]
        # 执行 GT-MCC
        Z_pred = self._gtmcc_core(
            Z, L,
            k = gtmcc_k,
            test = test_idx,
            c = c,
            beta = gtmcc_beta,
            gam = gtmcc_gam,
            lam1 = gtmcc_lam1,
            lam2 = gtmcc_lam2
        )

        Y_pred = Z_pred[:, :c]
        X_pred = Z_pred[:, c:]

        return pd.DataFrame(X_pred, columns=X.columns, index=X.index), \
                pd.DataFrame(Y_pred, columns=y.columns, index=y.index)

    def _get_lap(self, S: np.ndarray, nn: int) -> np.ndarray:
        n = S.shape[0]
        # 对角归一化
        S = S.copy()
        if np.any(np.diag(S) != 0):
            np.fill_diagonal(S, 0)
        # 构造 KNN 图
        ind = np.argsort(-S, axis=1)[:, :nn]
        Sknn = np.zeros_like(S)
        for i in range(n):
            Sknn[i, ind[i]] = S[i, ind[i]]
            Sknn[ind[i], i] = S[ind[i], i]
        D = np.diag(1.0 / (np.sqrt(Sknn.sum(axis=1)) + 1e-9))
        Snorm = D @ Sknn @ D
        return np.eye(n) - Snorm

    def _gtmcc_core(self, Z: np.ndarray, L: np.ndarray,
                    k: int, test: np.ndarray, c: int,
                    beta: float, gam: float, lam1: float, lam2: float) -> np.ndarray:
        n, _ = Z.shape
        # 初始化
        U, s, Vt = svd(Z, full_matrices=False)
        W = U[:, :k].T
        H = Vt[:k, :]
        obj_old = self._obj_fun(W, H, Z, L, test, c, beta, gam, lam1, lam2)
        for _ in range(100):
            # 更新 W
            gW = self._gradW(W, H, Z, test, c, beta)
            tW = 2*beta*norm(H @ H.T)
            G = W - gW / tW
            W = tW * G @ inv(2*gam*L + 2*lam1*np.eye(n) + tW*np.eye(n))
            # 更新 H
            gH = self._gradH(W, H, Z, test, c, beta)
            tH = 2*beta*norm(W @ W.T)
            V = H - gH / tH
            H = tH * V / (tH + 2*lam2)

        return W.T @ H

    def _gradW(self, W, H, Z, test, c, beta):
        Oz = np.ones_like(Z)
        Oz[test, :c] = 0
        Pz = np.array(Z != 0, int)
        return 2*(1 - beta)*H @ ((H.T @ W -Z.T) * Oz.T) \
                + 2*(2*beta - 1)*H @ ((H.T @ W - Z.T) * Pz.T)

    def _gradH(self, W, H, Z, test, c, beta):
        Oz = np.ones_like(Z)
        Oz[test, :c] = 0
        Pz = np.array(Z != 0, int)
        return 2*(1 - beta)*W @ ((W.T @ H - Z) * Oz) \
                + 2*(2*beta - 1)*W @ ((W.T @ H - Z) * Pz)

    def _obj_fun(self, W, H, Z, L, test, c, beta, gam, lam1, lam2):
        Oz = np.ones_like(Z)
        Oz[test, :c] = 0
        Pz = np.array(Z!= 0, int)
        f1 = (1-beta)*norm(Oz*(Z - W.T@H), 'fro')**2
        f2 = (2*beta-1)*norm(Pz*(Z - W.T@H), 'fro')**2
        p = f1 + f2 + gam*np.trace(W @ L @ W.T) \
            + lam1*norm(W, 'fro')**2 + lam2*norm(H, 'fro')**2
        return p

    def _hyperimpute_methods(self,X, y, method, **kwargs):
        df = pd.concat([X, y], axis=1)

        if method == 'vae':
            from uhpc.vae import VAEImputer
            plugin = VAEImputer(**kwargs)
        else:
            if Imputers is None:
                raise ImportError(f"hyperimpute is required for method '{method}' but is not installed.")
            plugin_name = {
                'hyperimpute': 'hyperimpute',
                'missforest': 'missforest',
                'gain': 'gain',
                'sinkhorn': 'sinkhorn',
                'miwae': 'miwae',
                'miracle': 'miracle',
                'MICE': 'mice',
                'ice': 'ice',
                'em': 'EM',
                'softimpute': 'softimpute',
                'median': 'median',
                'mean': 'mean',
            }.get(method)
            
            if plugin_name is None:
                raise ValueError(f'Unknown method: {method}')
            plugin = Imputers().get(plugin_name, **kwargs)

        df_filled = plugin.fit_transform(df)

        X_filled = df_filled.iloc[:, :X.shape[1]].reset_index(drop=True)
        y_filled = df_filled.iloc[:, X.shape[1]:].reset_index(drop=True)
        X_imputed = pd.DataFrame(
            X_filled.values,
            columns=X.columns,
            index=X.index
        )
        y_imputed = pd.DataFrame(
            y_filled.values,
            columns=y.columns,
            index=y.index
        )
        return X_imputed, y_imputed



from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
class Subset:
    """
    通过划分子集来弥补数据中的缺失状况
    """
    def __init__(self, model, X_train, y_train, X_test, y_test,f_num, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.f_num = f_num
        self.model = model
        self.random_state = random_state

    def masks(self, variables=0):
        if variables == 0:
            fix_num = self.X_train.shape[1] - 3
            num = self.f_num - fix_num
            masks = []
            prefix = np.ones(fix_num, dtype=bool)
            for k in range(1, num + 1):
                for feature_subset in combinations(range(num), k):
                    mask = np.zeros(num, dtype=bool)
                    mask[list(feature_subset)] = True
                    masks.append(np.concatenate([prefix, mask]))
            return masks

        elif variables == 1:
            num = self.y_train.shape[1]
            masks = []
            fix_num = 0
            prefix = np.ones(fix_num, dtype=bool)
            for k in range(1, num + 1):
                for feature_subset in combinations(range(num), k):
                    mask = np.zeros(num, dtype=bool)
                    mask[list(feature_subset)] = True
                    masks.append(np.concatenate([prefix, mask]))
            return masks
        
    def sub(self):
        feature_masks = self.masks(variables=0)
        label_masks = self.masks(variables=1)
        # 只保留单一的label
        label_masks = [m for m in label_masks if m.sum() == 1]

        models = {}
        feature_subset_mapping = {}
        label_subset_mapping = {}
        predictions = []
        for i, f_mask in enumerate(feature_masks):
            for j, l_mask in enumerate(label_masks):
                X_train_subset = self.X_train.iloc[:, f_mask]
                y_train_subset = self.y_train.iloc[:, l_mask]
                # 通过 drop 的方法保证子集的数据的完整
                X = X_train_subset.copy().reset_index(drop=True)
                y = y_train_subset.copy().reset_index(drop=True)
                if isinstance(X, pd.Series):
                    X_missing = X.isna()
                else:
                    X_missing = X.isna().any(axis=1)
                if isinstance(y, pd.Series):
                    y_missing = y.isna()
                else:
                    y_missing = y.isna().any(axis=1)
                miss_idx = X[X_missing].index.union(y[y_missing].index)
                X_clean = X.drop(index=miss_idx)
                y_clean = y.drop(index=miss_idx)

                if X_clean.shape[1] == 0 or y_clean.shape[1] == 0:
                    continue
                model = self.model.XGBRegressor(random_state=self.random_state)
                model.fit(X_clean, y_clean.values.squeeze())
                model_name = f'M_F_{i + 1}_L_{j + 1}'
                models[model_name] = model
                feature_subset_mapping[model_name] = f_mask
                label_subset_mapping[model_name] = l_mask

            for model_name, model in models.items():

                f_mask = feature_subset_mapping[model_name]
                l_mask = label_subset_mapping[model_name]

                X_test_subset = self.X_test.iloc[:, f_mask]
                y_pred = model.predict(X_test_subset)

                for idx, label_idx in enumerate(np.where(l_mask)[0]):
                    if y_pred.ndim > 1:
                        for col_idx in range(y_pred.shape[1]):
                            predictions.append({
                                'model_name': f'{model_name}_output_{col_idx}',
                                'prediction': y_pred[:, col_idx]
                            })
                    else:
                        predictions.append({
                            'model_name': model_name,
                            'prediction': y_pred
                        })
        return predictions

    def _get_output(self, name,X_miss_pro,y_miss_pro,seed):
        predictions = self.sub()
        df_dict = {}
        for r in predictions:
            model_name = r['model_name']
            df_dict[model_name] = r['prediction']
        df_sub = pd.DataFrame(df_dict)
        # 子集结果处理
        results = []

        label1_array = []
        label2_array = []
        label3_array = []
        label4_array = []
        for i in range(1, 2 ** 3):
            label1 = [f'M_F_{i}_L_1']
            label2 = [f'M_F_{i}_L_2']
            label3 = [f'M_F_{i}_L_3']
            label4 = [f'M_F_{i}_L_4']
            label1_data = df_sub[label1]
            label2_data = df_sub[label2]
            label3_data = df_sub[label3]
            label4_data = df_sub[label4]
            label1_array.append(label1_data)
            label2_array.append(label2_data)
            label3_array.append(label3_data)
            label4_array.append(label4_data)
        label1_array = np.array(label1_array)
        label2_array = np.array(label2_array)
        label3_array = np.array(label3_array)
        label4_array = np.array(label4_array)
        arr_squeezed1 = np.squeeze(label1_array, axis=2)
        arr_squeezed2 = np.squeeze(label2_array, axis=2)
        arr_squeezed3 = np.squeeze(label3_array, axis=2)
        arr_squeezed4 = np.squeeze(label4_array, axis=2)
        mean_label1 = np.median(arr_squeezed1, axis=0)
        mean_label2 = np.median(arr_squeezed2, axis=0)
        mean_label3 = np.median(arr_squeezed3, axis=0)
        mean_label4 = np.median(arr_squeezed4, axis=0)
        # 先将 (9,) 转换为 (9,1)
        mean_label1 = mean_label1.reshape(-1, 1)  # (9,1)
        mean_label2 = mean_label2.reshape(-1, 1)  # (9,1)
        mean_label3 = mean_label3.reshape(-1, 1)
        mean_label4 = mean_label4.reshape(-1, 1)
        pred = [mean_label1, mean_label2, mean_label3, mean_label4]

        for i, col in enumerate(self.y_train.columns):
            r2 = r2_score(self.y_test[col], pred[i])
            mse = mean_squared_error(self.y_test[col], pred[i])
            mae = mean_absolute_error(self.y_test[col], pred[i])
            results.append({
                'method': 'Subset',
                'seed': seed,
                'X_miss_pro': X_miss_pro,
                'y_miss_pro': y_miss_pro,
                'label': col,
                'r2': r2,
                'mae': mae,
                'mse': mse
            })
        
        return results
