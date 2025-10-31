from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.experimental import enable_iterative_imputer   # 为了能够正常使用 InterativeImputer
from sklearn.impute import IterativeImputer
from scipy.linalg import svd, norm, inv
from hyperimpute.plugins.imputers import Imputers
from sklearn.metrics import r2_score
from RFE_missforest import *
from KNN_RFE_missforest import *
from MIDA import Autoencoder

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
        """
        :param n_clusters:      聚类的个数
        :param n_components:
        :param svd_max_iter:
        :param svd_tol:
        :param random_state:
        """
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
               method: str = 'drop',
               miss_col=None,
               **kwargs) -> (pd.DataFrame, pd.DataFrame):

        # 为了不修改原始数据，先复制一份并重置索引
        X = X.copy().reset_index(drop=True)
        if method == 'lm':
            return self._low_rank_impute(X, y)
        elif method == 'KNN':
            return self._KNN_impute(X, y)
        elif method == 'InterativeImputer':
            return self._interative_impute(X, y)
        elif method == 'gtmcc':
            X = X.fillna(0)
            S = rbf_kernel(X, gamma=0.5)
            np.fill_diagonal(S, 0)
            return self._gtmcc_impute(X, y, S)
        elif method in ['hyperimpute', 'missforest', 'gain', 'sinkhorn', 'miwae']:
            return self._hyperimpute_methods(X, y, method, **kwargs)
        elif method == 'RFE_mf':
            return self._RFE_mf(X, y)
        elif method == 'MIDA':
            return self._MIDA(X, y)
        elif method == 'softimpute':
            pass
        elif method == 'MICE':
            pass
        else:
            raise ValueError(f'Unknown method: {method}')

    def _MIDA(self,
              X: pd.DataFrame,
              y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        dim = X.shape[1]
        ae = Autoencoder(dim=dim)
        ae.fit(X)
        X_imputed = ae.transform(X)
        return X_imputed, y

    def _RFE_mf(self,
                   X: pd.DataFrame,
                   y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        mf = RFE_MissForest(
            rfe_n_estimators=100,
            rfe_cv=5,
            rfe_step=1,
            random_state=self.random_state,
            feature_selection=True
        )
        mf.fit(X)
        return mf.transform(X), y

    def _low_rank_impute(self,
                         X: pd.DataFrame,
                         y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        对 X 和 y 分别做低秩矩阵分解填充：
        - 若矩阵只有一列，则直接用均值填充；
        - 否则，用 TruncatedSVD 做低秩分解（n_components），重构后将原来缺失的位置替换。
        """
        X_clean = self._low_rank_impute_single(X)
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            y_clean = self._low_rank_impute_single(y)
        else:
            y_clean = y.fillna(y.mean())
        return X_clean, y_clean

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

    def _KNN_impute(self,
                    X: pd.DataFrame,
                    y: pd.DataFrame,
                    cv_splits: int = 3,
                    neighbors_range: range = range(1,21)
                    ) -> (pd.DataFrame, pd.DataFrame):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        best_nbr, best_r2 = None, -np.inf

        for n_neighbors in neighbors_range:
            fold_r2 = []
            for train_idx, val_idx in kf.split(X):
                X_tr = X.iloc[train_idx]
                y_tr = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                imputer = KNNImputer(n_neighbors=n_neighbors)
                imputer.fit(pd.concat([X_tr, y_tr], axis=1))

                X_tr_imp, y_tr_imp = self._KNN_impute_xy(imputer, X_tr, y_tr)
                X_val_imp, _ = self._KNN_impute_xy(imputer, X_val, y_val)

                for col in y_tr.columns:
                    mask = ~y_val[col].isna()
                    if mask.sum() == 0:
                        continue
                    model = XGBRegressor(random_state=self.random_state)

                    model.fit(X_tr_imp, y_tr_imp[col])
                    y_pred = model.predict(X_val_imp)
                    #print(y_val[col][mask].shape)
                    fold_r2.append(r2_score(y_val[col][mask], y_pred[mask]))
            if fold_r2:
                mean_r2 = np.mean(fold_r2)
                if mean_r2 > best_r2:
                    best_r2, best_nbr = mean_r2, n_neighbors
        #print(best_nbr)
        Xy_KNN = pd.concat([X, y], axis=1)
        imp_KNN = KNNImputer(n_neighbors=best_nbr)
        imputed_KNN = imp_KNN.fit_transform(Xy_KNN)
        cols = Xy_KNN.columns.tolist()
        k = y.shape[1]
        X_cols = cols[:-k]
        y_cols = cols[-k:]
        X_imputed = pd.DataFrame(imputed_KNN[:, :-k], columns=X_cols, index=X.index)
        y_imputed = pd.DataFrame(imputed_KNN[:, -k:], columns=y_cols, index=y.index)

        return X_imputed, y_imputed

    def _KNN_impute_xy(self, imputer, X: pd.DataFrame, y: pd.DataFrame):
        """
        用同一个 imputer 对 X 和 y 合并后的数组做 transform，并拆回 X, y
        """
        Xy = pd.concat([X, y], axis=1)
        arr = imputer.transform(Xy)
        k = y.shape[1]
        X_imp = pd.DataFrame(arr[:, :-k], columns=X.columns, index=X.index)
        y_imp = pd.DataFrame(arr[:, -k:], columns=y.columns, index=y.index)
        return X_imp, y_imp

    def _interative_impute(self, X: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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
        """
        支持 hyperimpute 插补方法
        :param X:
        :param y:
        :param method:
        :param kwargs:
        :return:
        """

        df = pd.concat([X, y], axis=1)
        if method in ['hyperimpute']:
            plugin = Imputers().get('hyperimpute', **kwargs)
        elif method == 'missforest':
            plugin = Imputers().get('missforest', **kwargs)
        elif method == 'gain':
            plugin = Imputers().get('gain', **kwargs)
        elif method == 'sinkhorn':
            plugin = Imputers().get('sinkhorn', **kwargs)
        elif method == 'miwae':
            plugin = Imputers().get('miwae', **kwargs)
        else:
            raise ValueError(f'Unknown method: {method}')
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