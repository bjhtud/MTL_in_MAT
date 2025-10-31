'''
特征缺失而标签完整
'''
import numpy as np
import pandas as pd

def center_columns_with_nans(X):
    # 强制转为 ndarray，再做 NumPy 索引
    X = np.asarray(X, dtype=float).copy()
    col_means = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_means, idx[1])  # 临时填平用于中心化
    X -= col_means
    X[idx] = np.nan                     # 还原缺失
    return X, col_means

def pairwise_cov_and_rho(X, y):
    """
    X: (n,p) with np.nan for missing, assumed column-centered on observed entries
    y: (n,) centered
    Returns: Spair (p,p), rho_pair (p,), R (p,p)
    """
    n, p = X.shape
    mask = ~np.isnan(X)
    X0 = np.where(mask, X, 0.0)
    # n_jk: number of samples jointly observed for (j,k)
    Nj = mask.astype(float).T @ mask.astype(float)  # (p,p)
    # Sum_{i in I_jk} X_ij X_ik
    Ssum = X0.T @ X0  # (p,p)
    # Spair
    with np.errstate(invalid='ignore', divide='ignore'):
        Spair = np.where(Nj > 0, Ssum / Nj, 0.0)

    # rho_pair_j = (1 / n_jj) sum_{i in I_jj} X_ij y_i
    yy = y.reshape(-1, 1)
    # 只在 X 观测处累计
    rsum = (X0 * mask) .T @ y
    n_jj = mask.sum(axis=0)  # (p,)
    rho_pair = np.where(n_jj > 0, rsum / n_jj, 0.0)

    R = Nj / float(n)  # observed ratio matrix
    return Spair, rho_pair, R

def proj_psd(M):
    # 对称化+特征值截断
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T

def admm_weighted_frob(Spair, W, mu=1.0, max_iter=500, tol=1e-6, verbose=False):
    """
    Solve: minimize || W ⊙ (Σ - Spair) ||_F^2  s.t. Σ ⪰ 0
    ADMM with variables A (PSD), B = Σ - Spair; A = B + Spair
    Updates (from paper Appendix / Alg.1):
      A^{k+1} ← Proj_PSD(B^k + Spair + μ Λ^k)
      B^{k+1} ← (A^{k+1} - Spair - μ Λ^k) ⊙ ( μ W⊙W + I )^{-1}
      Λ^{k+1} ← Λ^k - (A^{k+1} - B^{k+1} - Spair)/μ
    """
    p = Spair.shape[0]
    A = proj_psd(Spair)
    B = np.zeros_like(Spair)
    Lam = np.zeros_like(Spair)
    denom = (mu * (W * W) + 1.0)  # elementwise
    for it in range(max_iter):
        A_prev = A
        # A-update
        A = proj_psd(B + Spair + mu * Lam)
        # B-update (elementwise)
        B = (A - Spair - mu * Lam) / denom
        # Λ-update
        Lam = Lam - (A - B - Spair) / mu
        # Convergence check
        r_norm = np.linalg.norm(A - B - Spair, 'fro')
        s_norm = np.linalg.norm(A - A_prev, 'fro')
        if verbose and it % 50 == 0:
            print(f"[ADMM] iter {it}, r={r_norm:.3e}, s={s_norm:.3e}")
        if r_norm < tol and s_norm < tol:
            break
    Σ_tilde = A
    return Σ_tilde

def soft_threshold(z, lam):
    if z > lam:  return z - lam
    if z < -lam: return z + lam
    return 0.0

def lasso_cov_coordinate_descent(Sigma, rho, lam, max_iter=1000, tol=1e-6, warm_start=None):
    """
    Solve: min 0.5 β^T Σ β - ρ^T β + λ ||β||_1   (Σ is PSD)
    Coord-descent with closed-form soft-thresholding updates.
    """
    p = Sigma.shape[0]
    beta = np.zeros(p) if warm_start is None else warm_start.copy()
    # Diagonal safeguard
    diag = np.clip(np.diag(Sigma), 1e-12, None)
    Sigma = Sigma.copy()
    np.fill_diagonal(Sigma, diag)
    for it in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # z = ρ_j - Σ_{j,-j} β_{-j}
            z = rho[j] - (Sigma[j, :] @ beta - Sigma[j, j] * beta[j])
            beta[j] = soft_threshold(z, lam) / Sigma[j, j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

class HMLasso:
    def __init__(self, alpha=1.0, mu=1.0, admm_max_iter=500, admm_tol=1e-6, cd_max_iter=2000, cd_tol=1e-6):
        self.alpha = alpha
        self.mu = mu
        self.admm_max_iter = admm_max_iter
        self.admm_tol = admm_tol
        self.cd_max_iter = cd_max_iter
        self.cd_tol = cd_tol
        # learned stuff
        self.coef_ = None
        self.Sigma_ = None
        self.rho_ = None
        self.col_means_ = None   # 训练集各列均值（只用观测值算）
        self.y_mean_ = None      # 训练集 y 的均值
        self.intercept_ = None   # 等价拦截项（可选）
        self._fitted = False

    def fit(self, X, y, lam):
        # —— 这里也统一转 ndarray，并确保 y 一维、无 NaN
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        # 过滤掉 y 缺失的样本
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # 1) 居中
        Xc, _ = center_columns_with_nans(X)
        yc = y - float(np.nanmean(y))
        # 2) 后续与你原来的逻辑一致...
        Spair, rho_pair, R = pairwise_cov_and_rho(Xc, yc)
        W = np.power(R, self.alpha)
        Sigma_tilde = admm_weighted_frob(Spair, W, mu=self.mu,
                                         max_iter=self.admm_max_iter, tol=self.admm_tol)
        beta = lasso_cov_coordinate_descent(Sigma_tilde, rho_pair, lam,
                                            max_iter=self.cd_max_iter, tol=self.cd_tol)
        self.coef_ = beta
        self.Sigma_ = Sigma_tilde
        self.rho_ = rho_pair
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # 简单线性预测：X @ beta（如果你有截距，记得补）
        return X @ self.coef_

def hmlasso_train(X_train, y_train,task_name, lam=0.1, **kwargs):
    y_train = y_train[task_name]
    model = HMLasso(**kwargs).fit(X_train, y_train, lam)
    return model

def hmlasso_fit_predict(train_x, train_y, test_x, task_name):
    model = hmlasso_train(train_x, train_y, task_name, lam=0.1)
    pred_y = model.predict(test_x)
    return pred_y