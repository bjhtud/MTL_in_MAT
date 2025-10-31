import numpy as np
import torch
import gpytorch

def mtgp_fit_predict(train_x,
              train_y,
              test_x,
              *,
              iters: int = 50,
              lr: float = 0.1,
              rank: int = 1,
              ard: bool = False,
              seed: int | None = None,
              device: str | None = None):
    """
    训练 Multi-Task Exact GP 并在 test_x 上预测，返回 pred_y（均值）。
    - train_x: (n, d)
    - train_y: (n,) 或 (n,1) 或 (n,T)
    - test_x : (m, d)

    返回:
      pred_y: 若 T==1 则形状 (m,)，否则 (m, T)
    """
    # --------- 准备数据 ---------
    Xtr = np.asarray(train_x, dtype=np.float32)
    Xte = np.asarray(test_x,  dtype=np.float32)
    Ytr = np.asarray(train_y, dtype=np.float32)

    if Xtr.ndim == 1: Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1: Xte = Xte.reshape(-1, 1)
    if Ytr.ndim == 1: Ytr = Ytr.reshape(-1, 1)  # 单任务 → 一列

    if np.isnan(Xtr).any() or np.isnan(Xte).any() or np.isnan(Ytr).any():
        raise ValueError("ExactGP 不能直接处理 NaN/Inf，请先做插补。")

    n, d = Xtr.shape
    if Ytr.shape[0] != n:
        raise ValueError("train_x 与 train_y 行数不一致。")

    T = Ytr.shape[1]  # 任务数

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr_t = torch.as_tensor(Xtr, device=device)
    Ytr_t = torch.as_tensor(Ytr, device=device)
    Xte_t = torch.as_tensor(Xte, device=device)

    # --------- 定义模型 ---------
    class _MTGPR(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, num_tasks, ard_num_dims=None, rank=1):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                base_kernel, num_tasks=num_tasks, rank=rank
            )
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T).to(device)
    ard_num_dims = d if ard else None
    model = _MTGPR(Xtr_t, Ytr_t, likelihood, num_tasks=T, ard_num_dims=ard_num_dims, rank=rank).to(device)

    # --------- 训练 ---------
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(int(iters)):
        optimizer.zero_grad()
        output = model(Xtr_t)
        loss = -mll(output, Ytr_t)
        loss.backward()
        optimizer.step()

    # --------- 预测（只要均值）---------
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(Xte_t))
        mean = pred.mean  # (m, T)

    y_pred = mean.detach().cpu().numpy()
    # 单任务时返回一维
    if T == 1:
        y_pred = y_pred.reshape(-1)

    return y_pred
