import numpy as np
import torch
import gpytorch
import os
import pandas as pd

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.array(x), dtype=torch.float32)

def _prepare_train_xy(train_x, train_y):
    """
    返回: X_t (n, d), Y_t (n, T), col_names(list[str])
    过滤掉在任一任务上为 NaN 的样本
    """
    # 转 DataFrame 方便对齐/去 NaN
    Xdf = pd.DataFrame(train_x)
    Ydf = pd.DataFrame(train_y)
    mask = ~Ydf.isna().any(axis=1)
    Xdf = Xdf.loc[mask]
    Ydf = Ydf.loc[mask]
    col_names = list(Ydf.columns)

    X_t = _to_tensor(Xdf.values)            # (n, d)
    Y_t = _to_tensor(Ydf.values)            # (n, T)
    return X_t, Y_t, col_names

def _prepare_test_x(test_x):
    Xdf = pd.DataFrame(test_x)
    return _to_tensor(Xdf.values), Xdf.index

class MultitaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks:int):
        super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def model_train(train_x, train_y, lr=0.05, iters=10000):
    X_t, Y_t, col_names = _prepare_train_xy(train_x, train_y)
    num_tasks = Y_t.shape[1]
    if num_tasks < 1:
        raise ValueError("train_y 至少需要 1 列目标。")
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGP(X_t, Y_t, likelihood, num_tasks=num_tasks)
    smoke_test = ('CI' in os.environ)
    train_iterations = 2 if smoke_test else iters
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(train_iterations):
        optimizer.zero_grad()
        output = model(X_t)
        loss = -mll(output, Y_t)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f' % (i, train_iterations, loss.item()))
        optimizer.step()

    return likelihood, model, col_names

def model_fit_predict(train_x, train_y, test_x):
    likelihood, model, col_names = model_train(train_x, train_y)

    model.eval()
    likelihood.eval()

    Xte_t, te_index = _prepare_test_x(test_x)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(Xte_t))
        mean = preds.mean                      # (m, T)
        Y_pred = mean.detach().cpu().numpy()

    pred_df = pd.DataFrame(Y_pred, index=te_index, columns=col_names)
    return pred_df

