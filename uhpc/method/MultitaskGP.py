import numpy as np
import torch
import gpytorch
import pandas as pd
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.mlls import SumMarginalLogLikelihood

# --------- utils ---------
def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.array(x), dtype=torch.float32)

def _prepare_train(train_x, train_y):
    """不丢行，仅做类型整理。返回 DataFrame 便于逐列掩码。"""
    Xdf = pd.DataFrame(train_x)
    Ydf = pd.DataFrame(train_y)
    col_names = list(Ydf.columns)
    X_t = _to_tensor(Xdf.values)         # (N, D)
    Y_t = _to_tensor(Ydf.values)         # (N, T) 允许 NaN
    return X_t, Y_t, col_names, Xdf.index

def _prepare_test_x(test_x):
    Xdf = pd.DataFrame(test_x)
    return _to_tensor(Xdf.values), Xdf.index

# --------- 单任务 GP 模型 ---------
class SingleTaskGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --------- 训练（Plan B） ---------
def model_train(train_x, train_y, lr=0.05, iters=1000):
    X_t, Y_t, col_names, _ = _prepare_train(train_x, train_y)
    N, T = Y_t.shape
    models, likes = [], []

    # 逐任务建模（仅用该任务有标签的行）
    for j in range(T):
        yj = Y_t[:, j]
        mask = torch.isfinite(yj)
        if mask.sum().item() == 0:
            raise ValueError(f"任务 {j}（列名: {col_names[j]}）没有任何观测标签，无法训练。")

        Xj = X_t[mask]
        yj_obs = yj[mask]

        lik = GaussianLikelihood()
        mdl = SingleTaskGP(Xj, yj_obs, lik)   # 这里 ExactGP 会把 (Xj, yj_obs) 存进 mdl.train_inputs / train_targets
        models.append(mdl)
        likes.append(lik)

    model_list = IndependentModelList(*models)
    like_list  = LikelihoodList(*likes)
    mll = SumMarginalLogLikelihood(like_list, model_list)

    model_list.train(); like_list.train()
    opt = torch.optim.Adam(model_list.parameters(), lr=lr)

    # ！！！在构造好 model_list 之后，再从“子模型内部”取训练输入/标签
    X_list = [m.train_inputs for m in model_list.models]     # 注意：这是一个“tuple”的列表
    y_list = [m.train_targets for m in model_list.models]

    # 可选：做硬校验，确保就是同一块内存/同一设备/同一 dtype
    for k, mdl in enumerate(model_list.models):
        xin = X_list[k][0]
        assert xin.data_ptr() == mdl.train_inputs[0].data_ptr()
        assert xin.device == mdl.train_inputs[0].device
        assert xin.dtype   == mdl.train_inputs[0].dtype

    for i in range(iters):
        opt.zero_grad()

        # 方案一（最稳）：逐子模型直接调用它自己的 train_inputs
        outputs = [mdl(*mdl.train_inputs) for mdl in model_list.models]

        # 方案二（等价，也可用）：把每个子模型的 tuple 原样传给 IndependentModelList
        # outputs = model_list(*(mdl.train_inputs for mdl in model_list.models))

        loss = -mll(outputs, y_list)
        loss.backward()
        opt.step()

    return like_list, model_list, col_names


# --------- 拟合并预测 ---------
def model_fit_predict(train_x, train_y, test_x, lr=0.05, iters=1000):
    like_list, model_list, col_names = model_train(train_x, train_y, lr=lr, iters=iters)

    Xte_t, te_index = _prepare_test_x(test_x)
    model_list.eval(); like_list.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # 对同一 X* 复用，按任务列表传入
        preds = model_list(*([Xte_t] * len(col_names)))
        means = [p.mean for p in preds]                 # list of (M,)
        Y_pred = torch.stack(means, dim=1).cpu().numpy()  # (M, T)

    pred_df = pd.DataFrame(Y_pred, index=te_index, columns=col_names)
    return pred_df
