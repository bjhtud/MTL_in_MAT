


class Model_Inter():
    '''
    *LightGBM：use_missing    参数可以处理缺失值。
    *HistGradientBoostingRegressor：可以直接处理缺失值。
    *HMlasso
    *梯度提升决策树（GBDT）框架
    *CatBoost(MultiRMSE / MultiRMSEWithMissingValues) 显式建模标签 / 任务间相似性
    *XGBoost：可以直接处理缺失值。
    *MT - ExtraTrees(Tree - Based Ensemble Multi - Task Learning Method for Classification and Regression)
    '''

class Iterative_Inter():
    '''
    * 堆叠(Multi-target regression via input space expansion: treating targets as inputs)
    * 回归链(同上，还有：leveraging multi-task learning regressor chains for small and sparse tabular data in materials design)

    '''


class GP_Reg():
    '''
    * Multitask GP Regression (缺失特征：用 Uncertain Inputs——对缺失列设高斯分布 μ±σ，内核对该维做解析积分。GPyTorch里有完整例子。)
    '''

class SubSet():
