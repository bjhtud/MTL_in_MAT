# 多任务学习
关注重点是多任务学习，所以单纯插值的方式可能不太合适。
起点为 UHPC 数据集。 UHPC中有四个任务，分别是：Flowability, porosity, compressive strength, and tensile strength.
在针对每个任务时，使用其他任务的信息作为辅助，这涉及到多任务学习的概念。既然是多任务学习，那么就需要考虑如何利用标签之间的关系。

所以在UHPC数据集上，测试集就需要包含所有任务的标签。这样可以确保在测试时，模型能够利用所有任务的信息。
依旧采取旧的训练策略，合并数据集之后划分出所有没有缺失的样本作为测试集，其他样本作为训练集。

那么要测试的方法就要满足：可以处理缺失特征和标签的数据集，并且一定要结合多个标签的关系。

## 方法总结：
### 转为单任务学习 作为基准 

其实就是**针对每个任务**进行删除，包括两种删除方式

1. 样本删除法：直接删除有缺失的样本
2. 特征删除法：不删除行，直接删除有缺失的特征，找到最大的特征子集

### 插补方法+多任务学习模型

之前总结的插补方法，插补出最终的数据集之后，使用多任务学习模型进行训练。

- 插补方法：

| 名字                             | 引用量 | 来源                                                                                                                     |
|--------------------------------|-------|------------------------------------------------------------------------------------------------------------------------|
| MissForest                     |   | (核心文章)                                                                                                                 |
| RFE-MissForest                 | 7 | A novel MissForest-based missing values imputation approach with recursive feature elimination in medical applications |
| Hyperimpute                    |  | (核心文章)                                                                                                                 |
| KNN based method (MatImputer)  |   | (核心文章)                                                                                                                 |
| GAIN                           |   | (核心文章)                                                                                                                 |
| Sinkhorn                       |   | (核心文章)                                                                                                                 |
| MICE (Iterativeimputer)        | 19525 | mice: Multivariate Imputation by Chained Equations in R                                                                |
| KNN插补                          |   |                                                                                                                        |
| MIDA(去噪自编码器)                   | 429  | MIDA: Multiple Imputation using Denoising Autoencoders                                                                 |
| MIWAE(变分自编码器)                  | 358  | MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets                                                |
| SoftImpute (低秩矩阵改良)(还有其他方法在包中) | 686   | Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares                                                  |
| 低秩矩阵                           |   |                                                                                                                        |
| 堆叠                             | 422  | Multi-target regression via input space expansion: treating targets as inputs                                          |
| VAE                            |   | 还没找到                                                                                                                   |

- 多任务学习模型(其实就是多元回归)：
  - 多输出随机森林
  - Ridge Regression 基于正则化
  - Lasso Regression(MT Lasso) 基于正则化
  - Elastic Net(MT Elastic Net) 基于正则化
  - 多输出GP 
  - 多输出SVR (跟GP差不多)
  - autosklearn 支持多元回归

### 传统机器学习方法(直接能处理缺失特征和标签的模型) 

梯度提升决策树（GBDT）框架
* CatBoost (MultiRMSE / MultiRMSEWithMissingValues) 显式建模标签/任务间相似性
* XGBoost：可以直接处理缺失值。
* MT-ExtraTrees (Tree-Based Ensemble Multi-Task Learning Method for Classification and Regression)

基于贝叶斯推理(不一定好使，但可以试试)
* Multitask GP Regression (缺失特征：用 Uncertain Inputs——对缺失列设高斯分布 μ±σ，内核对该维做解析积分。GPyTorch里有完整例子。)

基于集成学习
* 堆叠(Multi-target regression via input space expansion: treating targets as inputs)
* 回归链(同上，还有：leveraging multi-task learning regressor chains for small and sparse tabular data in materials design)
* subset

### 深度学习方法

在不完整数据集上进行训练的深度学习方法：
* TabNet
* TabTransformer
* MMOE(实现也挺复杂)

~~- 端到端的 插值+多元回归预测 (但是是同时进行训练的) (不存在，不测试了)~~
*   ~~- MIWAE~~
*   ~~- VAEAC (Variational Autoencoder with Arbitrary Conditioning)~~
*   ~~- MTGAIN (End-to-end Multi-task Learning of Missing Value Imputation and Forecasting in Time-Series Data)~~
*   ~~- A multi-task learning-based generative adversarial network for red tide multivariate time series imputation~~


### 只能处理特征缺失的模型：(仅仅记录，以防后边脑子一热又要做)
* LightGBM：use_missing 参数可以处理缺失值。
* HistGradientBoostingRegressor：可以直接处理缺失值。
* HMlasso