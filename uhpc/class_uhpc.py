import pandas as pd
import os
from sklearn.model_selection import train_test_split

def clean_duplicates(df: pd.DataFrame, label_col_list: list) -> pd.DataFrame:
    """
    清理重复行，保留每组重复行的第一行，并计算其他行的平均值。

    :param df: 输入的DataFrame
    :param label_col_list: 用于分组的列名列表
    :return: 清理后的DataFrame
    """

    agg_dict = {col: 'mean' for col in label_col_list}
    cols_list = list(df.columns.difference(label_col_list))
    df_merged = (
        df
        .groupby(cols_list, dropna=False, sort=False)  # 若有缺失值也能一起分组
        .agg(agg_dict)
        .reset_index()
    )

    return df_merged

def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    合并两个DataFrame，保留所有样本。

    :param df1: 第一个DataFrame
    :param df2: 第二个DataFrame
    :param on: 用于合并的列名列表
    :param how: 合并方式，默认为'outer'
    :return: 合并后的DataFrame
    """
    cols_list = list(df1.columns.intersection(df2.columns))
    df_final = pd.merge(
        df1,
        df2,
        on=cols_list,
        how='outer',          # outer → 保留所有样本
    )
    return df_final


class DataSet_uhpc:
    """
    DataSet 类：
    1. 从给定路径加载数据（支持 .xlsx, .csv）
    2. 拆分特征和标签
    3. 返回测试集和训练集
    """
    def __init__(self, path, num_feature, test_size=0.2, seed=42, task_name=None):
        """
        :param path:        数据文件路径
        :param num_feature: 特征的数量
        :param test_size:   测试集的比例
        :param seed:        随机种子
        """
        # 合并path 和 UHPC_original.xlsx
        self.folder_path = path
        self.file_path = os.path.join(path, 'UHPC_original.xlsx')
        self.task_name = task_name
        self.tasks = {
            'Flowability': 21,
            'Porosity': 22,
            'Compressive strength': 24,
            'Flexural strength': 24
        }
        if self.task_name not in list(self.tasks.keys()):
            raise ValueError(f"Task name must be provided. Available tasks: {list(self.tasks.keys())}")
        self.num_feature = None
        self.test_size = test_size
        self.seed = seed



        # 从文件中读入 pandas.DataFrame
        self._load_raw()

        # 拆分训练集和测试集
        self._split_train_test()

        # 拆分特征和标签
        self._split_feature_label()

    def _load_raw(self):
        """
        根据扩展名自动选择读取方式
        :return: 读取到的数据集
        """

        self.df_f = pd.read_excel(self.file_path, sheet_name=list(self.tasks.keys())[0]).astype(float)
        self.df_p = pd.read_excel(self.file_path, sheet_name=list(self.tasks.keys())[1]).astype(float)
        self.df_cs = pd.read_excel(self.file_path, sheet_name=list(self.tasks.keys())[2]).astype(float)
        self.df_fs = pd.read_excel(self.file_path, sheet_name=list(self.tasks.keys())[3]).astype(float)

        self.df_f_cleaned = clean_duplicates(self.df_f, [list(self.tasks.keys())[0]])
        self.df_p_cleaned = clean_duplicates(self.df_p, [list(self.tasks.keys())[1]])
        self.df_cs_cleaned = clean_duplicates(self.df_cs, [list(self.tasks.keys())[2]])
        self.df_fs_cleaned = clean_duplicates(self.df_fs, [list(self.tasks.keys())[3]])

        self.columns = list(self.df_cs_cleaned.columns.intersection(self.df_fs_cleaned.columns)) + list(self.tasks.keys())

    def _split_train_test(self):
        """
        获取训练集的特征和标签
        :param task_name: 任务名称
        :return: 训练集特征和标签
        """

        self.num_feature = self.tasks[self.task_name]

        if self.task_name == 'Flowability':
            self.df_f_cleaned, self.test_df = train_test_split(self.df_f_cleaned, test_size=self.test_size,
                                                     random_state=self.seed)
        elif self.task_name == 'Porosity':
            self.df_p_cleaned, self.test_df = train_test_split(self.df_p_cleaned, test_size=self.test_size,
                                                     random_state=self.seed)
        elif self.task_name == 'Compressive strength':
            self.df_cs_cleaned, self.test_df = train_test_split(self.df_cs_cleaned, test_size=self.test_size,
                                                       random_state=self.seed)
        elif self.task_name == 'Flexural strength':
            self.df_fs_cleaned, self.test_df = train_test_split(self.df_fs_cleaned, test_size=self.test_size,
                                                       random_state=self.seed)

        self.train_df = combine_dataframes(combine_dataframes(combine_dataframes(self.df_f_cleaned, self.df_p_cleaned),
                                                         self.df_cs_cleaned), self.df_fs_cleaned)

        self.train_df = self.train_df[self.columns]

    def _split_feature_label(self):

        self.X_train = self.train_df.iloc[:, :self.num_feature].reset_index(drop=True)
        self.y_train = self.train_df.iloc[:, self.num_feature:].reset_index(drop=True)

        self.X_test = self.test_df.iloc[:, :self.num_feature].reset_index(drop=True)
        self.y_test = self.test_df.iloc[:, self.num_feature:].reset_index(drop=True)

    def get_train(self, split=True):
        if split:
            return self.X_train.copy(), self.y_train.copy()
        else:
            return self.train_df.copy()

    def get_test(self, split=True):
        if split:
            return self.X_test.copy(), self.y_test.copy()
        else:
            return self.test_df.copy()


    def get_name(self):

        return self.task_name


