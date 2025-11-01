"""This module contains `RFE_MissForest` code."""
from collections import OrderedDict
from copy import deepcopy
from typing import Union
import numpy as np
import pandas as pd

from uhpc.helpers.errors import NotFittedError
from uhpc.helpers._validate import (
    _validate_clf,
    _validate_rgr,
    _validate_initial_guess,
    _validate_max_iter,
    _validate_early_stopping,
    _validate_feature_dtype_consistency,
    _validate_2d,
    _validate_cat_var_consistency,
    _validate_categorical,
    _validate_infinite,
    _validate_empty_feature,
    _validate_imputable,
    _validate_verbose,
    _validate_column_consistency,
)
from uhpc.helpers._metrics import pfc, nrmse
from uhpc.helpers._array import SafeArray
from typing import Any, Iterable, Dict
from sklearn.base import BaseEstimator
from tqdm import tqdm
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, StratifiedKFold

def _validate_feature_selection(fs):
    if not isinstance(fs, bool):
        raise ValueError("Argument `feature_selection` must be a boolean.")

# lgbm_clf = LGBMClassifier(verbosity=-1, linear_tree=True)
# lgbm_rgr = LGBMRegressor(verbosity=-1, linear_tree=True)
# xgb_clf = XGBClassifier(verbosity=0)
# xgb_rgr = XGBRegressor(verbosity=0)

class RFE_MissForest:
    """
    Attributes
    ----------
    classifier : Union[Any, BaseEstimator]
        Estimator that predicts missing values of categorical columns.
    regressor : Union[Any, BaseEstimator]
        Estimator that predicts missing values of numerical columns.
    initial_guess : str
        Determines the method of initial imputation.
    max_iter : int
        Maximum iterations of imputing.
    early_stopping : bool
        Determines if early stopping will be executed.
    _categorical : list
        All categorical columns of given dataframe `x`.
    _numerical : list
        All numerical columns of given dataframe `x`.
    column_order : pd.Index
        Sorting order of features.
    _is_fitted : bool
        A state that determines if an instance of `RFE_MissForest` is fitted.
    _estimators : list
        A ordered dictionary that stores estimators for each feature of each
        iteration.
    _verbose : int
        Determines if messages will be printed out.
    feature_selection : bool
        Determines if feature selection will be executed.
    _selected_features : list
        A list of dictionaries that stores selected features for each
        iteration.
    rfe_n_estimators : int
        The number of trees in the forest for RF-RFE.
    rfe_cv : int
        The number of cross-validation folds for RF-RFE.
    random_state : int
        The random state for RF-RFE.

    Methods
    -------
    _get_n_missing(x: pd.DataFrame)
        Compute and return the total number of missing values in `x`.
    _get_missing_indices(x: pd.DataFrame)
        Gather the indices of any rows that have missing values.
    _compute_initial_imputations(self, x: pd.DataFrame,
                                     categorical: Iterable[Any])
        Computes and stores the initial imputation values for each feature
        in `x`.
    _initial_impute(x: pd.DataFrame,
                        initial_imputations: Dict[Any, Union[str, np.float64]])
        Imputes the values of features using the mean or median for
        numerical variables; otherwise, uses the mode for imputation.
    fit(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
        Fit `RFE_MissForest`.
    transform(self, x: pd.DataFrame)
        Imputes all missing values in `x` with fitted estimators.
    fit_transform(self, x: pd.DataFrame, categorical: Iterable[Any] = None)
        Calls class methods `fit` and `transform` on `x`.
    """

    def __init__(self, clf: Union[Any, BaseEstimator] = None,
                 rgr: Union[Any, BaseEstimator] = None,
                 categorical: Iterable[Any] = None,
                 initial_guess: str = "median", max_iter: int = 5,
                 early_stopping: bool = True,
                 feature_selection: bool = False,
                 rfe_n_estimators: int = 100,
                 rfe_cv: int = 5,
                 rfe_step: int = 1,
                 random_state: int = 42,
                 verbose: int = 2) -> None:
        """
        Parameters
        ----------
        clf : estimator object, default=None.
            This object is assumed to implement the scikit-learn estimator api.
        rgr : estimator object, default=None.
            This object is assumed to implement the scikit-learn estimator api.
        categorical : Iterable[Any], default=None
            All categorical features of `x`.
        max_iter : int, default=5
            Determines the number of iteration.
        initial_guess : str, default=`median`
            If `mean`, initial imputation will be the mean of the features.
            If `median`, initial imputation will be the median of the features.
        early_stopping : bool
            Determines if early stopping will be executed.
        feature_selection : bool, default=False
            Whether to perform feature selection before training the imputation model.
        rfe_n_estimators : int, default=100
            The number of trees in the forest for RF-RFE.
        rfe_cv : int, default=5
            The number of cross-validation folds for RF-RFE.
        random_state : int, default=42
            The random state for RF-RFE.
        verbose : int
            Determines if message will be printed out.

        Raises
        ------
        ValueError
            - If argument `clf` is not an estimator.
            - If argument `rgr` is not an estimator.
            - If argument `categorical` is not a list of strings or NoneType.
            - If argument `categorical` is NoneType and has a length of less
              than one.
            - If argument `initial_guess` is not a str.
            - If argument `initial_guess` is neither `mean` nor `median`.
            - If argument `max_iter` is not an int.
            - If argument `early_stopping` is not a bool.
        """

        _validate_categorical(categorical)
        _validate_initial_guess(initial_guess)
        _validate_max_iter(max_iter)
        _validate_early_stopping(early_stopping)
        _validate_verbose(verbose)
        _validate_feature_selection(feature_selection)
        self.random_state = random_state
        # self.classifier = clf if clf is not None else XGBClassifier(verbosity=0, random_state=self.random_state)
        # self.regressor = rgr if rgr is not None else XGBRegressor(verbosity=0, random_state=self.random_state)
        self.classifier = clf if clf is not None else ExtraTreesClassifier(random_state=self.random_state)
        self.regressor = rgr if rgr is not None else ExtraTreesRegressor(random_state=self.random_state)
        # self.classifier = clf if clf is not None else LGBMClassifier(verbosity=-1, linear_tree=True, random_state=self.random_state)
        # self.regressor = rgr if rgr is not None else LGBMRegressor(verbosity=-1, linear_tree=True, random_state=self.random_state)
        self._categorical = [] if categorical is None else categorical
        self.initial_guess = initial_guess
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.feature_selection = feature_selection
        self.rfe_n_estimators = rfe_n_estimators
        self.rfe_cv = rfe_cv
        self.rfe_step = rfe_step
        self._numerical = None
        self.column_order = None
        self.initial_imputations = None
        self._is_fitted = False
        self._estimators = []
        self._selected_features = []
        # self.selected_features_ = {}
        self._verbose = verbose
        _validate_clf(self.classifier)
        _validate_rgr(self.regressor)

    @staticmethod
    def rf_rfe(y_obs, X_obs,
               n_estimators: int = 100,
               cv: int = 5,
               step: int = 1,
               random_state: int = 42):
        """
        Recursive Feature Elimination using Random Forest (RF-RFE).

        Parameters
        ----------
        y_obs : pd.Series or array-like of shape (n_samples,)
            目标变量的观测值。连续值将触发回归，否则触发分类。
        X_obs : pd.DataFrame or array-like of shape (n_samples, n_features)
            预测变量的观测值。
        n_estimators : int, default=100
            随机森林中树的棵数。
        cv : int, default=5
            交叉验证折数。
        step : int, default=1
            每轮迭代剔除的特征数目。
        random_state : int, default=42
            随机种子，保证结果可复现。

        Returns
        -------
        selected_features : list
            被 RF-RFE 筛选出的最优特征名（若 X_obs 是 DataFrame）或特征索引列表。
        rfecv : RFECV
            完成拟合的 RFECV 对象，包含更多中间结果（比如 ranking_、grid_scores_）。
        """
        # ———————— 1. 格式预处理 ————————
        # 把 X 转成 numpy，记录列名
        if isinstance(X_obs, pd.DataFrame):
            feature_names = X_obs.columns.tolist()
            X = X_obs.values
        else:
            X = np.asarray(X_obs)
            feature_names = list(range(X.shape[1]))
        # 把 y 转成 pandas.Series（方便判别 dtype）
        if not isinstance(y_obs, pd.Series):
            y = pd.Series(y_obs)
        else:
            y = y_obs

        # ———————— 2. 根据 y 的类型选择模型和评分 ————————
        if y.dtype.kind == 'f':  # float → 回归
            estimator = RandomForestRegressor(n_estimators=n_estimators,
                                              n_jobs=-1,
                                              random_state=random_state)
            cv_strategy = KFold(n_splits=cv,
                                shuffle=True,
                                random_state=random_state)
            scoring = 'neg_mean_squared_error'
        else:  # int/object → 分类
            estimator = RandomForestClassifier(n_estimators=n_estimators,
                                               n_jobs=-1,
                                               random_state=random_state)
            cv_strategy = StratifiedKFold(n_splits=cv,
                                          shuffle=True,
                                          random_state=random_state)
            scoring = 'accuracy'

        # ———————— 3. RFECV 执行 RFE ————————
        rfecv = RFECV(estimator=estimator,
                      step=step,
                      cv=cv_strategy,
                      scoring=scoring,
                      n_jobs=-1)
        rfecv.fit(X, y)

        # ———————— 4. 返回结果 ————————
        mask = rfecv.support_
        selected = [feature_names[i] for i, keep in enumerate(mask) if keep]
        return selected, rfecv

    @staticmethod
    def _get_n_missing(x: pd.DataFrame) -> int:
        """Compute and return the total number of missing values in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        int
            Total number of missing values in `x`.
        """
        return int(x.isnull().sum().sum())

    @staticmethod
    def _get_missing_indices(x: pd.DataFrame) -> Dict[Any, pd.Index]:
        """Gather the indices of any rows that have missing values.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        missing_indices : dict
            Dictionary containing features with missing values as keys,
            and their corresponding indices as values.
        """
        missing_indices = {}
        for c in x.columns:
            feature = x[c]
            missing_indices[c] = feature[feature.isnull()].index

        return missing_indices

    def _compute_initial_imputations(self, x: pd.DataFrame,
                                     categorical: Iterable[Any]
                                     ) -> Dict[Any, Union[str, np.float64]]:
        """Computes and stores the initial imputation values for each feature
        in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            The dataset consisting solely of features that require imputation.
        categorical : Iterable[Any]
            An iterable containing identifiers for all categorical features
            present in `x`.

        Raises
        ------
        ValueError
            - If any feature specified in the `categorical` argument does not
            exist within the columns of `x`.
            - If argument `initial_guess` is provided and its value is
            neither `mean` nor `median`.
        """
        initial_imputations = {}
        for c in x.columns:
            if c in categorical:
                initial_imputations[c] = x[c].mode().values[0]
            elif c not in categorical and self.initial_guess == "mean":
                initial_imputations[c] = x[c].mean()
            elif c not in categorical and self.initial_guess == "median":
                initial_imputations[c] = x[c].median()
            elif c not in categorical:
                raise ValueError("Argument `initial_guess` only accepts "
                                 "`mean` or `median`.")

        return initial_imputations

    @staticmethod
    def _initial_impute(x: pd.DataFrame,
                        initial_imputations: Dict[Any, Union[str, np.float64]]
                        ) -> pd.DataFrame:
        """Imputes the values of features using the mean or median for
        numerical variables; otherwise, uses the mode for imputation.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.
        initial_imputations : dict
            Dictionary containing initial imputation values for each feature.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """
        x = x.copy()
        for c in x.columns:
            x[c] = x[c].fillna(initial_imputations[c])

        return x

    def _is_stopping_criterion_satisfied(self, pfc_score: SafeArray,
                                         nrmse_score: SafeArray) -> bool:
        """Checks if stopping criterion satisfied. If satisfied, return True.
        Otherwise, return False.

        Parameters
        ----------
        pfc_score : SafeArray
            Latest 2 PFC scores.
        nrmse_score : SafeArray
            Latest 2 NRMSE scores.

        Returns
        -------
        bool
            - True, if stopping criterion satisfied.
            - False, if stopping criterion not satisfied.
        """
        is_pfc_increased = False
        if any(self._categorical) and len(pfc_score) >= 2:
            is_pfc_increased = pfc_score[-1] > pfc_score[-2]

        is_nrmse_increased = False
        if any(self._numerical) and len(nrmse_score) >= 2:
            is_nrmse_increased = nrmse_score[-1] > nrmse_score[-2]

        if (
                any(self._categorical) and
                any(self._numerical) and
                is_pfc_increased * is_nrmse_increased
        ):
            if self._verbose >= 2:
                warnings.warn("Both PFC and NRMSE have increased.")

            return True
        elif (
                any(self._categorical) and
                not any(self._numerical) and
                is_pfc_increased
        ):
            if self._verbose >= 2:
                warnings.warn("PFC have increased.")

            return True
        elif (
                not any(self._categorical) and
                any(self._numerical) and
                is_nrmse_increased
        ):
            if self._verbose >= 2:
                warnings.warn("NRMSE increased.")

            return True

        return False

    def fit(self, x: pd.DataFrame):
        """Fit `RFE_MissForest`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        x : pd.DataFrame of shape (n_samples, n_features)
            Reverse label-encoded dataset (features only).

        Raises
        ------
        ValueError
            - If argument `x` is not a pandas DataFrame or NumPy array.
            - If argument `categorical` is not a list of strings or NoneType.
            - If argument `categorical` is NoneType and has a length of
              less than one.
            - If there are inf values present in argument `x`.
            - If there are one or more columns with all rows missing.
        """

        self.original_column_order = x.columns


        if self._verbose >= 2:
            warnings.warn("Label encoding is no longer performed by default. "
                          "Users will have to perform categorical features "
                          "encoding by themselves.")

        x = x.copy()

        # Make sure `x` is either pandas dataframe, numpy array or list of
        # lists.
        if (
                not isinstance(x, pd.DataFrame) and
                not isinstance(x, np.ndarray)
        ):
            raise ValueError("Argument `x` can only be pandas dataframe, "
                             "numpy array or list of list.")

        # If `x` is a list of list, convert `x` into a pandas dataframe.
        if (
                isinstance(x, np.ndarray) or
                (isinstance(x, list) and all(isinstance(i, list) for i in x))
        ):
            x = pd.DataFrame(x)

        _validate_2d(x)
        _validate_empty_feature(x)
        _validate_feature_dtype_consistency(x)
        _validate_imputable(x)
        _validate_cat_var_consistency(x.columns, self._categorical)

        if any(self._categorical):
            _validate_infinite(x.drop(self._categorical, axis=1))
        else:
            _validate_infinite(x)

        self._numerical = [c for c in x.columns if c not in self._categorical]

        # Sort column order according to the amount of missing values
        # starting with the lowest amount.
        pct_missing = x.isnull().sum() / len(x)
        self.column_order = pct_missing.sort_values().index
        x = x[self.column_order].copy()

        n_missing = self._get_n_missing(x[self._categorical])
        missing_indices = self._get_missing_indices(x)
        self.initial_imputations = self._compute_initial_imputations(
            x, self._categorical
        )
        x_imp = self._initial_impute(x, self.initial_imputations)

        x_imp_cat = SafeArray(dtype=pd.DataFrame)
        x_imp_num = SafeArray(dtype=pd.DataFrame)
        pfc_score = SafeArray(dtype=float)
        nrmse_score = SafeArray(dtype=float)

        loop = range(self.max_iter)
        if self._verbose >= 1:
            loop = tqdm(loop)

        for _ in loop:
            fitted_estimators = OrderedDict()
            iter_selected_features = {}

            for c in missing_indices:
                if c in self._categorical:
                    estimator = deepcopy(self.classifier)
                else:
                    estimator = deepcopy(self.regressor)

                # Fit estimator with imputed x.
                x_obs_all = x_imp.drop(c, axis=1)
                y_obs = x_imp[c]

                if self.feature_selection:
                    if self._verbose >= 2:
                        print(f"Performing feature selection for column '{c}'...")
                    selected_features, _ = self.rf_rfe(
                        y_obs,
                        x_obs_all,
                        n_estimators=self.rfe_n_estimators,
                        cv=self.rfe_cv,
                        step=self.rfe_step,
                        random_state=self.random_state)
                    iter_selected_features[c] = selected_features
                    x_obs = x_obs_all[selected_features]
                    if self._verbose >= 2:
                        print(f"Selected {len(selected_features)} features for '{c}': {selected_features}")
                else:
                    x_obs = x_obs_all

                estimator.fit(x_obs, y_obs)

                # Predict the missing column with the trained estimator.
                if self.feature_selection:
                    x_missing = x_imp.loc[missing_indices[c]].drop(c, axis=1)[iter_selected_features[c]]
                else:
                    x_missing = x_imp.loc[missing_indices[c]].drop(c, axis=1)

                if not x_missing.empty:
                    # Update imputed matrix.
                    x_imp.loc[missing_indices[c], c] = (
                        estimator.predict(x_missing).tolist()
                    )
                print(f"Imputed column '{c}' with {x_obs.shape}")
                # Store trained estimators.
                fitted_estimators[c] = estimator

            if self.feature_selection:
                self._selected_features.append(iter_selected_features)

            self._estimators.append(fitted_estimators)

            # Store imputed categorical and numerical features after
            # each iteration.
            # Compute and store PFC.
            if any(self._categorical):
                x_imp_cat.append(x_imp[self._categorical])

                if len(x_imp_cat) >= 2:
                    pfc_score.append(
                        pfc(
                            x_true=x_imp_cat[-1],
                            x_imp=x_imp_cat[-2],
                            n_missing=n_missing,
                        )
                    )

            # Compute and store NRMSE.
            if any(self._numerical):
                x_imp_num.append(x_imp[self._numerical])

                if len(x_imp_num) >= 2:
                    nrmse_score.append(
                        nrmse(
                            x_true=x_imp_num[-1],
                            x_imp=x_imp_num[-2],
                        )
                    )

            if (
                    self.early_stopping and
                    self._is_stopping_criterion_satisfied(
                        pfc_score,
                        nrmse_score
                    )):
                self._is_fitted = True
                if self._verbose >= 2:
                    warnings.warn(
                        "Stopping criterion triggered during fitting. "
                        "Before last imputation matrix will be returned."
                    )

                # Remove last iteration of estimators.
                self._estimators = self._estimators[:-1]

                return self

        self._is_fitted = True

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Imputes all missing values in `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            - Before last imputation matrix, if stopping criterion is
              triggered.
            - Last imputation matrix, if all iterations are done.

        Raises
        ------
        NotFittedError
            If `RFE_MissForest` is not fitted.
        ValueError
            If there are no missing values in `x`.
        """
        if self._verbose >= 2:
            warnings.warn("Label encoding is no longer performed by default. "
                          "Users will have to perform categorical features "
                          "encoding by themselves.")

            warnings.warn(f"In version, estimator fitting process "
                          f"is moved to `fit` method. `RFE_MissForest` will now "
                          f"imputes unseen missing values with fitted "
                          f"estimators with `transform` method. To retain the "
                          f"old behaviour, use `fit_transform` to fit the "
                          f"whole unseen data instead.")

        if not self._is_fitted:
            raise NotFittedError("RFE_MissForest is not fitted yet.")

        _validate_2d(x)
        _validate_empty_feature(x)
        _validate_feature_dtype_consistency(x)
        _validate_imputable(x)
        _validate_cat_var_consistency(x.columns, self._categorical)
        _validate_column_consistency(set(x.columns), set(self.column_order))

        x = x[self.column_order].copy()

        n_missing = self._get_n_missing(x[self._categorical])
        missing_indices = self._get_missing_indices(x)
        x_imp = self._initial_impute(x, self.initial_imputations)

        x_imps = SafeArray(dtype=pd.DataFrame)
        x_imp_cat = SafeArray(dtype=pd.DataFrame)
        x_imp_num = SafeArray(dtype=pd.DataFrame)
        pfc_score = SafeArray(dtype=float)
        nrmse_score = SafeArray(dtype=float)

        loop = range(len(self._estimators))
        if self._verbose >= 1:
            loop = tqdm(loop)

        for i in loop:
            for feature, estimator in self._estimators[i].items():
                if x[feature].isnull().any():
                    if self.feature_selection:
                        selected_features = self._selected_features[i][feature]
                        x_obs = (
                            x_imp.loc[missing_indices[feature]]
                            .drop(feature, axis=1)[selected_features]
                        )
                        print(feature)
                        print(selected_features)
                        print(x_obs.shape)
                    else:
                        x_obs = (
                            x_imp.loc[missing_indices[feature]]
                            .drop(feature, axis=1)
                        )

                    x_imp.loc[missing_indices[feature], feature] = (
                        estimator.predict(x_obs).tolist()
                    )
            x_imp = x_imp[self.original_column_order]
            # Store imputed categorical and numerical features after
            # each iteration.
            if any(self._categorical):
                x_imp_cat.append(x_imp[self._categorical])

                # Compute and store PFC.
                if len(x_imp_cat) >= 2:
                    pfc_score.append(
                        pfc(
                            x_true=x_imp_cat[-1],
                            x_imp=x_imp_cat[-2],
                            n_missing=n_missing,
                        )
                    )

            if any(self._numerical):
                x_imp_num.append(x_imp[self._numerical])

                # Compute and store NRMSE.
                if len(x_imp_num) >= 2:
                    nrmse_score.append(
                        nrmse(
                            x_true=x_imp_num[-1],
                            x_imp=x_imp_num[-2],
                        )
                    )

            x_imps.append(x_imp)

            if (
                    self.early_stopping and
                    self._is_stopping_criterion_satisfied(
                        pfc_score,
                        nrmse_score
                    )):
                if self._verbose >= 2:
                    warnings.warn(
                        "Stopping criterion triggered during transform. "
                        "Before last imputation matrix will be returned."
                    )

                return x_imps[-2]

        return x_imps[-1]

    def fit_transform(self, x: pd.DataFrame = None) -> pd.DataFrame:
        """Calls class methods `fit` and `transform` on `x`.

        Parameters
        ----------
        x : pd.DataFrame of shape (n_samples, n_features)
            Dataset (features only) that needs to be imputed.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            Imputed dataset (features only).
        """
        return self.fit(x).transform(x)
