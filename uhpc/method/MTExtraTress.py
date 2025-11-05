import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple

# =========================
# 单棵 极端随机树（仅回归 + 任务分裂）
# =========================

@dataclass
class _Node:
    is_leaf: bool
    n_samples: int
    # 叶子
    y_mean: float = 0.0
    # 内部结点（两类分裂的描述其一必填）
    feature: int = -1
    threshold: float = 0.0
    left: int = -1
    right: int = -1
    split_type: str = "feature"  # "feature" or "task"
    left_task_ids: Optional[Set[int]] = None  # 当 split_type == "task" 时使用

class _MTExtraTreeRegressor:
    """
    单棵 MT-ExtraTrees 回归树（实现论文中的 λ 概率任务分裂与 φ_t 定义），只支持标量回归。
    外部：X 为 DataFrame，y 为 Series，task_id 为 Series（int）。
    内部：转为 numpy 计算。
    """
    def __init__(self,
                 max_depth: int = 12,
                 min_samples_split: int = 8,
                 min_samples_leaf: int = 2,
                 max_features: Optional[int] = None,
                 n_thresholds: int = 16,
                 lambda_task_split: float = 0.3,  # 论文中的 λ
                 alpha: float = 1.0,              # 任务特征平滑超参 α
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.lambda_task_split = float(lambda_task_split)
        self.alpha = float(alpha)
        self.rng = np.random.RandomState(random_state)
        self.nodes: List[_Node] = []
        # 训练时绑定
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.task: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.task_ids_all_: Optional[np.ndarray] = None

    # ---- 工具函数 ----
    @staticmethod
    def _neg_variance(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        return -float(np.var(y))  # population var

    def _score_split(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        nL, nR = y_left.size, y_right.size
        if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
            return -np.inf
        return nL * self._neg_variance(y_left) + nR * self._neg_variance(y_right)

    def _feature_subset(self, n_features: int) -> np.ndarray:
        m = n_features if self.max_features is None else min(self.max_features, n_features)
        return self.rng.choice(n_features, size=m, replace=False)

    def _rand_thresholds(self, col: np.ndarray) -> np.ndarray:
        # 只在有限值上取范围
        x = col[np.isfinite(col)]
        if x.size == 0:
            return np.array([])
        lo, hi = np.min(x), np.max(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([])
        return self.rng.uniform(lo, hi, size=self.n_thresholds)

    # ---- 训练主流程 ----
    def fit(self, X: pd.DataFrame, y: pd.Series, task_id: pd.Series):
        X = pd.DataFrame(X)
        y = pd.Series(y).astype(float)
        task_id = pd.Series(task_id).astype(int)

        assert len(X) == len(y) == len(task_id)
        self.X = X.to_numpy(dtype=float, copy=False)
        self.y = y.to_numpy(dtype=float, copy=False).reshape(-1)
        self.task = task_id.to_numpy(dtype=int, copy=False)
        self.n_features_ = X.shape[1]
        self.task_ids_all_ = np.unique(self.task)

        self.nodes = []
        self._build_node(np.arange(X.shape[0], dtype=int), depth=0)
        return self

    def _make_leaf(self, idx: np.ndarray) -> int:
        node = _Node(
            is_leaf=True, n_samples=int(idx.size),
            y_mean=float(np.mean(self.y[idx])) if idx.size > 0 else 0.0
        )
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _build_node(self, idx: np.ndarray, depth: int) -> int:
        # 停止条件
        if depth >= self.max_depth or idx.size < self.min_samples_split:
            return self._make_leaf(idx)

        Xn = self.X[idx]
        yn = self.y[idx]
        tn = self.task[idx]
        parent_mean = float(np.mean(yn))

        # 1) 特征划分候选
        best_score = -np.inf
        best_feature, best_thr = -1, 0.0
        best_left_idx, best_right_idx = None, None
        best_split_type = "feature"
        best_left_task_set = None

        feat_subset = self._feature_subset(self.n_features_)
        for f in feat_subset:
            thrs = self._rand_thresholds(Xn[:, f])
            if thrs.size == 0:
                continue
            col = Xn[:, f]
            for tval in thrs:
                left_mask = np.isfinite(col) & (col < tval)
                left_idx = idx[left_mask]
                right_idx = idx[~left_mask]
                if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
                    continue
                sc = self._score_split(self.y[left_idx], self.y[right_idx])
                if sc > best_score:
                    best_score = sc
                    best_feature, best_thr = f, float(tval)
                    best_left_idx, best_right_idx = left_idx, right_idx
                    best_split_type = "feature"
                    best_left_task_set = None

        # 2) 任务划分候选（按 λ 的概率评估）
        if self.rng.rand() < self.lambda_task_split:
            tasks_here = np.unique(tn)
            if tasks_here.size >= 2:
                gamma_regr = parent_mean
                phi = []
                for t_id in tasks_here:
                    mask_t = (tn == t_id)
                    sum_y_t = float(np.sum(yn[mask_t]))
                    cnt_t = int(np.sum(mask_t))
                    phi_t = (sum_y_t + self.alpha * gamma_regr) / (cnt_t + self.alpha)
                    phi.append((t_id, phi_t))
                phi_vals = np.array([v for _, v in phi], dtype=float)
                lo, hi = phi_vals.min(), phi_vals.max()
                if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
                    c = float(self.rng.uniform(lo, hi))
                    left_task_set = {t_id for (t_id, v) in phi if v <= c}
                    left_mask = np.isin(tn, list(left_task_set))
                    left_idx = idx[left_mask]
                    right_idx = idx[~left_mask]
                    if left_idx.size >= self.min_samples_leaf and right_idx.size >= self.min_samples_leaf:
                        sc = self._score_split(self.y[left_idx], self.y[right_idx])
                        if sc > best_score:
                            best_score = sc
                            best_feature, best_thr = -1, 0.0
                            best_left_idx, best_right_idx = left_idx, right_idx
                            best_split_type = "task"
                            best_left_task_set = set(left_task_set)

        if best_score == -np.inf or best_left_idx is None:
            return self._make_leaf(idx)

        node_id = len(self.nodes)
        self.nodes.append(_Node(
            is_leaf=False, n_samples=int(idx.size),
            feature=int(best_feature), threshold=float(best_thr),
            left=-1, right=-1, split_type=best_split_type,
            left_task_ids=best_left_task_set
        ))
        left_id = self._build_node(best_left_idx, depth + 1)
        right_id = self._build_node(best_right_idx, depth + 1)
        self.nodes[node_id].left = left_id
        self.nodes[node_id].right = right_id
        return node_id

    # ---- 预测 ----
    def _traverse_one(self, x: np.ndarray, t_id: int) -> float:
        i = 0
        while True:
            node = self.nodes[i]
            if node.is_leaf:
                return node.y_mean
            if node.split_type == "feature":
                go_left = (np.isfinite(x[node.feature]) and (x[node.feature] < node.threshold))
            else:  # task split
                go_left = (node.left_task_ids is not None) and (t_id in node.left_task_ids)
            i = node.left if go_left else node.right

    def predict(self, X: pd.DataFrame, task_id: pd.Series) -> np.ndarray:
        X = pd.DataFrame(X)
        task_id = pd.Series(task_id).astype(int)
        assert len(X) == len(task_id)
        Xv = X.to_numpy(dtype=float, copy=False)
        tv = task_id.to_numpy(dtype=int, copy=False)
        yhat = np.empty(len(X), dtype=float)
        for i in range(len(X)):
            yhat[i] = self._traverse_one(Xv[i], int(tv[i]))
        return yhat

# =========================
# 森林封装
# =========================

class MTExtraTreesRegressor:
    """
    只回归的 MT-ExtraTrees（DataFrame/Series 接口）
    """
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 14,
                 min_samples_split: int = 8,
                 min_samples_leaf: int = 2,
                 max_features: Optional[int] = None,
                 n_thresholds: int = 16,
                 lambda_task_split: float = 0.3,
                 alpha: float = 1.0,
                 bootstrap: bool = False,
                 random_state: Optional[int] = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.lambda_task_split = lambda_task_split
        self.alpha = alpha
        self.bootstrap = bootstrap
        self.rng = np.random.RandomState(random_state)
        self.trees: List[_MTExtraTreeRegressor] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, task_id: pd.Series):
        X = pd.DataFrame(X)
        y = pd.Series(y).astype(float)
        task_id = pd.Series(task_id).astype(int)
        n = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            rs = int(self.rng.randint(0, 2**31 - 1))
            tree = _MTExtraTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                lambda_task_split=self.lambda_task_split,
                alpha=self.alpha,
                random_state=rs
            )
            if self.bootstrap:
                idx = self.rng.randint(0, n, size=n)
                tree.fit(X.iloc[idx].reset_index(drop=True),
                         y.iloc[idx].reset_index(drop=True),
                         task_id.iloc[idx].reset_index(drop=True))
            else:
                tree.fit(X.reset_index(drop=True),
                         y.reset_index(drop=True),
                         task_id.reset_index(drop=True))
            self.trees.append(tree)
        return self

    def predict(self, X: pd.DataFrame, task_id: pd.Series) -> np.ndarray:
        preds = [t.predict(X, task_id) for t in self.trees]
        return np.mean(np.vstack(preds), axis=0)

# =========================
# 仅 DataFrame/Series 的外部封装
# =========================
def mtet_fit_predict(train_x: pd.DataFrame,
                     train_y: pd.DataFrame,
                     test_x: pd.DataFrame,
                     test_y: pd.DataFrame,
                     *,
                     n_estimators: int = 150,
                     max_depth: int = 12,
                     min_samples_split: int = 10,
                     min_samples_leaf: int = 3,
                     max_features: Optional[int] = None,
                     n_thresholds: int = 24,
                     lambda_task_split: float = 0.4,
                     alpha: float = 1.0,
                     bootstrap: bool = False,
                     random_state: Optional[int] = 42
                     ) -> pd.DataFrame:
    """
    仅用 DataFrame/Series: train_x, train_y, test_x, test_y 训练并预测。
    - 若 y 为 Series 或 (n,1) DataFrame：按“单任务”（task=0）。
    - 若 y 为 (n,T) DataFrame：每列一个任务，内部长表展开训练，预测后还原为 (m,T)。
    返回：与 test_y 类型/形状一致的 pred_y（仅预测，不返回模型）。
    """

    X_tr = pd.DataFrame(train_x).copy()
    X_te = pd.DataFrame(test_x).copy()
    y_tr_in = train_y.copy()
    y_te_in = test_y.copy()

    if len(X_tr) != len(y_tr_in):
        raise ValueError("train_x 与 train_y 行数不一致")
    if len(X_te) != len(y_te_in):
        raise ValueError("test_x 与 test_y 行数不一致")

    d = X_tr.shape[1]
    if max_features is None:
        max_features = max(1, int(np.sqrt(d)))

    model = MTExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_thresholds=n_thresholds,
        lambda_task_split=lambda_task_split,
        alpha=alpha,
        bootstrap=bootstrap,
        random_state=random_state
    )


    T = y_tr_in.shape[1]

    n_train = len(X_tr); n_test = len(X_te)

    X_tr_long = pd.concat([X_tr.reset_index(drop=True) for _ in range(T)], ignore_index=True)
    y_tr_long = pd.concat([y_tr_in.iloc[:, t].reset_index(drop=True) for t in range(T)], ignore_index=True)
    t_tr = pd.concat([pd.Series(t, index=np.arange(n_train), dtype=int) for t in range(T)],
                     ignore_index=True)
    valid_mask = y_tr_long.notna()
    if valid_mask.sum() < len(y_tr_long):
        X_tr_long = X_tr_long[valid_mask].reset_index(drop=True)
        y_tr_long = y_tr_long[valid_mask].reset_index(drop=True)
        t_tr = t_tr[valid_mask].reset_index(drop=True)

    X_te_long = pd.concat([X_te.reset_index(drop=True) for _ in range(T)], ignore_index=True)
    t_te = pd.concat([pd.Series(t, index=np.arange(n_test), dtype=int) for t in range(T)],
                     ignore_index=True)

    model.fit(X_tr_long, y_tr_long, t_tr)
    pred_long = model.predict(X_te_long, t_te)   # 长向量，长度 T*n_test
    pred_mat = pred_long.reshape(T, n_test).T

    return pd.DataFrame(pred_mat, index=X_te.index, columns=y_tr_in.columns)

