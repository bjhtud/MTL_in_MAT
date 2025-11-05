'''
梯度提升决策树（GBDT）框架
特征可以缺失
'''
from sklearn.ensemble import GradientBoostingRegressor

def gbdt_fit_predict(train_x, train_y, test_x,):
    mask = ~train_y.isna().any(axis=1)
    train_x = train_x.loc[mask]
    train_y = train_y.loc[mask]

    reg = GradientBoostingRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-4
    )

    reg.fit(train_x, train_y)
    pred_y = reg.predict(test_x)
    return pred_y
