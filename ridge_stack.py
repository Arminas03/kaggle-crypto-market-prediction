from sklearn.linear_model import Ridge
import sklearn.metrics
import numpy as np

from models.xgboost_model import XGBoost
from models.lightgbm_model import LightGBM
from models.linear_regression_model import LR
from utils.data_preprocessing import data_setup


def get_model_preds(split, data):
    model_xgb = XGBoost()
    model_lgbm = LightGBM()
    model_linear = LR()

    model_xgb.train_model(data, no_print=True)
    model_lgbm.train_model(data, no_print=True)
    model_linear.fit_model(data, no_print=True)

    return (
        model_xgb.get_y_pred(data, split),
        model_lgbm.get_y_pred(data, split),
        model_linear.get_y_pred(data, split),
    )


def get_ridge_ensemble_coefs():
    data = data_setup("data/train.parquet", "data/test.parquet", True)

    y_pred_xgb, y_pred_lgbm, y_pred_ridge = get_model_preds("val", data)
    y_pred_stack = np.column_stack([y_pred_xgb, y_pred_lgbm, y_pred_ridge])

    ridge = Ridge(fit_intercept=False, alpha=1)

    ridge.fit(y_pred_stack, data["val"]["y"])

    model_names = ["XGBoost", "LightGBM", "Ridge"]
    for name, coef in zip(model_names, ridge.coef_):
        print(f"{name}: {coef:.4f}")

    return ridge.coef_


def test_ridge_stack(weights):
    data = data_setup("data/train.parquet", "data/test.parquet", False)
    y_pred_xgb, y_pred_lgbm, y_pred_ridge = get_model_preds("test", data)

    y_pred = (
        weights["xgb"] * y_pred_xgb
        + weights["lgbm"] * y_pred_lgbm
        + weights["lr"] * y_pred_ridge
    )

    print(f"mse: {sklearn.metrics.mean_squared_error(y_pred, data["test"]["y"])}")


if __name__ == "__main__":
    weights = {
        "lgbm": 0.4,
        "lr": 0.4,
        "xgb": 0.2,
    }
    test_ridge_stack(weights)
