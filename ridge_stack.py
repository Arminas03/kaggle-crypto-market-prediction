from sklearn.linear_model import Ridge
import sklearn.metrics
import numpy as np

from utils.data_preprocessor import DataPreprocessor
from models.xgboost_model import XGBoost
from models.lightgbm_model import LightGBM
from models.linear_regression_model import LR


def get_model_preds(split, data, data_k_best):
    model_xgb = XGBoost()
    model_lgbm = LightGBM()
    model_huber = LR()

    model_xgb.train_model(data, no_print=True)
    model_lgbm.train_model(data_k_best, no_print=True)
    model_huber.fit_model(data)

    return (
        model_xgb.get_y_pred(data, split),
        model_lgbm.get_y_pred(data_k_best, split),
        model_huber.get_y_pred(data, split),
    )


def get_ridge_ensemble_coefs():
    data_preprocessor = DataPreprocessor()

    data = data_preprocessor.get_preprocessed_data()
    data_k_best = data.copy()
    data_preprocessor.select_k_best(data_k_best, False, 400)

    y_true = data_preprocessor.get_preprocessed_data()["val"]["y"]
    y_pred_xgb, y_pred_lgbm, y_pred_huber = get_model_preds("val", data, data_k_best)
    y_pred_stack = np.column_stack([y_pred_xgb, y_pred_lgbm, y_pred_huber])

    ridge = Ridge(fit_intercept=False, alpha=1)

    ridge.fit(y_pred_stack, y_true)

    model_names = ["XGBoost", "LightGBM", "Huber"]
    for name, coef in zip(model_names, ridge.coef_):
        print(f"{name}: {coef:.4f}")

    return ridge.coef_


def test_ridge_stack(coefs, data_preprocessor: DataPreprocessor):
    data = data_preprocessor.get_preprocessed_data(split_val=False)
    data_k_best = data.copy()
    data_preprocessor.select_k_best(data_k_best, False, 400)
    y_scale_factor = data_preprocessor.get_y_std() ** 2

    y_pred_xgb, y_pred_lgbm, y_pred_huber = get_model_preds("test", data, data_k_best)

    y_pred = coefs[0] * y_pred_xgb + coefs[1] * y_pred_lgbm + coefs[2] * y_pred_huber

    print(
        f"mse: {sklearn.metrics.mean_squared_error(y_pred, data["test"]["y"])*y_scale_factor}"
    )


if __name__ == "__main__":
    test_ridge_stack(
        [0.6, 0.3, 0.1],
        DataPreprocessor(),
    )
