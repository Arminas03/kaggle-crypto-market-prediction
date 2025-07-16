import lightgbm
from lightgbm import early_stopping, log_evaluation
import delu
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

from utils.data_preprocessor import DataPreprocessor
from important_cols import *


OVERFIT_PARAMS = {
    "n_estimators": 1500,
    "max_depth": 12,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.01,
    "reg_lambda": 0.01,
    "min_child_samples": 100,
    "objective": "regression",
    "random_state": 0,
    "verbosity": -1,
}

UNDERFIT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "reg_alpha": 1,
    "reg_lambda": 1,
    "min_child_samples": 20,
    "objective": "regression",
    "random_state": 0,
    "verbosity": -1,
}

BALANCED_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 50,
    "objective": "regression",
    "random_state": 0,
    "verbosity": -1,
}


class LightGBM:
    def __init__(self, params=None):
        self.params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "min_child_samples": 36,
            "objective": "regression",
            "random_state": 0,
            "verbosity": -1,
        }
        if params:
            self.params.update(params)

        self.setup_model()

    def setup_model(self):
        self.model = lightgbm.LGBMRegressor(**self.params)

    def train_model(self, data, no_print=False):
        if not no_print:
            print("training...")
        timer = delu.tools.Timer()
        timer.run()

        self.model.fit(data["train"]["X"], data["train"]["y"])
        if not no_print:
            print(f"train time: {timer}")

    def train_val(self, data, no_print=False):
        self.train_model(data, no_print)

        y_pred_val = self.get_y_pred(data, "val")
        val_loss = sklearn.metrics.mean_squared_error(y_pred_val, data["val"]["y"])
        corr = pearsonr(y_pred_val.ravel(), data["val"]["y"].values.ravel()).statistic

        if not no_print:
            print(f"val loss: {val_loss:.4f}")
            print(f"Pearson corr: {corr}")
            print("-" * 40)

        return val_loss

    def get_y_pred(self, data, split):
        return self.model.predict(data[split]["X"])

    def test(self, data, no_print=False):
        y_pred = self.get_y_pred(data, "test")
        test_loss = sklearn.metrics.mean_squared_error(y_pred, data["test"]["y"])
        corr = pearsonr(y_pred.ravel(), data["test"]["y"].values.ravel()).statistic

        if not no_print:
            print(f"test loss: {test_loss:.4f}")
            print(f"Pearson corr: {corr}")
            print("-" * 40)

        return test_loss

    @staticmethod
    def run(path_to_data_file):
        model = LightGBM()

        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_data_file)
        # data = data_preprocessor.get_preprocessed_data(standardize=False)
        # y_rescale_factor = data_preprocessor.get_y_std() ** 2

        # model.train_val(data, y_rescale_factor)

        data = data_preprocessor.get_preprocessed_data(
            split_val=False, standardize=False
        )
        data = data_preprocessor.del_cols(data)
        # data_preprocessor.select_k_best(data, False, 400)
        y_rescale_factor = data_preprocessor.get_y_std() ** 2

        model.train_model(data)
        model.test(data, y_rescale_factor)

    @staticmethod
    def run_submit(path_to_input, path_to_save):
        models = {
            "overfit": LightGBM(params=OVERFIT_PARAMS),
            "underfit": LightGBM(params=UNDERFIT_PARAMS),
            "balanced": LightGBM(params=BALANCED_PARAMS),
        }

        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_input)

        data = data_preprocessor._get_data_from_file()
        data = data[good_cols_manual + ["label"]]
        data_preprocessor.add_features(data)

        X, y = data.drop(columns="label"), data[["label"]]

        all_idx = np.arange(len(y))
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            all_idx, train_size=0.9, random_state=4
        )

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        submit_data = pd.read_parquet("data/test.parquet")
        submit_data = submit_data[good_cols_manual + ["label"]]
        data_preprocessor.add_features(submit_data)

        data = {
            "train": {"X": X_train, "y": y_train},
            "val": {"X": X_val, "y": y_val},
            "submit": {"X": submit_data.drop(columns="label")},
        }

        models["balanced"].train_model(data)
        y_pred = models["balanced"].get_y_pred(data, "submit")

        # val_preds = dict()
        # for name, model in models.items():
        #     model.train_model(data)
        #     val_preds[name] = model.get_y_pred(data, "val")

        # y_pred_stack = np.column_stack([val_pred for val_pred in val_preds.values()])

        # ridge = Ridge(fit_intercept=False, alpha=1)

        # ridge.fit(y_pred_stack, data["val"]["y"])
        # weights = dict()
        # for name, coef in zip(val_preds.keys(), ridge.coef_):
        #     print(f"{name}: {coef:.4f}")
        #     weights[name] = coef

        # weight_sum = sum(weights.values())
        # for name in weights:
        #     weights[name] = weights[name] / weight_sum

        # print(f"Weights: {weights}")

        # y_pred = np.zeros(len(data["submit"]["X"]))

        # for name, model in models.items():
        #     y_pred += (1 / 3) * model.get_y_pred(data, "submit")

        pd.DataFrame(
            {"ID": np.arange(1, len(y_pred) + 1), "prediction": y_pred}
        ).to_csv(path_to_save, index=False)
