import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import Ridge

from models.lightgbm_model import LightGBM
from models.xgboost_model import XGBoost
from models.linear_regression_model import LR
from models.tabm_model import ModelTabM
from utils.data_preprocessing import *
from important_cols import *

OVERFIT_PARAMS_LGBM = {
    "n_estimators": 1500,
    "max_depth": 12,
    "learning_rate": 0.01,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.01,
    "reg_lambda": 0.01,
    "min_child_samples": 100,
    "objective": "regression",
    "random_state": 0,
    "verbosity": -1,
}

UNDERFIT_PARAMS_LGBM = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "reg_alpha": 2,
    "reg_lambda": 2,
    "min_child_samples": 20,
    "objective": "regression",
    "random_state": 5,
    "verbosity": -1,
}

BALANCED_PARAMS_LGBM = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 50,
    "objective": "regression",
    "random_state": 15,
    "verbosity": -1,
}

OVERFIT_PARAMS_XGB = {
    "n_estimators": 800,
    "max_depth": 10,
    "learning_rate": 0.01,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.001,
    "reg_lambda": 0.001,
    "min_child_weight": 5,
    "gamma": 0.001,
    "objective": "reg:squarederror",
    "random_state": 81,
}

UNDERFIT_PARAMS_XGB = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "reg_alpha": 2,
    "reg_lambda": 2,
    "min_child_weight": 10,
    "gamma": 0.001,
    "objective": "reg:squarederror",
    "random_state": 27,
}

BALANCED_PARAMS_XGB = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.01,
    "reg_lambda": 0.01,
    "min_child_weight": 7,
    "gamma": 0.001,
    "objective": "reg:squarederror",
    "random_state": 9,
}


def preprocess(data: pd.DataFrame):
    add_engineered_features(data)
    data = select_features(data, good_cols_manual)

    return data


def data_setup(path_build, path_submit, split_val):
    data = get_data(path_build)
    data = preprocess(data)

    submit_data = get_data(path_submit)
    submit_data = preprocess(submit_data)

    X, y = data.drop(columns="label"), data[["label"]]

    if not split_val:
        return {
            "train": {"X": X, "y": y},
            "submit": {"X": submit_data.drop(columns="label")},
        }

    all_idx = np.arange(len(y))
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8, random_state=4
    )

    return {
        "train": {"X": X.iloc[train_idx], "y": y.iloc[train_idx]},
        "val": {"X": X.iloc[val_idx], "y": y.iloc[val_idx]},
        "submit": {"X": submit_data.drop(columns="label")},
    }


def xgb_oub_spec(data):
    print("Starting XGB Overfit-Underfit-Balanced spec")
    print("-" * 60)
    models = {
        "overfit": XGBoost(params=OVERFIT_PARAMS_XGB),
        "underfit": XGBoost(params=UNDERFIT_PARAMS_XGB),
        "balanced": XGBoost(params=BALANCED_PARAMS_XGB),
    }
    model_weights = {
        "overfit": 0.25,
        "underfit": 0.25,
        "balanced": 0.5,
    }

    y_pred = np.zeros(len(data["submit"]["X"]))

    for name, model in models.items():
        print(f"Model: {name} XGB")
        model.train_model(data)
        y_pred += model_weights[name] * model.get_y_pred(data, "submit")

    print("-" * 60)
    print("Finished XGB Overfit-Underfit-Balanced spec")
    return y_pred


def lgbm_oub_spec(data):
    print("Starting LGBM Overfit-Underfit-Balanced spec")
    print("-" * 60)
    models = {
        "overfit": LightGBM(params=OVERFIT_PARAMS_LGBM),
        "underfit": LightGBM(params=UNDERFIT_PARAMS_LGBM),
        "balanced": LightGBM(params=BALANCED_PARAMS_LGBM),
    }
    model_weights = {
        "overfit": 0.25,
        "underfit": 0.25,
        "balanced": 0.5,
    }

    y_pred = np.zeros(len(data["submit"]["X"]))

    for name, model in models.items():
        print(f"Model: {name} LGBM")
        model.train_model(data)
        y_pred += model_weights[name] * model.get_y_pred(data, "submit")

    print("-" * 60)
    print("Finished LGBM Overfit-Underfit-Balanced spec")
    return y_pred


def ridge_stack_lgbm_oub():
    data = data_setup("data/train.parquet", "data/test.parquet", True)
    val_preds = dict()
    models = {
        "overfit": LightGBM(params=OVERFIT_PARAMS_LGBM),
        "underfit": LightGBM(params=UNDERFIT_PARAMS_LGBM),
        "balanced": LightGBM(params=BALANCED_PARAMS_LGBM),
    }
    for name, model in models.items():
        model.train_model(data)
        val_preds[name] = model.get_y_pred(data, "val")

    y_pred_stack = np.column_stack([val_pred for val_pred in val_preds.values()])

    ridge = Ridge(fit_intercept=False, alpha=100_000)

    ridge.fit(y_pred_stack, data["val"]["y"])
    weights = dict()
    for name, coef in zip(val_preds.keys(), ridge.coef_):
        print(f"{name}: {coef:.4f}")
        weights[name] = coef

    weight_sum = sum(weights.values())
    for name in weights:
        weights[name] = weights[name] / weight_sum

    print(f"Weights: {weights}")


def lgbm_base_spec(data):
    print("Starting LGBM base spec")
    print("-" * 60)

    model = LightGBM(params=BALANCED_PARAMS_LGBM)
    model.train_model(data)

    print("-" * 60)
    print("Finished LGBM base spec")
    return model.get_y_pred(data, "submit")


def xgb_base_spec(data):
    print("Starting XGB base spec")
    print("-" * 60)

    model = XGBoost(params=BALANCED_PARAMS_XGB)
    model.train_model(data)

    print("-" * 60)
    print("Finished XGB base spec")
    return model.get_y_pred(data, "submit")


def lr_base_spec(data):
    print("Starting Linear Regression base spec")
    print("-" * 60)

    model = LR()
    model.fit_model(data)

    print("-" * 60)
    print("Finished Linear Regression base spec")
    return model.get_y_pred(data, "submit")


def lr_xgb_lgbm_ensemble_spec(data):
    print("Ensembling LR, XGB OUB, LGBM OUB...")
    print("=" * 60)

    y_pred_lgbm = lgbm_oub_spec(data)
    y_pred_xgb = xgb_oub_spec(data)
    y_pred_lr = lr_base_spec(data)

    y_pred = 0.5 * y_pred_lgbm + 0.2 * y_pred_xgb + 0.3 * y_pred_lr

    print("=" * 60)
    print("Ensemble complete")

    return y_pred


def data_setup_tabm(path_build, path_submit, split_val):
    data = get_data(path_build)
    data = preprocess(data)

    submit_data = get_data(path_submit)
    submit_data = preprocess(submit_data)

    X, y = data.drop(columns="label"), data[["label"]]

    if not split_val:
        data_to_return = {
            "train": {"X": X, "y": y},
            "submit": {"X": submit_data.drop(columns="label")},
        }
    else:
        all_idx = np.arange(len(y))
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            all_idx, train_size=0.8, random_state=4
        )

        data_to_return = {
            "train": {"X": X.iloc[train_idx], "y": y.iloc[train_idx]},
            "val": {"X": X.iloc[val_idx], "y": y.iloc[val_idx]},
            "submit": {"X": submit_data.drop(columns="label")},
        }

    return convert_data_to_tensor(data_to_return)


def tabm_base_spec(data):
    print("Starting TabM base spec")
    print("-" * 60)

    model = ModelTabM(n_features=data["train"]["X"].shape[1], embedding=True)
    model.train_model(data)

    print("-" * 60)
    print("Finished TabM base spec")
    return model.get_y_pred(data, "submit")


def diff_sample_training_tabm(model_spec):
    data_for_sample = {
        "40": data_setup_tabm(
            "data/train_reduced_40.parquet", "data/test.parquet", True
        ),
        "70": data_setup_tabm(
            "data/train_reduced_70.parquet", "data/test.parquet", True
        ),
        "full": data_setup_tabm("data/train.parquet", "data/test.parquet", True),
    }
    weight_per_sample = {
        "40": 0.33,
        "70": 0.33,
        "full": 0.34,
    }

    y_pred = np.zeros(len(data_for_sample["full"]["submit"]["X"]))
    for sample_name, data in data_for_sample.items():
        print(f"Sample: {sample_name}")
        print("=" * 60)
        y_pred += weight_per_sample[sample_name] * model_spec(data)

    return pd.DataFrame({"ID": np.arange(1, len(y_pred) + 1), "prediction": y_pred})


def diff_sample_training(model_spec):
    data_for_sample = {
        "40": data_setup("data/train_reduced_40.parquet", "data/test.parquet", False),
        "70": data_setup("data/train_reduced_70.parquet", "data/test.parquet", False),
        "full": data_setup("data/train.parquet", "data/test.parquet", False),
    }
    weight_per_sample = {
        "40": 0.33,
        "70": 0.33,
        "full": 0.34,
    }

    y_pred = np.zeros(len(data_for_sample["full"]["submit"]["X"]))
    for sample_name, data in data_for_sample.items():
        print(f"Sample: {sample_name}")
        print("=" * 60)
        y_pred += weight_per_sample[sample_name] * model_spec(data)

    return pd.DataFrame({"ID": np.arange(1, len(y_pred) + 1), "prediction": y_pred})
