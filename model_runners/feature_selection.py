import pandas as pd
import numpy as np
import lightgbm
import xgboost
import shap
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from utils.data_preprocessing import *


def get_shap_selection_df() -> pd.DataFrame:
    data = data_setup("data/train.parquet", "data/test.parquet", False)
    model = lightgbm.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=13,
        verbosity=-1,
    )

    model.fit(data["train"]["X"], data["train"]["y"])

    shap_values = shap.TreeExplainer(model)(data["train"]["X"])

    return pd.DataFrame(
        {
            "feature": data["train"]["X"].columns,
            "shap_importance": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("shap_importance", ascending=False)


def get_mutual_info_selection_df() -> pd.DataFrame:
    data = data_setup("data/train.parquet", "data/test.parquet", False)

    selector = SelectKBest(score_func=mutual_info_regression, k=500)
    selector.fit(data["train"]["X"], data["train"]["y"])

    return pd.DataFrame(
        {
            "feature": data["train"]["X"].columns,
            "mutual_info_score": selector.scores_,
        }
    ).sort_values("mutual_info", ascending=False)


def get_xgb_selection_df() -> pd.DataFrame:
    data = data_setup("data/train.parquet", "data/test.parquet", False)

    model = xgboost.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=19,
        verbosity=0,
    )

    model.fit(data["train"]["X"], data["train"]["y"])

    xgb_importances = pd.Series(
        model.get_booster().get_score(importance_type="gain")
    ).reindex(data["train"]["X"].columns, fill_value=0)

    return pd.DataFrame(
        {
            "feature": xgb_importances.index,
            "xgb_importance": xgb_importances.values,
        }
    ).sort_values("xgb_importance", ascending=False)
