import pandas as pd
from sklearn.linear_model import Ridge

from utils.data_preprocessor import DataPreprocessor
from models.xgboost_model import XGBoost
from models.lightgbm_model import LightGBM
from models.mlp_model import MLP
from models.ae_mlp_model import AE_MLP
from models.linear_regression_model import LR
from utils.bayesian_optimization import BayesianOptimization
from model_runners.model_run_specs import *


def combine_preds_lgbm_xgb():
    paths = [
        "submission_lgbm_full.csv",
        "submission_lgbm_70.csv",
        "submission_lgbm_40.csv",
        "submission_xgb_full.csv",
        "submission_xgb_70.csv",
        "submission_xgb_40.csv",
    ]
    weights = [0.14, 0.13, 0.13, 0.2, 0.2, 0.2]
    preds = [pd.read_csv(path) for path in paths]

    final_pred = sum([weights[i] * preds[i]["prediction"] for i in range(len(preds))])

    pd.DataFrame({"ID": preds[0]["ID"], "prediction": final_pred}).to_csv(
        "submission.csv", index=False
    )


def combine_preds_lgbm():
    paths = [
        "submission_lgbm_full.csv",
        "submission_lgbm_70.csv",
        "submission_lgbm_40.csv",
    ]
    weights = [0.34, 0.33, 0.33]
    preds = [pd.read_csv(path) for path in paths]

    final_pred = sum([weights[i] * preds[i]["prediction"] for i in range(len(preds))])

    pd.DataFrame({"ID": preds[0]["ID"], "prediction": final_pred}).to_csv(
        "submission.csv", index=False
    )


def main():
    # XGBoost.run("data/train_reduced_20.parquet")
    # MLP.run("data/train_reduced_20.parquet")
    # AE_MLP.run("data/train_reduced_5.parquet")
    # HuberRegression.run("data/train_reduced_5.parquet")
    # LightGBM.run("data/train.parquet")
    # BayesianOptimization().run_lightgbm_tuning()
    # data = DataPreprocessor().get_preprocessed_data(
    #     split_val=True, return_as_tensor=True, device_to_save_tensor=None
    # )
    # print(data["train"]["y"])
    # print(data["train"]["X"])

    # LightGBM.run_submit("data/train_reduced_40.parquet", "submission_lgbm_40.csv")
    # LightGBM.run_submit("data/train.parquet", "submission_lgbm_full.csv")
    # XGBoost.run_submit("data/train.parquet", "submission_xgb_full.csv")
    # for i in [40, 70]:
    #     XGBoost.run_submit(f"data/train_reduced_{i}.parquet", f"submission_xgb_{i}.csv")
    #     LightGBM.run_submit(
    #         f"data/train_reduced_{i}.parquet", f"submission_lgbm_{i}.csv"
    #     )

    # LightGBM.run_submit("data/train.parquet", "submission_lgbm_full.csv")

    # for i in [40, 70]:
    #     LightGBM.run_submit(
    #         f"data/train_reduced_{i}.parquet", f"submission_lgbm_{i}.csv"
    #     )

    diff_sample_training_tabm(tabm_base_spec).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
