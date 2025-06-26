from utils.data_preprocessor import DataPreprocessor
from models.xgboost_model import XGBoost
from models.lightgbm_model import LightGBM
from models.mlp_model import MLP
from models.ae_mlp_model import AE_MLP
from models.huber_regression_model import HuberRegression
from utils.bayesian_optimization import BayesianOptimization
from sklearn.linear_model import Ridge


def main():
    # XGBoost.run("data/train_reduced_5.parquet")
    # MLP.run("data/train_reduced_5.parquet")
    # AE_MLP.run("data/train_reduced_5.parquet")
    # HuberRegression.run("data/train_reduced_5.parquet")
    LightGBM.run("data/train_reduced_5.parquet")
    # BayesianOptimization().run_lightgbm_tuning()
    # data = DataPreprocessor().get_preprocessed_data(
    #     split_val=True, return_as_tensor=True, device_to_save_tensor=None
    # )
    # print(data["train"]["y"])
    # print(data["train"]["X"])


if __name__ == "__main__":
    main()
