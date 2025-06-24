from utils.data_preprocessor import DataPreprocessor
from models.xgboost_model import XGBoost
from models.mlp_model import MLP
from models.linear_regression_model import HuberRegression


def main():
    # XGBoost.run("data/train_reduced_5.parquet")
    MLP.run("data/train_reduced_5.parquet")
    # HuberRegression.run("data/train_reduced_5.parquet")
    # data = DataPreprocessor().get_preprocessed_data(
    #     split_val=True, return_as_tensor=True, device_to_save_tensor=None
    # )
    # print(data["train"]["y"])
    # print(data["train"]["X"])


if __name__ == "__main__":
    main()
