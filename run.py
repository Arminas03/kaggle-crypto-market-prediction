from models.xgboost_model import XGBoost
from utils.data_preprocessor import DataPreprocessor
from models.mlp_model import MLP


def main():
    XGBoost.run_xgboost("data/train.parquet")
    # MLP.run_mlp("data/train_reduced_5.parquet")
    # data = DataPreprocessor().get_preprocessed_data(
    #     split_val=True, return_as_tensor=True, device_to_save_tensor=None
    # )
    # print(data["train"]["y"])
    # print(data["train"]["X"])


if __name__ == "__main__":
    main()
