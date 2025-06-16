from models.xgboost_model import XGBoostModel
from utils.data_preprocessor import DataPreprocessor


def main():
    model = XGBoostModel()

    data_preprocessor = DataPreprocessor(path_to_data_file="data/train_reduced.parquet")
    data = data_preprocessor.get_preprocessed_data()
    y_rescale_factor = data_preprocessor.get_y_std()

    model.train_val(data, y_rescale_factor)

    data = data_preprocessor.get_preprocessed_data(split_val=False)
    y_rescale_factor = data_preprocessor.get_y_std()

    model.train(data)
    model.test(data, y_rescale_factor)


if __name__ == "__main__":
    main()
