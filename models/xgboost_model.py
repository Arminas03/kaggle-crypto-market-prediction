import xgboost
import delu
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from utils.data_preprocessor import DataPreprocessor


class XGBoost:
    def __init__(self, params=None):
        self.params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mae",
            "random_state": 0,
        }
        if params:
            self.params.update(params)

        self.setup_model()

    def setup_model(self):
        self.model = xgboost.XGBRegressor(**self.params)

    def train_model(self, data):
        print("training...")
        timer = delu.tools.Timer()
        timer.run()

        self.model.fit(data["train"]["X"], data["train"]["y"])

        print(f"train time: {timer}")

    def train_val(self, data, y_scale_factor=1):
        self.train_model(data)

        y_pred_val = self.model.predict(data["val"]["X"])
        val_loss = mean_squared_error(y_pred_val, data["val"]["y"]) * y_scale_factor
        corr = pearsonr(y_pred_val.ravel(), data["val"]["y"].ravel()).statistic

        print(f"val loss: {val_loss:.4f}")
        print(f"Pearson corr: {corr}")
        print("-" * 40)

        return val_loss

    def test(self, data, y_scale_factor=1):
        y_pred = self.model.predict(data["test"]["X"])
        test_loss = mean_squared_error(y_pred, data["test"]["y"]) * y_scale_factor
        corr = pearsonr(y_pred.ravel(), data["test"]["y"].ravel()).statistic

        print(f"test loss: {test_loss:.4f}")
        print(f"Pearson corr: {corr}")
        print("-" * 40)

        return test_loss

    @staticmethod
    def run_xgboost(path_to_data_file):
        model = XGBoost()

        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_data_file)
        data = data_preprocessor.get_preprocessed_data()
        y_rescale_factor = data_preprocessor.get_y_std()

        model.train_val(data, y_rescale_factor)

        data = data_preprocessor.get_preprocessed_data(split_val=False)
        y_rescale_factor = data_preprocessor.get_y_std()

        model.train_model(data)
        model.test(data, y_rescale_factor)
