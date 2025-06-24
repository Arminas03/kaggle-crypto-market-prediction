import lightgbm
import delu
import sklearn.metrics
from scipy.stats import pearsonr

from utils.data_preprocessor import DataPreprocessor


class LightGBM:
    def __init__(self, params=None):
        self.params = {
            "n_estimators": 1500,
            "max_depth": 10,
            "learning_rate": 0.048,
            "subsample": 0.88,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.018,
            "reg_lambda": 0.33,
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

    def train_val(self, data, y_scale_factor=1, no_print=False):
        self.train_model(data, no_print)

        y_pred_val = self.model.predict(data["val"]["X"])
        val_loss = (
            sklearn.metrics.mean_squared_error(y_pred_val, data["val"]["y"])
            * y_scale_factor
        )
        corr = pearsonr(y_pred_val.ravel(), data["val"]["y"].values.ravel()).statistic

        if not no_print:
            print(f"val loss: {val_loss:.4f}")
            print(f"Pearson corr: {corr}")
            print("-" * 40)

        return val_loss

    def test(self, data, y_scale_factor=1, no_print=False):
        y_pred = self.model.predict(data["test"]["X"])
        test_loss = (
            sklearn.metrics.mean_squared_error(y_pred, data["test"]["y"])
            * y_scale_factor
        )
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
        data = data_preprocessor.get_preprocessed_data()
        y_rescale_factor = data_preprocessor.get_y_std()

        model.train_val(data, y_rescale_factor)

        data = data_preprocessor.get_preprocessed_data(split_val=False)
        y_rescale_factor = data_preprocessor.get_y_std()

        model.train_model(data)
        model.test(data, y_rescale_factor)
