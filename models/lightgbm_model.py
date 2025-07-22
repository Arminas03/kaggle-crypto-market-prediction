import lightgbm
import delu
import sklearn.metrics
from scipy.stats import pearsonr


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
