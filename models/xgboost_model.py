import xgboost
import delu
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


class XGBoostModel:
    def __init__(self, params=None):
        self.params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mae",
        }
        if params:
            self.params.update(params)

        self.setup_model()

    def setup_model(self):
        self.model = xgboost.XGBRegressor(**self.params, random_state=0)

    def train(self, data):
        print("training...")
        timer = delu.tools.Timer()
        timer.run()

        self.model.fit(data["train"]["X"], data["train"]["y"])

        print(f"train time: {timer}")

    def train_val(self, data, rescale_factor=1):
        self.train(data)

        y_pred_val = self.model.predict(data["val"]["X"])
        val_loss = mean_squared_error(y_pred_val, data["val"]["y"]) * rescale_factor
        corr = pearsonr(y_pred_val.ravel(), data["val"]["y"].ravel()).statistic

        print(f"val loss: {val_loss:.4f}")
        print(f"Pearson corr: {corr}")
        print("-" * 40)

        return val_loss

    def test(self, data, rescale_factor=1):
        y_pred = self.model.predict(data["test"]["X"])
        test_loss = mean_squared_error(y_pred, data["test"]["y"]) * rescale_factor
        corr = pearsonr(y_pred.ravel(), data["test"]["y"].ravel()).statistic

        print(f"test loss: {test_loss:.4f}")
        print(f"Pearson corr: {corr}")
        print("-" * 40)

        return test_loss
