from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


from utils.data_preprocessor import DataPreprocessor


class HuberRegression:
    def __init__(self, hyperparams=None):
        self.hyperparams = {
            "epsilon": 1.345,
            "max_iter": 1000,
            "alpha": 0.0001,
            "fit_intercept": True,
        }

        if hyperparams:
            self.hyperparams.update(hyperparams)

        self.model = HuberRegressor(**self.hyperparams)

    def fit_model(self, data):
        self.model.fit(data["train"]["X"], data["train"]["y"].squeeze())

    def test(self, data, y_scale_factor=1, no_print=False):
        y_pred = self.model.predict(data["test"]["X"])

        test_loss = mean_squared_error(y_pred, data["test"]["y"]) * y_scale_factor

        if not no_print:
            print(f"test loss: {test_loss:.4f}")
            print("-" * 40)

        plt.plot(data["test"]["y"].values.ravel(), label="True values", color="blue")
        plt.plot(y_pred, label="Predicted values", color="orange")
        plt.xlabel("Sample index")
        plt.ylabel("Target value")
        plt.title("True vs Predicted values (Line Plot)")
        plt.legend()
        plt.show()

        return test_loss

    @staticmethod
    def run(path_to_data_file):
        model = HuberRegression()
        data_preprocessor = DataPreprocessor()

        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_data_file)
        data = data_preprocessor.get_preprocessed_data(split_val=False)
        data_preprocessor.ha_cluster_data(data, 6)
        y_rescale_factor = data_preprocessor.get_y_std()

        model.fit_model(data)
        model.test(data, y_rescale_factor)
