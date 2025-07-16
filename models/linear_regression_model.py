from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import matplotlib.pyplot as plt
import delu


from utils.data_preprocessor import DataPreprocessor


class LR:
    def __init__(self, hyperparams=None):
        self.hyperparams = {
            "fit_intercept": True,
        }

        if hyperparams:
            self.hyperparams.update(hyperparams)

        self.model = LinearRegression(**self.hyperparams)

    def fit_model(self, data, no_print=False):
        if not no_print:
            print("fitting...")
        timer = delu.tools.Timer()
        timer.run()

        self.model.fit(data["train"]["X"], data["train"]["y"].squeeze())

        if not no_print:
            print(f"fitting time: {timer}")

    def get_y_pred(self, data, split):
        return self.model.predict(data[split]["X"])

    def test(self, data, split="test", y_scale_factor=1, no_print=False):
        y_pred = self.get_y_pred(data, split)

        test_loss = (
            sklearn.metrics.mean_squared_error(y_pred, data[split]["y"])
            * y_scale_factor
        )

        if not no_print:
            print(f"{split} loss: {test_loss:.4f}")
            print("-" * 40)

        # plt.plot(data[split]["y"].values.ravel(), label="True values", color="blue")
        # plt.plot(y_pred, label="Predicted values", color="orange")
        # plt.xlabel("Sample index")
        # plt.ylabel("Target value")
        # plt.title("True vs Predicted values (Line Plot)")
        # plt.legend()
        # plt.show()

        return test_loss

    @staticmethod
    def run(path_to_data_file):
        model = LinearRegression()

        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_data_file)
        data = data_preprocessor.get_preprocessed_data(split_val=False)
        y_rescale_factor = data_preprocessor.get_y_std() ** 2

        model.fit_model(data)
        model.test(data, "test", y_rescale_factor)
