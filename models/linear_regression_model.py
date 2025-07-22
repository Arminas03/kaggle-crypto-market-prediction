from sklearn.linear_model import Ridge
import sklearn.metrics
import delu


class LR:
    def __init__(self, hyperparams=None):
        self.hyperparams = {
            "fit_intercept": True,
            "alpha": 10,
        }

        if hyperparams:
            self.hyperparams.update(hyperparams)

        self.model = Ridge(**self.hyperparams)

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

        return test_loss
