import optuna
import sklearn.metrics

from models.xgboost_model import XGBoost
from utils.data_preprocessor import DataPreprocessor


class BayesianOptimization:
    def _suggest_params(self, trial: optuna.trial.Trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.05),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 0.5),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "objective": "reg:squarederror",
            "random_state": 0,
        }

    def tune_xgboost(self, trial: optuna.trial.Trial, data, y_rescale_factor):
        params = self._suggest_params(trial)
        model = XGBoost(params)

        y_pred = model.fit(
            data["train"]["X"], data["train"]["y"], verbose=False
        ).predict(data["val"]["X"])

        return (
            sklearn.metrics.mean_squared_error(y_pred, data["val"]["y"])
            * y_rescale_factor
        )

    @staticmethod
    def run_xgboost_tuning():
        data_preprocessor = DataPreprocessor()
        data = data_preprocessor.get_preprocessed_data()
        y_rescale_factor = data_preprocessor.get_y_std()

        bayesian_optimization = BayesianOptimization()
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: bayesian_optimization.tune_xgboost(
                trial, data, y_rescale_factor
            ),
            n_trials=500,
        )

        print("Best trial:")
        print(study.best_trial.value)
        print(study.best_trial.params)
