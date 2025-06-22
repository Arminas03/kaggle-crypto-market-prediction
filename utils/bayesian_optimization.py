import optuna
import sklearn.metrics

from models.xgboost_model import XGBoost
from utils.data_preprocessor import DataPreprocessor


class BayesianOptimization:
    def _suggest_params(self, trial: optuna.trial.Trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "objective": "reg:squarederror",
            "random_state": 0,
        }

    def _tune_xgboost(self, trial: optuna.trial.Trial, data, y_rescale_factor):
        params = self._suggest_params(trial)
        model = XGBoost(params)

        y_val_loss = model.train_val(data, no_print=True)

        return y_val_loss * y_rescale_factor

    def run_xgboost_tuning(self):
        data_preprocessor = DataPreprocessor()
        data = data_preprocessor.get_preprocessed_data()
        y_rescale_factor = data_preprocessor.get_y_std()

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._tune_xgboost(trial, data, y_rescale_factor),
            n_trials=500,
            n_jobs=2,
        )

        print("Best trial:")
        print(study.best_trial.value)
        print(study.best_trial.params)
