import optuna

from models.xgboost_model import XGBoost
from models.mlp_model import MLP
from models.huber_regression_model import HuberRegression
from models.lightgbm_model import LightGBM
from utils.data_preprocessor import DataPreprocessor


class BayesianOptimization:
    def _suggest_xgboost_params(self, trial: optuna.trial.Trial):
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
        }

    def _tune_xgboost(self, trial: optuna.trial.Trial, data, y_rescale_factor):
        params = self._suggest_xgboost_params(trial)
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

    def _suggest_mlp_params(self, trial: optuna.trial.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "layer_neurons": trial.suggest_categorical(
                "layer_neurons", [2, 4, 8, 16, 32, 64, 128, 256, 512]
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256, 512, 1024]
            ),
            "n_epochs": trial.suggest_int("n_epochs", 20, 150),
        }

    def _tune_mlp(self, trial: optuna.trial.Trial, data, y_rescale_factor):
        params = self._suggest_mlp_params(trial)
        model = MLP(data["train"]["X"].shape[1], params)

        model.run_learning(data, validate=False)
        y_val_loss = model._evaluate(data, "val") * y_rescale_factor

        return y_val_loss

    def run_mlp_tuning(self):
        data_preprocessor = DataPreprocessor()
        data = data_preprocessor.get_preprocessed_data(split_val=True)
        data_preprocessor.ha_cluster_data(data, 6)
        data_preprocessor.transform_data_to_tensor(data)
        y_rescale_factor = data_preprocessor.get_y_std()

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._tune_mlp(trial, data, y_rescale_factor),
            n_trials=200,
            n_jobs=2,
        )

        print("Best trial:")
        print(study.best_trial.value)
        print(study.best_trial.params)

    def _suggest_huber_params(self, trial: optuna.trial.Trial):
        return {
            "epsilon": trial.suggest_float("epsilon", 1, 3),
            "max_iter": trial.suggest_int("max_iter", 500, 2500),
            "alpha": trial.suggest_float("alpha", 1e-6, 1),
        }

    def _tune_huber(self, trial: optuna.trial.Trial, data, y_rescale_factor):
        params = self._suggest_huber_params(trial)
        model = HuberRegression(params)

        model.fit_model(data)
        y_val_loss = model.test(data, "val", y_rescale_factor, True)

        return y_val_loss

    def run_huber_regression_tuning(self):
        data_preprocessor = DataPreprocessor()
        data = data_preprocessor.get_preprocessed_data(split_val=True)
        y_rescale_factor = data_preprocessor.get_y_std()

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._tune_huber(trial, data, y_rescale_factor),
            n_trials=200,
            n_jobs=2,
        )

        print("Best trial:")
        print(study.best_trial.value)
        print(study.best_trial.params)

    def _suggest_lightgbm_params(self, trial: optuna.trial.Trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.5),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 60),
            "verbosity": -1,
            "force_col_wise": True,
        }

    def _tune_lightgbm(self, trial: optuna.trial.Trial, data, y_rescale_factor):
        params = self._suggest_lightgbm_params(trial)
        model = LightGBM(params)

        y_val_loss = model.train_val(data, no_print=True)

        return y_val_loss * y_rescale_factor

    def run_lightgbm_tuning(self):
        data_preprocessor = DataPreprocessor()
        data = data_preprocessor.get_preprocessed_data()
        y_rescale_factor = data_preprocessor.get_y_std()

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._tune_lightgbm(trial, data, y_rescale_factor),
            n_trials=500,
            n_jobs=2,
        )

        print("Best trial:")
        print(study.best_trial.value)
        print(study.best_trial.params)
