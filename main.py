from model_runners.model_run_specs import *
from model_runners.feature_selection import *


def main():
    diff_sample_training(lr_xgb_lgbm_ensemble_spec).to_csv(
        "submission.csv", index=False
    )


if __name__ == "__main__":
    main()
