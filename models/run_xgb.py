import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scripts.data import get_data


def split_data(data: pd.DataFrame):
    return train_test_split(
        data.drop("label"), data["label"], test_size=0.2, random_state=42
    )


def get_xgboost_model() -> xgboost.XGBRegressor:
    return xgboost.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mae",
    )


def main():
    x_train, x_test, y_train, y_test = split_data(get_data())

    xgb_model = get_xgboost_model().fit(x_train, y_train)

    y_pred = xgb_model.predict(x_test)

    print(mean_squared_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))


if __name__ == "__main__":
    main()
