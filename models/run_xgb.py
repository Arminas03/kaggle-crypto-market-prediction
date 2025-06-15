import xgboost
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from scripts.data import get_data


def split_data(data: pd.DataFrame):
    return train_test_split(
        data.drop(columns="label"), data["label"], test_size=0.2, random_state=42
    )


def get_xgboost_model() -> xgboost.XGBRegressor:
    return xgboost.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mae",
    )


# def fit_pca(X):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     pca = PCA()
#     pca.fit(X_scaled)

#     n_components = (pca.explained_variance_ > 1).sum()

#     pca = PCA(n_components=n_components)
#     pca.fit(X_scaled)

#     return scaler, pca


# def to_df_after_pca(X, index):
#     return pd.DataFrame(X, columns=[f"pca_{i}" for i in range(X.shape[1])], index=index)


# def transform_data_with_pca(x_train, x_test):
#     norm_scaler, pca = fit_pca(x_train)

#     x_train_transformed = pca.transform(norm_scaler.transform(x_train))
#     x_test_transformed = pca.transform(norm_scaler.transform(x_test))

#     x_train_transformed = to_df_after_pca(x_train_transformed, x_train.index)
#     x_test_transformed = to_df_after_pca(x_test_transformed, x_test.index)

#     return x_train_transformed, x_test_transformed


def main():
    x_train, x_test, y_train, y_test = split_data(get_data())

    start_time = time.time()
    xgb_model = get_xgboost_model().fit(x_train, y_train)
    print(f"time fitting: {(time.time() - start_time):.2f}")
    y_pred = xgb_model.predict(x_test)

    print(mean_squared_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))

    xgboost.plot_importance(xgb_model)
    plt.show()


if __name__ == "__main__":
    main()
