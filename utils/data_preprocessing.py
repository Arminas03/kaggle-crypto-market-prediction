import pandas as pd
import numpy as np
import torch
import sklearn.model_selection

from constants import *


def get_data(path):
    return pd.read_parquet(path)


def preprocess(data: pd.DataFrame):
    add_engineered_features(data)
    data = select_features(data, selected_cols)

    return data


def data_setup(path_build, path_submit, split_val):
    data = get_data(path_build)
    data = preprocess(data)

    submit_data = get_data(path_submit)
    submit_data = preprocess(submit_data)

    X, y = data.drop(columns="label"), data[["label"]]

    if not split_val:
        return {
            "train": {"X": X, "y": y},
            "submit": {"X": submit_data.drop(columns="label")},
        }

    all_idx = np.arange(len(y))
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8, random_state=4
    )

    return {
        "train": {"X": X.iloc[train_idx], "y": y.iloc[train_idx]},
        "val": {"X": X.iloc[val_idx], "y": y.iloc[val_idx]},
        "submit": {"X": submit_data.drop(columns="label")},
    }


def add_engineered_features(data: pd.DataFrame):
    data["bid_ask_spread"] = data["ask_qty"] - data["bid_qty"]
    data["depth"] = data["bid_qty"] + data["ask_qty"]
    data["order_flow"] = data["buy_qty"] - data["sell_qty"]

    data["bid_cross_ask"] = data["bid_qty"] * data["ask_qty"]
    data["buy_cross_sell"] = data["buy_qty"] * data["sell_qty"]
    data["bid_cross_buy"] = data["bid_qty"] * data["buy_qty"]
    data["bid_cross_sell"] = data["bid_qty"] * data["sell_qty"]
    data["ask_cross_buy"] = data["ask_qty"] * data["buy_qty"]
    data["ask_cross_sell"] = data["ask_qty"] * data["sell_qty"]

    data["buy_pressure"] = data["buy_qty"] / (data["volume"] + 1e-10)
    data["sell_pressure"] = data["sell_qty"] / (data["volume"] + 1e-10)
    data["buy_sell_ratio"] = data["buy_qty"] / (data["sell_qty"] + 1e-10)
    data["vol_depth_ratio"] = data["volume"] / (data["depth"] + 1e-10)
    data["buy_sell_imbalance"] = data["order_flow"] / (data["volume"] + 1e-10)
    data["cross_vol_depth"] = data["bid_cross_buy"] / (data["ask_cross_sell"] + 1e-10)

    data["liquidity_ratio"] = data["depth"] / (data["volume"] + 1e-10)
    data["market_activity_ratio"] = data["volume"] / (data["depth"] + 1e-10)
    data["order_imbalance"] = -data["bid_ask_spread"] / (data["depth"] + 1e-10)
    data["kyle_lambda"] = np.abs(data["order_flow"]) / (data["volume"] + 1e-10)

    data["signed_flow"] = np.sign(data["buy_qty"] - data["sell_qty"])
    data["signed_quotes"] = np.sign(data["bid_qty"] - data["ask_qty"])
    data["signed_flow"] = data["signed_flow"].astype("category")
    data["signed_quotes"] = data["signed_quotes"].astype("category")

    data["log_volume"] = np.log(data["volume"] + 1e-10)
    data["log_depth"] = np.log(data["depth"] + 1e-10)
    data["bid_ask_squared"] = data["bid_ask_spread"] ** 2
    data["order_flow_squared"] = data["order_flow"] ** 2


def select_features(data, features):
    return data[features + ["label"]]


def convert_data_to_tensor(data):
    return {
        split: {
            var: torch.as_tensor(
                data[split][var].to_numpy().astype(np.float32),
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )
            for var in data[split]
        }
        for split in data
    }
