import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import numpy as np
import torch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from collections import defaultdict

from del_col import *
from important_cols import *


class DataPreprocessor:
    def __init__(self, path_to_data_file="data/train_reduced_5.parquet"):
        self.path_to_data_file = path_to_data_file
        self.y_std = 1

    def select_400_best_cols(self, data: pd.DataFrame):
        cols = (
            ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
            + select_400_best
            + ["label"]
        )
        return data[cols]

    def add_features(self, data: pd.DataFrame):
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
        data["cross_vol_depth"] = data["bid_cross_buy"] / (
            data["ask_cross_sell"] + 1e-10
        )

        data["liquidity_ratio"] = data["depth"] / (data["volume"] + 1e-10)
        data["market_activity_ratio"] = data["volume"] / (data["depth"] + 1e-10)
        data["order_imbalance"] = -data["bid_ask_spread"] / (data["depth"] + 1e-10)
        data["kyle_lambda"] = np.abs(data["order_flow"]) / (data["volume"] + 1e-10)

    def del_99_cols(self, data: pd.DataFrame):
        data.drop(columns=high_corr_col_99, inplace=True)

        return data

    def _get_data_from_file(self):
        return pd.read_parquet(self.path_to_data_file)

    def _get_data_splits(self, split_val):
        data = self._get_data_from_file()
        X = data.drop(columns="label")
        y = data[["label"]]

        all_idx = np.arange(len(y))
        trainval_idx, test_idx = sklearn.model_selection.train_test_split(
            all_idx, train_size=0.8, random_state=0
        )
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            trainval_idx, train_size=0.8, random_state=0
        )
        if split_val:
            splits = {
                "train": {"X": X.iloc[train_idx], "y": y.iloc[train_idx]},
                "val": {"X": X.iloc[val_idx], "y": y.iloc[val_idx]},
                "test": {"X": X.iloc[test_idx], "y": y.iloc[test_idx]},
            }
        else:
            splits = {
                "train": {"X": X.iloc[trainval_idx], "y": y.iloc[trainval_idx]},
                "test": {"X": X.iloc[test_idx], "y": y.iloc[test_idx]},
            }

        return splits

    def _preprocess_X(self, data_numpy):
        return sklearn.preprocessing.StandardScaler().fit(data_numpy["train"]["X"])

    def _preprocess_y(self, data_numpy):
        return sklearn.preprocessing.StandardScaler().fit(data_numpy["train"]["y"])

    def get_y_std(self):
        return self.y_std

    def transform_data_to_tensor(self, data, device=None):
        for split in data:
            data[split]["X"] = torch.as_tensor(
                data[split]["X"].to_numpy().astype(np.float32), device=device
            )
            data[split]["y"] = torch.as_tensor(
                data[split]["y"].to_numpy().astype(np.float32), device=device
            )

    def get_preprocessed_data(self, split_val=True, standardize=True):
        data = self._get_data_splits(split_val=split_val)

        col_names_X = data["train"]["X"].columns.tolist()
        col_name_y = data["train"]["y"].columns.tolist()
        index_map = {split: data[split]["X"].index for split in data}

        if standardize:
            X_preprocessing = self._preprocess_X(data)
            for split in data:
                data[split]["X"] = X_preprocessing.transform(data[split]["X"])

            y_preprocessing = self._preprocess_y(data)
            for split in data:
                data[split]["y"] = y_preprocessing.transform(data[split]["y"])
            self.y_std = y_preprocessing.scale_[0]

            for split in data:
                data[split]["X"] = pd.DataFrame(
                    data[split]["X"], columns=col_names_X, index=index_map[split]
                )
                data[split]["y"] = pd.DataFrame(
                    data[split]["y"], columns=col_name_y, index=index_map[split]
                )

        print("preprocessing finished")
        print("-" * 40)
        return data

    def _get_cluster_feature_dict(self, linkage_matrix, n_clusters, col_names):
        cluster_feature_dict = defaultdict(list)
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

        for i in range(clusters.shape[0]):
            cluster_feature_dict[clusters[i]].append(col_names[i])

        return cluster_feature_dict

    def _transform_by_clusters(self, cluster_feature_dict, x_data):
        new_columns = []

        for cluster_idx, feature_list in cluster_feature_dict.items():
            col = x_data[feature_list].mean(axis=1)
            col.name = f"cluster_{cluster_idx}"
            new_columns.append(col)

        return pd.concat(new_columns, axis=1)

    def ha_cluster_data(
        self,
        data,
        n_clusters,
        keep_cols=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
    ):
        cluster_columns = [
            col for col in data["train"]["X"].columns if col not in keep_cols
        ]
        x_data = data["train"]["X"][cluster_columns]

        corr_matrix = np.corrcoef(x_data.values, rowvar=False)
        distance_matrix = 1 - abs(corr_matrix)
        col_names = x_data.columns

        linkage_matrix = linkage(
            squareform(distance_matrix, checks=False), method="ward"
        )

        cluster_feature_dict = self._get_cluster_feature_dict(
            linkage_matrix, n_clusters, col_names
        )

        for split in data:
            data[split]["X"] = pd.concat(
                [
                    self._transform_by_clusters(
                        cluster_feature_dict, data[split]["X"][cluster_columns]
                    ),
                    data[split]["X"][keep_cols],
                ],
                axis=1,
            )

    def select_k_best(self, data, linear, k=200):
        selector = SelectKBest(
            score_func=f_regression if linear else mutual_info_regression, k=k
        )
        selector.fit(data["train"]["X"], data["train"]["y"].squeeze())

        cols = data["train"]["X"].columns[selector.get_support()]

        for split in data:
            data[split]["X"] = pd.DataFrame(
                selector.transform(data[split]["X"]),
                columns=cols,
                index=data[split]["X"].index,
            )
