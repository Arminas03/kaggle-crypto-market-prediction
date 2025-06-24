import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import numpy as np
import torch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from collections import defaultdict


class DataPreprocessor:
    def __init__(self, path_to_data_file="data/train_reduced_5.parquet"):
        self.path_to_data_file = path_to_data_file
        self.y_std = None

    def _corr_one_columns(self):
        # fmt: off
        return [
            'X387', 'X429', 'X381', 'X423', 'X417', 'X333', 'X369', 'X327', 'X321', 'X405', 'X399',
            'X315', 'X309', 'X393', 'X140', 'X182', 'X134', 'X176', 'X170', 'X86', 'X122', 'X80',
            'X116', 'X74', 'X68', 'X110', 'X62', 'X146'
        ]
        # fmt: on

    def _fixed_columns(self, data):
        return [col for col in data.columns if data[col].nunique() == 1]

    def _get_data_from_file(self):
        data = pd.read_parquet(self.path_to_data_file)

        data.drop(columns=self._fixed_columns(data), inplace=True)
        data.drop(columns=self._corr_one_columns(), inplace=True)

        return data

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

    def get_preprocessed_data(self, split_val=True):
        data = self._get_data_splits(split_val=split_val)

        col_names_X = data["train"]["X"].columns.tolist()
        col_name_y = data["train"]["y"].columns.tolist()
        index_map = {split: data[split]["X"].index for split in data}

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
