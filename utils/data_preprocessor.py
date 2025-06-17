import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import numpy as np
import torch


class DataPreprocessor:
    def __init__(self, path_to_data_file="data/train_reduced.parquet"):
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

    def _get_data_splits(self, return_type, split_val):
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

        if return_type == "np":
            for split in splits:
                splits[split]["X"] = splits[split]["X"].to_numpy().astype(np.float32)
                splits[split]["y"] = splits[split]["y"].to_numpy().astype(np.float32)

        return splits

    def _preprocess_X(self, data_numpy):
        return sklearn.preprocessing.StandardScaler().fit(data_numpy["train"]["X"])

    def _preprocess_y(self, data_numpy):
        return sklearn.preprocessing.StandardScaler().fit(data_numpy["train"]["y"])

    def get_y_std(self):
        return self.y_std

    def get_preprocessed_data(
        self,
        split_val=True,
        return_as_tensor=False,
        device_to_save_tensor=None,
    ):
        data = self._get_data_splits(
            return_type="np" if return_as_tensor else "pd", split_val=split_val
        )

        X_preprocessing = self._preprocess_X(data)
        for split in data:
            data[split]["X"] = X_preprocessing.transform(data[split]["X"])

        y_preprocessing = self._preprocess_y(data)
        for split in data:
            data[split]["y"] = y_preprocessing.transform(data[split]["y"])
        self.y_std = y_preprocessing.scale_[0]

        if return_as_tensor:
            data = {
                split: {
                    k: torch.as_tensor(v, device=device_to_save_tensor)
                    for k, v in data[split].items()
                }
                for split in data
            }

            for split in data:
                data[split]["y"] = data[split]["y"].float()

        print("preprocessing finished")
        print("-" * 40)
        return data
