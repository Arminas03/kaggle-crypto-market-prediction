import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import torch
from rtdl_revisiting_models import FTTransformer
import delu
from tqdm.std import tqdm
import math
from typing import Dict
import gc

from utils.data_preprocessor import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
delu.random.seed(0)


def get_data_splits():
    data = get_data()

    X = data.drop("label", axis=1).to_numpy()
    y = data[["label"]].to_numpy()

    global n_cont_features
    n_cont_features = X.shape[1]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    all_idx = np.arange(len(y))

    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=0.8
    )

    return {
        "train": {"X": X[train_idx], "y": y[train_idx]},
        "val": {"X": X[val_idx], "y": y[val_idx]},
        "test": {"X": X[test_idx], "y": y[test_idx]},
    }


def preprocess_X(data_numpy):
    return sklearn.preprocessing.StandardScaler().fit(data_numpy["train"]["X"])


def preprocess_y(data_numpy):
    return sklearn.preprocessing.StandardScaler().fit(data_numpy["train"]["y"])


def get_preprocessed_data():
    data_numpy = get_data_splits()

    X_preprocessing = preprocess_X(data_numpy)
    for split in data_numpy:
        data_numpy[split]["X"] = X_preprocessing.transform(data_numpy[split]["X"])

    y_preprocessing = preprocess_y(data_numpy)
    for split in data_numpy:
        data_numpy[split]["y"] = y_preprocessing.transform(data_numpy[split]["y"])

    global y_std
    y_std = y_preprocessing.scale_[0]

    data = {
        split: {
            k: torch.as_tensor(v, device=device) for k, v in data_numpy[split].items()
        }
        for split in data_numpy
    }

    del data_numpy
    gc.collect()

    for split in data:
        data[split]["y"] = data[split]["y"].float()

    return data


def get_ft_transformer_model(data):
    return FTTransformer(
        n_cont_features=data["train"]["X"].shape[1],
        cat_cardinalities=0,
        d_out=1,
        **FTTransformer.get_default_kwargs(),
    ).to(device)


def train_epoch(model, optimiser, loss_fn, data, epoch_size, batch_size, epoch):
    for batch in tqdm(
        delu.iter_batches(data["train"], batch_size, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size,
    ):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(apply_model(model, batch), batch["y"])
        loss.backward()
        optimiser.step()


def apply_model(model: FTTransformer, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(batch["X"], batch.get("x_cat")).squeeze(-1)


@torch.no_grad()
def evaluate(model, data, split: str) -> float:
    model.eval()
    eval_batch_size = 8096

    y_pred = (
        torch.cat(
            [
                apply_model(batch)
                for batch in delu.iter_batches(data[split], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[split]["y"].cpu().numpy()

    return sklearn.metrics.mean_squared_error(y_true, y_pred) * y_std


def ft_main():
    data = get_preprocessed_data()
    print("data preprocessed")
    print("-" * 88 + "\n")

    model = get_ft_transformer_model(data)
    optimiser = model.make_default_optimizer()
    loss_fn = torch.nn.functional.mse_loss

    n_epochs = 50
    patience = 10
    batch_size = 16
    epoch_size = math.ceil(data["train"]["X"].shape[0] / batch_size)

    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="min")
    best = {"val": -math.inf, "test": -math.inf, "epoch": -1}

    print(f"Device: {device.type.upper()}")
    print("-" * 88 + "\n")
    timer.run()
    for epoch in range(n_epochs):
        train_epoch(model, optimiser, loss_fn, data, epoch_size, batch_size, epoch)

        val_score = evaluate("val")
        test_score = evaluate("test")
        print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

        early_stopping.update(val_score)
        if early_stopping.should_stop():
            break

        if val_score > best["val"]:
            print("🌸 New best epoch! 🌸")
            best = {"val": val_score, "test": test_score, "epoch": epoch}
        print()

    print("\n\nResult:")
    print(best)


if __name__ == "__main__":
    ft_main()
