import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import delu
import sklearn.metrics


delu.random.seed(0)


class MLP(nn.Module):
    def __init__(self, n_features, hyperparams=None, device=None):
        super().__init__()
        self.hyperparams = {
            "learning_rate": 0.003,
            "weight_decay": 0.05,
            "n_layers": 3,
            "layer_neurons": 256,
            "dropout_rate": 0.5,
            "batch_size": 1024,
            "n_epochs": 50,
            "noise": True,
            "noise_std": 0.1,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        self.model = self._setup_mlp(n_features)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )
        self.loss_fn = torch.nn.functional.mse_loss

    def _setup_mlp(self, input_dim):
        layers = []

        for _ in range(self.hyperparams["n_layers"]):
            layers.append(nn.Linear(input_dim, self.hyperparams["layer_neurons"]))
            layers.append(nn.BatchNorm1d(self.hyperparams["layer_neurons"]))
            layers.append(nn.Dropout(p=self.hyperparams["dropout_rate"]))
            layers.append(nn.LeakyReLU())
            input_dim = self.hyperparams["layer_neurons"]

        layers.append(nn.Linear(input_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _get_dataloader(self, data, split, batch_size):
        return (
            DataLoader(
                TensorDataset(data[split]["X"], data[split]["y"]),
                batch_size=batch_size,
                shuffle=True,
            )
            if split != "submit"
            else DataLoader(
                TensorDataset(data[split]["X"], data[split]["X"]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

    def train_model(self, data):
        self.train()

        train_loss = 0
        n_samples = 0
        train_dataloader = self._get_dataloader(
            data, "train", self.hyperparams["batch_size"]
        )

        for X_batch, y_batch in train_dataloader:
            if self.hyperparams["noise"]:
                X_batch += torch.randn_like(X_batch) * self.hyperparams["noise_std"]
            self.optimizer.zero_grad()
            loss = self.loss_fn(self(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

        return train_loss / n_samples

    def run_learning(self, data, y_rescale_factor=1, validate=False, no_print=False):
        if not no_print:
            print("training...")
        timer = delu.tools.Timer()
        timer.run()

        for epoch in range(1, self.hyperparams["n_epochs"] + 1):
            if validate:
                train_loss, val_loss = self.train_val(data)
                if not no_print:
                    print(
                        f"epoch: {epoch}, "
                        + f"train_loss = {train_loss*y_rescale_factor:.4f}, "
                        + f"val_loss = {val_loss*y_rescale_factor:.4f}"
                    )
            if not validate:
                train_loss = self.train_model(data)
                if not no_print and not epoch % 10:
                    print(
                        f"epoch: {epoch}, "
                        + f"train_loss = {train_loss*y_rescale_factor:.4f}, "
                    )

        if not no_print:
            print(f"train time: {timer}")

    @torch.no_grad
    def get_y_pred(self, data, split):
        dataloader = self._get_dataloader(data, split, 256)

        return (
            torch.cat([self(X_batch).squeeze(-1) for X_batch, _ in dataloader])
            .cpu()
            .numpy()
        )

    @torch.no_grad
    def _evaluate(self, data, split):
        self.eval()

        y_pred = self.get_y_pred(data, split)
        y_true = data[split]["y"].cpu().numpy()

        return sklearn.metrics.mean_squared_error(y_true, y_pred)

    def train_val(self, data):
        train_loss = self.train_model(data)

        return train_loss, self._evaluate(data, "val")

    def test(self, data, y_scale_factor=1):
        test_loss = self._evaluate(data, "test")

        print("-" * 40)
        print(f"test loss: {test_loss * y_scale_factor:.4f}")

        return test_loss
