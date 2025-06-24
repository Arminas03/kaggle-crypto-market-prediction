import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional
import delu
import sklearn.metrics
import matplotlib.pyplot as plt

from utils.data_preprocessor import DataPreprocessor


delu.random.seed(0)


class MLP(nn.Module):
    def __init__(self, input_dim, hyperparams=None, device=None):
        super().__init__()
        self.hyperparams = {
            "learning_rate": 0.0003,
            "n_layers": 3,
            "layer_neurons": 2,
            "batch_size": 128,
            "n_epochs": 20,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        self.model = self._setup_mlp(input_dim)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hyperparams["learning_rate"]
        )
        self.loss_fn = torch.nn.functional.mse_loss

    def _setup_mlp(self, input_dim):
        layers = []

        for _ in range(self.hyperparams["n_layers"]):
            layers.append(nn.Linear(input_dim, self.hyperparams["layer_neurons"]))
            layers.append(nn.LeakyReLU())
            input_dim = self.hyperparams["layer_neurons"]

        layers.append(nn.Linear(input_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _get_dataloader(self, data, split, batch_size):
        return DataLoader(
            TensorDataset(data[split]["X"], data[split]["y"]),
            batch_size=batch_size,
            shuffle=True,
        )

    def train_model(self, data):
        self.train()

        train_dataloader = self._get_dataloader(
            data, "train", self.hyperparams["batch_size"]
        )

        for X_batch, y_batch in train_dataloader:
            self.optimizer.zero_grad()
            loss = self.loss_fn(self(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()

    def run_learning(self, data, y_rescale_factor=1, validate=True, no_print=False):
        # early_stopping = delu.tools.EarlyStopping(20, mode="min")

        for epoch in range(1, self.hyperparams["n_epochs"] + 1):
            if validate:
                val_loss = self.train_val(data)
                if not no_print:
                    print(f"epoch: {epoch}, val_loss = {val_loss*y_rescale_factor:.4f}")
            if not validate:
                self.train_model(data)

            # early_stopping.update(val_loss)
            # if early_stopping.should_stop():
            #     break

    @torch.no_grad
    def _evaluate(self, data, split, plot=False):
        self.eval()

        dataloader = self._get_dataloader(data, split, 256)

        y_pred = (
            torch.cat([self(X_batch).squeeze(-1) for X_batch, _ in dataloader])
            .cpu()
            .numpy()
        )
        y_true = data[split]["y"].cpu().numpy()

        if plot:
            plt.plot(y_true, label="True values", color="blue")
            plt.plot(y_pred, label="Predicted values", color="orange")
            plt.xlabel("Sample index")
            plt.ylabel("Target value")
            plt.title("True vs Predicted values (Line Plot)")
            plt.legend()
            plt.show()

        return sklearn.metrics.mean_squared_error(y_true, y_pred)

    def train_val(self, data):
        self.train_model(data)

        return self._evaluate(data, "val")

    def test(self, data, y_scale_factor=1):
        test_loss = self._evaluate(data, "test", True)

        print("-" * 40)
        print(f"test loss: {test_loss * y_scale_factor:.4f}")

        return test_loss

    @staticmethod
    def run(path_to_data_file, device=None):
        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_data_file)
        data = data_preprocessor.get_preprocessed_data(split_val=True)
        data_preprocessor.ha_cluster_data(data, 6)
        data_preprocessor.transform_data_to_tensor(data, device)
        y_rescale_factor = data_preprocessor.get_y_std()

        model = MLP(data["train"]["X"].shape[1]).to(device)

        model.run_learning(data, y_rescale_factor)
        model.test(data, y_rescale_factor)
