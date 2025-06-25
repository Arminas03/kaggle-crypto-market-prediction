import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import sklearn.metrics
import matplotlib.pyplot as plt

from utils.data_preprocessor import DataPreprocessor


class AE_MLP(nn.Module):
    def __init__(self, input_dim, hyperparams=None):
        super().__init__()

        self._set_hyperparams(hyperparams)

        self.encoder = self._setup_encoder(input_dim)
        self.decoder = self._setup_decoder(input_dim)
        self.mlp = self._setup_mlp()

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hyperparams["general"]["learning_rate"]
        )
        self.loss_fn = torch.nn.functional.mse_loss

    def _set_hyperparams(self, hyperparams):
        self.hyperparams = {
            "general": {
                "latent_dim": 6,
                "batch_size": 128,
                "n_epochs": 20,
                "learning_rate": 1e-4,
                "loss_weight_ae": 0.2,
            },
            "encoder": {"n_layers": 2, "layer_neurons": 4},
            "decoder": {"n_layers": 2, "layer_neurons": 4},
            "mlp": {"n_layers": 2, "layer_neurons": 4},
        }

        if hyperparams:
            self.hyperparams.update(hyperparams)

    def _setup_encoder(self, input_dim):
        layers = []
        curr_dim = input_dim

        for _ in range(self.hyperparams["encoder"]["n_layers"]):
            layers.append(
                nn.Linear(curr_dim, self.hyperparams["encoder"]["layer_neurons"])
            )
            layers.append(nn.SiLU())
            curr_dim = self.hyperparams["encoder"]["layer_neurons"]

        layers.append(nn.Linear(curr_dim, self.hyperparams["general"]["latent_dim"]))

        return nn.Sequential(*layers)

    def _setup_decoder(self, input_dim):
        layers = []
        curr_dim = self.hyperparams["general"]["latent_dim"]

        for _ in range(self.hyperparams["decoder"]["n_layers"]):
            layers.append(
                nn.Linear(curr_dim, self.hyperparams["decoder"]["layer_neurons"])
            )
            layers.append(nn.SiLU())
            curr_dim = self.hyperparams["decoder"]["layer_neurons"]

        layers.append(nn.Linear(curr_dim, input_dim))

        return nn.Sequential(*layers)

    def _setup_mlp(self):
        layers = []
        curr_dim = self.hyperparams["general"]["latent_dim"]

        for _ in range(self.hyperparams["mlp"]["n_layers"]):
            layers.append(nn.Linear(curr_dim, self.hyperparams["mlp"]["layer_neurons"]))
            layers.append(nn.SiLU())
            curr_dim = self.hyperparams["mlp"]["layer_neurons"]

        layers.append(nn.Linear(curr_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_pred = self.mlp(z)

        return y_pred, x_hat

    def _get_dataloader(self, data, split, batch_size):
        return DataLoader(
            TensorDataset(data[split]["X"], data[split]["y"]),
            batch_size=batch_size,
            shuffle=True,
        )

    def train_model(self, data):
        self.train()

        train_dataloader = self._get_dataloader(
            data, "train", self.hyperparams["general"]["batch_size"]
        )

        for X_batch, y_batch in train_dataloader:
            self.optimizer.zero_grad()
            y_pred, x_hat = self(X_batch)

            loss_mlp = self.loss_fn(y_pred, y_batch)
            loss_ae = self.loss_fn(x_hat, X_batch)
            loss = loss_mlp + self.hyperparams["general"]["loss_weight_ae"] * loss_ae
            loss.backward()

            self.optimizer.step()

    def run_learning(self, data, y_rescale_factor=1, validate=True, no_print=False):
        for epoch in range(1, self.hyperparams["general"]["n_epochs"] + 1):
            if validate:
                val_loss = self.train_val(data)
                if not no_print:
                    print(f"epoch: {epoch}, val_loss = {val_loss*y_rescale_factor:.4f}")
            if not validate:
                self.train_model(data)

    @torch.no_grad
    def _evaluate(self, data, split, plot=False):
        self.eval()

        dataloader = self._get_dataloader(data, split, 256)

        y_pred = (
            torch.cat([self(X_batch)[0].squeeze(-1) for X_batch, _ in dataloader])
            .cpu()
            .numpy()
        )
        y_true = data[split]["y"].cpu().numpy()

        if plot:
            plt.plot(y_true, label="True values", color="blue")
            plt.plot(y_pred, label="Predicted values", color="orange")
            plt.xlabel("Index")
            plt.ylabel("Value")
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
        data_preprocessor.transform_data_to_tensor(data, device)
        y_rescale_factor = data_preprocessor.get_y_std()

        model = AE_MLP(data["train"]["X"].shape[1]).to(device)

        model.run_learning(data, y_rescale_factor)
        model.test(data, y_rescale_factor)
