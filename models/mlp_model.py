import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional
import delu
import sklearn.metrics

from utils.data_preprocessor import DataPreprocessor


delu.random.seed(0)


class MLP(nn.Module):
    def __init__(self, input_dim, learning_rate, device=None):
        super().__init__()
        self.model = self._setup_mlp(input_dim)
        self.device = device

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.functional.mse_loss

    def _setup_mlp(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

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

        train_dataloader = self._get_dataloader(data, "train", 64)

        for X_batch, y_batch in train_dataloader:
            self.optimizer.zero_grad()
            loss = self.loss_fn(self(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad
    def _evaluate(self, data, split):
        self.eval()

        dataloader = self._get_dataloader(data, split, 128)

        y_pred = (
            torch.cat([self(X_batch).squeeze(-1) for X_batch, _ in dataloader])
            .cpu()
            .numpy()
        )
        y_true = data[split]["y"].cpu().numpy()

        return sklearn.metrics.mean_squared_error(y_true, y_pred)

    def train_val(self, data):
        self.train_model(data)

        return self._evaluate(data, "val")

    def test(self, data, y_scale_factor=1):
        test_loss = self._evaluate(data, "test")

        print("-" * 40)
        print(f"test loss: {test_loss * y_scale_factor:.4f}")

        return test_loss

    @staticmethod
    def run(path_to_data_file, n_epochs=1000, patience=20, device=None):
        data_preprocessor = DataPreprocessor(path_to_data_file=path_to_data_file)
        data = data_preprocessor.get_preprocessed_data(
            split_val=True, return_as_tensor=True, device_to_save_tensor=None
        )
        y_rescale_factor = data_preprocessor.get_y_std()

        model = MLP(data["train"]["X"].shape[1], 0.005).to(device)
        early_stopping = delu.tools.EarlyStopping(patience, mode="min")

        for epoch in range(1, n_epochs + 1):
            val_loss = model.train_val(data)
            print(f"epoch: {epoch}, val_loss = {val_loss*y_rescale_factor:.4f}")

            early_stopping.update(val_loss)
            if early_stopping.should_stop():
                break

        model.test(data, y_rescale_factor)
