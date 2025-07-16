import torch
import torch.nn.functional as functional
import tabm
import rtdl_num_embeddings
import delu
from sklearn.metrics import mean_squared_error
from copy import deepcopy


class ModelTabM:
    def __init__(self, n_features, embedding, hyperparams={}):
        self.hyperparams = {
            "n_epochs": 1_000_000,
            "batch_size": 256,
            "max_patience": 16,
        }
        self.hyperparams.update(hyperparams)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = tabm.TabM.make(
            n_num_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            num_embeddings=(
                rtdl_num_embeddings.LinearReLUEmbeddings(n_features)
                if embedding
                else None
            ),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=2e-3, weight_decay=3e-4
        )

    def loss(self, y_pred, y_true):
        return functional.mse_loss(
            y_pred.flatten(0, 1), y_true.repeat_interleave(self.model.backbone.k)
        )

    def apply_model(self, data, split, batch):
        return self.model(data[split]["X"][batch]).squeeze(-1).float()

    @torch.no_grad()
    def get_y_pred(self, data, split, eval_batch_size=8096):
        return (
            torch.cat(
                [
                    self.apply_model(data, split, batch)
                    for batch in torch.arange(
                        len(data[split]["X"]), device=self.device
                    ).split(eval_batch_size)
                ]
            )
            .cpu()
            .numpy()
        )

    @torch.no_grad()
    def evaluate(self, data, split):
        self.model.eval()

        y_pred = self.get_y_pred(data, split).mean(1)

        return float(mean_squared_error(data[split]["y"].cpu().numpy(), y_pred))

    def make_checkpoint(self, epoch=-1, val_loss=float("inf")):
        return deepcopy(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
        )

    def train_model(self, data, no_print=False):
        if not no_print:
            print("training...")
        timer = delu.tools.Timer()
        timer.run()

        len_data = len(data)
        best_checkpoint = self.make_checkpoint()
        patience = self.hyperparams["max_patience"]

        for epoch in range(self.hyperparams["n_epochs"]):
            batches = torch.randperm(len_data, device=self.device).split(
                self.hyperparams["batch_size"]
            )

            for batch in batches:
                self.model.train()
                self.optimizer.zero_grad()

                loss = self.loss(
                    self.apply_model(data, "train", batch), data["train"]["y"][batch]
                )

                loss.backward()
                self.optimizer.step()

            val_loss = self.evaluate(data, "val")

            if val_loss < best_checkpoint["val_loss"]:
                best_checkpoint = self.make_checkpoint(epoch, val_loss)
                patience = self.hyperparams["max_patience"]

                if not no_print:
                    print(f"new best checkpoint: epoch {epoch}, val_loss {val_loss}")
            else:
                patience -= 1

            if patience < 0:
                if not no_print:
                    print("Note: patience limit reached, stopping training")
                break

            torch.cuda.empty_cache()

        self.model.load_state_dict(best_checkpoint["model"])

        if not no_print:
            print(f"train time: {timer}")
