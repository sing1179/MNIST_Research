
# Install uv
!curl -LsSf https://astral.sh/uv/install.sh | sh

# Install packages using uv
!uv pip install torch torchvision torchaudio pytorch-lightning matplotlib tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import random

pl.seed_everything(42)

# FFN_GeGLU implemented with einsum
class FFN_GeGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.W_gate = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.02)

    def forward(self, x):
        x_proj = torch.einsum('bi,ih->bh', x, self.W_in)
        gate_proj = F.gelu(torch.einsum('bi,ih->bh', x, self.W_gate))
        return torch.einsum('bh,ho->bo', x_proj * gate_proj, self.W_out)

# FFN_ReLU implemented with einsum
class FFN_ReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.02)
        self.W_out = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.02)

    def forward(self, x):
        return torch.einsum('bh,ho->bo', F.relu(torch.einsum('bi,ih->bh', x, self.W_in)), self.W_out)

# Lightning module wrapper
class MNIST_FFN(pl.LightningModule):
    def __init__(self, ffn_type="GeGLU", hidden_dim=16, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        input_dim = 28 * 28
        output_dim = 10
        if ffn_type == "GeGLU":
            self.ffn = FFN_GeGLU(input_dim, hidden_dim, output_dim)
        elif ffn_type == "ReLU":
            self.ffn = FFN_ReLU(input_dim, hidden_dim, output_dim)
        else:
            raise ValueError("Invalid FFN type")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.ffn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=False)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# Data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train_full = datasets.MNIST(root=".", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_len = int(0.8 * len(mnist_train_full))
val_len = len(mnist_train_full) - train_len
mnist_train, mnist_val = random_split(mnist_train_full, [train_len, val_len])

def random_search(k, ffn_type, hidden_dims, batch_sizes, lrs, bootstrap_iters=50):
    results = {}
    quiet_trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_progress_bar=False)

    for hd in hidden_dims:
        best_acc = -1
        best_config = None
        best_model = None

        for _ in range(k):
            bs = random.choice(batch_sizes)
            lr = random.choice(lrs)

            model = MNIST_FFN(ffn_type=ffn_type, hidden_dim=hd, lr=lr)
            trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=1,
                                 logger=False, enable_checkpointing=False, enable_progress_bar=False)
            train_loader = DataLoader(mnist_train, batch_size=bs, shuffle=True)
            val_loader = DataLoader(mnist_val, batch_size=bs)
            trainer.fit(model, train_loader, val_loader)

            val_acc = trainer.callback_metrics["val_acc"].item()
            if val_acc > best_acc:
                best_acc = val_acc
                best_config = (bs, lr)
                best_model = model

        # Bootstrap testing
        test_accs = []
        for _ in range(bootstrap_iters):
            sample_idx = np.random.choice(len(mnist_test), len(mnist_test), replace=True)
            sample_ds = Subset(mnist_test, sample_idx)
            acc = quiet_trainer.test(best_model, DataLoader(sample_ds, batch_size=best_config[0]), verbose=False)[0]["test_acc"]
            test_accs.append(acc)

        results[hd] = (np.mean(test_accs), np.std(test_accs))

    return results

hidden_dims = [2, 4, 8, 16]
batch_sizes = [8, 64]
lrs = [1e-1, 1e-2, 1e-3, 1e-4]

for k in [2, 4, 8]:
    geglu_results = random_search(k, "GeGLU", hidden_dims, batch_sizes, lrs)
    relu_results = random_search(k, "ReLU", hidden_dims, batch_sizes, lrs)

    geglu_acc = [geglu_results[hd][0] for hd in hidden_dims]
    geglu_err = [geglu_results[hd][1] for hd in hidden_dims]
    relu_acc = [relu_results[hd][0] for hd in hidden_dims]
    relu_err = [relu_results[hd][1] for hd in hidden_dims]

    plt.figure()
    plt.errorbar(hidden_dims, geglu_acc, yerr=geglu_err, label="GeGLU", capsize=5, marker="o")
    plt.errorbar(hidden_dims, relu_acc, yerr=relu_err, label="ReLU", capsize=5, marker="o")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test Accuracy")
    plt.title(f"MNIST FFN Comparison (k={k})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    if np.mean(geglu_acc) > np.mean(relu_acc):
        print(f"For k={k}, data agrees with claim: FFN_GeGLU > FFN_ReLU")
    else:
        print(f"For k={k}, data does NOT agree with claim.")
