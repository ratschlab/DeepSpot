import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from torch.backends import cudnn
from enum import Enum
import lightning as L
from torch import nn
import numpy as np
import random
import torch
import os


from .loss import (
    loss_cosine_function,
    loss_pearson_function,
    loss_mse_function,
    loss_poisson_function,
    loss_mse_pearson_function,
    loss_mse_cosine_function,
    loss_l1_cosine_function,
    loss_l1_function,
    loss_huber_function,
    loss_huber_cosine_function
)

from deepspot.utils.utils import fix_seed


class Operation(str, Enum):
    SUM = "sum"
    MAX = "max"
    NONE = "none"
    MEAN = "mean"


class Phi(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Rho(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepCell(L.LightningModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 loss_func: str = "mse",
                 lr: float = 1e-4,
                 p: float = 0.2,
                 p_phi: Union[None, float] = None,
                 p_rho: Union[None, float] = None,
                 n_ensemble: int = 10,
                 n_ensemble_phi: Union[None, int] = None,
                 n_ensemble_rho: Union[None, int] = None,
                 phi2rho_size: int = 1024,
                 emb_size: int = 1024,
                 weight_decay: float = 1e-6,
                 random_seed: int = 2024,
                 agg_neighbors: str = "max",
                 scaler=None,
                 cell_context: str = 'cell_neighbors'):
        super().__init__()

        self.save_hyperparameters()
        self.scaler = scaler
        fix_seed(random_seed)

        if loss_func == "mse":
            self.loss_func = loss_mse_function
        elif loss_func == "cos":
            self.loss_func = loss_cosine_function
        elif loss_func == "l1":
            self.loss_func = loss_l1_function
        elif loss_func == "huber":
            self.loss_func = loss_huber_function
        elif loss_func == "mse_cos":
            self.loss_func = loss_mse_cosine_function
        elif loss_func == "l1_cos":
            self.loss_func = loss_l1_cosine_function
        elif loss_func == "huber_cos":
            self.loss_func = loss_huber_cosine_function
        elif loss_func == "pearson":
            self.loss_func = loss_pearson_function
        elif loss_func == "mse_pearson":
            self.loss_func = loss_mse_pearson_function
        elif loss_func == "poisson":
            self.loss_func = loss_poisson_function

        if agg_neighbors == "max":
            self.agg_neighbors = Operation.MAX
        elif agg_neighbors == "mean":
            self.agg_neighbors = Operation.MEAN
        elif agg_neighbors == "sum":
            self.agg_neighbors = Operation.SUM

        self.p_phi = p_phi or p
        self.p_rho = p_rho or p
        self.n_ensemble_phi = n_ensemble_phi or n_ensemble
        self.n_ensemble_rho = n_ensemble_rho or n_ensemble

        self.training_loss = []
        self.validation_loss = []
        self.emb_size = emb_size
        self.phi2rho_size = phi2rho_size
        self.phi_cell = nn.ModuleList([Phi(input_size, phi2rho_size, self.p_phi) for _ in range(self.n_ensemble_phi)])
        self.rho = nn.ModuleList([Rho(phi2rho_size * self._get_phi_multiplier(cell_context),
                                 output_size, self.p_rho) for _ in range(self.n_ensemble_rho)])
        self._forward_fn = self._get_forward_function(cell_context)

    def _get_phi_multiplier(self, cell_context: str) -> int:
        context_multipliers = {
            'cell': 1,
            'cell_neighbors': 2,
        }
        return context_multipliers.get(cell_context, 1)

    def _get_forward_function(self, cell_context: str):
        forward_functions = {
            'cell': self._forward_cell,
            'cell_neighbors': self._forward_cell_neighbors,
        }
        return forward_functions.get(cell_context, self._forward_cell)

    def loop_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        X, y = batch

        X = [x.float() for x in X] if isinstance(X, list) else X.float()
        y = y.float()

        y_hat = self(X)
        loss = self.loss_func(y_hat, y)

        self.log(stage, loss, on_epoch=True, prog_bar=True)

        loss_score = loss.cpu().detach().item()

        if stage == "train":
            self.training_loss.append(loss_score)
        elif stage == "val":
            self.validation_loss.append(loss_score)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.loop_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.loop_step(batch, "val")

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = self._forward_fn(x)
        return x

    def inverse_transform(self, x) -> torch.Tensor:
        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)
        return x

    def _forward_cell(self, x: torch.Tensor) -> torch.Tensor:
        x_phi = self._apply_phi(x, self.phi_cell, operation=Operation.NONE)
        return self._apply_rho(x_phi)

    def _forward_cell_neighbors(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_cell, x_neighbors = x
        x_cell = self._apply_phi(x_cell, self.phi_cell)
        x_neighbors = self._apply_phi(x_neighbors, self.phi_cell, operation=self.agg_neighbors)
        x_phi = torch.cat((x_cell, x_neighbors), dim=1)
        return self._apply_rho(x_phi)

    def _apply_phi(self, x: torch.Tensor, phi_modules: nn.ModuleList,
                   operation: Operation = Operation.NONE) -> torch.Tensor:
        batch_size = x.shape[0]
        sample_size = x.shape[1]
        x = x.view(-1, self.hparams.input_size)
        x_phi = torch.stack([phi(x) for phi in phi_modules], dim=1)
        x_phi, _ = torch.median(x_phi, dim=1)

        x_phi = x_phi.view(batch_size, sample_size, -1)

        if operation == Operation.SUM:
            x_phi = x_phi.sum(dim=1)
        elif operation == Operation.MEAN:
            x_phi = x_phi.mean(dim=1)
        elif operation == Operation.MAX:
            x_phi, _ = x_phi.max(dim=1)
        elif operation == Operation.NONE:
            x_phi = x_phi.view(batch_size, -1)

        return x_phi

    def _apply_rho(self, x_phi: torch.Tensor) -> torch.Tensor:
        x_rho = torch.stack([rho(x_phi) for rho in self.rho], dim=1)
        x_rho = torch.mean(x_rho, dim=1)
        return x_rho

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}

    def smooth_curve(self, data, alpha=0.2):
        """
        Applies exponential smoothing to a data sequence.
        :param data: List or array of data points.
        :param alpha: Smoothing factor, between 0 (full smoothing) and 1 (no smoothing).
        :return: Smoothed data as a numpy array.
        """
        smoothed = []
        for i, point in enumerate(data):
            if i == 0:
                smoothed.append(point)
            else:
                smoothed.append(alpha * point + (1 - alpha) * smoothed[-1])
        return np.array(smoothed)

    def plot_loss(self):
        """
        Plots the training and validation loss.
        Assumes self.history is a dictionary containing 'loss' and 'val_loss' keys.
        If validation loss is missing, only training loss is plotted.
        """
        # Check if validation loss is present
        has_validation_loss = len(self.validation_loss) > 0

        # Compute corresponding x-axis for validation loss, if available
        total_steps = len(self.training_loss)

        if has_validation_loss:
            validation_steps = np.arange(0, len(self.validation_loss)) * \
                (len(self.training_loss) // len(self.validation_loss) + 1)
            validation_loss_full = np.interp(range(1, total_steps + 1), validation_steps, self.validation_loss)
        else:
            validation_loss_full = np.zeros(total_steps)  # If no validation loss, use a zero array

        # Smooth both curves
        smoothed_training_loss = self.smooth_curve(self.training_loss)
        smoothed_validation_loss = self.smooth_curve(
            validation_loss_full) if has_validation_loss else np.zeros(total_steps)

        # Plot original and smoothed losses
        plt.figure(figsize=(5, 4))

        # Training loss
        plt.plot(range(1, total_steps + 1), self.training_loss,
                 label='Original Training Loss', color='blue', alpha=0.4, linestyle='-')
        plt.plot(range(1, total_steps + 1), smoothed_training_loss,
                 label='Smoothed Training Loss', color='blue', linestyle='-')

        # If validation loss is available, plot it
        if has_validation_loss:
            plt.plot(range(1, total_steps + 1), validation_loss_full,
                     label='Original Validation Loss', color='orange', alpha=0.4, linestyle='--')
            plt.plot(range(1, total_steps + 1), smoothed_validation_loss,
                     label='Smoothed Validation Loss', color='orange', linestyle='--')

        # Add plot details
        plt.title('Training and Validation Loss (with Smoothing)')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
