from matplotlib import pyplot as plt
from torch.backends import cudnn
import numpy as np
import random
import torch
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import anndata as ad
import lightning as L
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def model_fine_tune(model, dataloader, rho=False, gene_expression=True, max_epochs=10):
    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    if rho:
        # Unfreeze the 'gene_expression' layer
        for param in model.rho.parameters():
            param.requires_grad = True
        # Unfreeze the 'gene_expression' layer
    if gene_expression:
        for param in model.gene_expression.parameters():
            param.requires_grad = True

    # Step 2: Update the optimizer to include only the trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model.hparams.lr, weight_decay=model.hparams.weight_decay
    )
    model.eval()

    # Set 'rho' and 'gene_expression' layers to train mode (since they are fine-tuned)
    if rho:
        model.rho.train()
    if gene_expression:
        model.gene_expression.train()
    # Step 3: Train the model using the fine-tuning DataLoader
    trainer = L.Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)


def run_inference_from_dataloader(model, dataloader, device):
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            if type(X) is list:
                X = (x.to(device) for x in X)
            else:
                X = X.to(device)
            y = model.forward(X)
            # y_zeros = y_zeros.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.array(out)


def plot_loss_values(train_losses, val_losses=None):
    train_losses = np.array(train_losses)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    train_idx = np.arange(0, len(train_losses))
    plt.plot(train_idx, train_losses, color="b", label="train")

    if val_losses is not None:
        val_losses = np.array(val_losses)
        val_idx = np.arange(0, len(val_losses)) * (len(train_losses) // len(val_losses) + 1)
        plt.plot(val_idx, val_losses, color="r", label="val")

    plt.legend()


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
