from matplotlib import pyplot as plt
from torch.backends import cudnn
import numpy as np
import random
import torch
import os


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
