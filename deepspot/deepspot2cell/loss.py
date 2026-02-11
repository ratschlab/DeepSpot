import torch
from torchmetrics.functional import pearson_corrcoef


def mse(pred, target):
    return torch.nn.functional.mse_loss(pred, target)


def pearsonr(pred, target, eps = 1e-8):
    y_pred_centered = pred - torch.mean(pred, dim=0)
    y_true_centered = target - torch.mean(target, dim=0)

    covariance = torch.sum(y_pred_centered * y_true_centered, dim=0)
    std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=0) + eps)
    std_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=0) + eps)

    pearson_corr = covariance / (std_pred * std_true)
    return torch.mean(1 - pearson_corr)


def pearson_mse(pred, target):
    return (pearsonr(pred, target) + mse(pred, target))


class wmse(torch.nn.Module):
    """MSE with each gene j is weighted by f(rank_j)."""
    def __init__(self, n_genes, upweight_hvg=True, power=0.1, device=None, dtype=torch.float32):
        super().__init__()

        if upweight_hvg:
            ranks = torch.arange(n_genes, 0, step=-1, device=device, dtype=dtype)
        else:
            ranks = torch.arange(1, n_genes + 1, device=device, dtype=dtype)
        weights = 1.0 / (ranks.float() ** power)
        weights = weights * n_genes / weights.sum()
        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        loss = self.weights * (pred - target) ** 2
        return loss.mean()
    

def safe_pearson(pred, target, sample_wise = False):
    """Torch-native Pearson along feature-dim (or sample-wise after transpose)."""
    if sample_wise:
        pred, target = pred.T, target.T
    r = pearson_corrcoef(pred, target)
    return torch.nan_to_num(r, nan=0.0)