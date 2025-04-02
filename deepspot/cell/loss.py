import torch

def loss_cosine_function(y, y_hat):
    loss_cosine = (1 - torch.nn.functional.cosine_similarity(y, y_hat, dim=0)).mean()
    return loss_cosine

def loss_pearson_function(y, y_hat, eps=1e-8):
    """
    Compute the Pearson correlation loss between true values and predicted values
    for each feature.
    
    Args:
        y (torch.Tensor): True values of shape (N, D), where N is the number of samples and D is the number of features.
        y_hat (torch.Tensor): Predicted values of shape (N, D).
        eps (float): Small value to avoid division by zero.
        
    Returns:
        torch.Tensor: The mean Pearson correlation loss across all features.
    """
    # Centering the predictions and targets
    y_pred_centered = y_hat - torch.mean(y_hat, dim=0)
    y_true_centered = y - torch.mean(y, dim=0)

    # Calculating the covariance for each feature
    covariance = torch.sum(y_pred_centered * y_true_centered, dim=0)

    # Calculating the standard deviations for each feature
    std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2, dim=0) + eps)
    std_true = torch.sqrt(torch.sum(y_true_centered ** 2, dim=0) + eps)

    # Calculating Pearson correlation coefficient for each feature
    pearson_corr = covariance / (std_pred * std_true)

    # Return mean negative Pearson correlation as loss
    return torch.mean(1 - pearson_corr)  # Minimize loss by maximizing 

def loss_mse_function(y, y_hat):
    loss_mse = torch.nn.functional.mse_loss(y, y_hat)
    return loss_mse

def loss_l1_function(y, y_hat):
    """Calculates the L1 Loss between y and y_hat."""
    loss_l1 = torch.nn.functional.l1_loss(y, y_hat)
    return loss_l1

def loss_huber_function(y, y_hat, delta=1.0):
    """Calculates the Huber Loss between y and y_hat.

    Args:
        y (torch.Tensor): Ground truth values.
        y_hat (torch.Tensor): Predicted values.
        delta (float): The threshold at which to change between L1 and L2 loss.

    Returns:
        torch.Tensor: The calculated Huber loss.
    """
    huber_loss = torch.nn.functional.huber_loss(y, y_hat, delta=delta)
    return huber_loss

def loss_poisson_function(y, y_hat):
    """Custom loss function for Poisson model."""
    y = torch.exp(y)
    loss_poisson = torch.nn.functional.poisson_nll_loss(y, y_hat)
    return loss_poisson


def loss_mse_pearson_function(y, y_hat, mse_w=0.1):
    loss_pearson = loss_cosine_function(y, y_hat)
    loss_mse = loss_mse_function(y, y_hat)
    loss = mse_w * loss_pearson + loss_mse
    return loss


def loss_mse_cosine_function(y, y_hat):
    loss_cosine = loss_cosine_function(y, y_hat)
    loss_mse = loss_mse_function(y, y_hat)
    loss = loss_cosine + loss_mse
    return loss


def loss_l1_cosine_function(y, y_hat):
    loss_cosine = loss_cosine_function(y, y_hat)
    loss_l1 = loss_l1_function(y, y_hat)
    loss = loss_cosine + loss_l1
    return loss

def loss_huber_cosine_function(y, y_hat):
    loss_cosine = loss_cosine_function(y, y_hat)
    loss_huber = loss_huber_function(y, y_hat)
    loss = loss_cosine + loss_huber
    return loss