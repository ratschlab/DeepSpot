import torch


def loss_cosine_function(y, y_hat):
    loss_cosine = (1 - torch.nn.functional.cosine_similarity(y, y_hat, dim=0)).mean()
    return loss_cosine


def loss_pearson_function(y, y_hat):

    loss_pearson = torch.nn.functional.cosine_similarity(y - y.mean(dim=1,
                                                                    keepdim=True),
                                                         y_hat - y_hat.mean(dim=1, keepdim=True))

    loss_pearson = (1 - loss_pearson).mean()
    return loss_pearson


def loss_mse_function(y, y_hat):
    loss_mse = torch.nn.functional.mse_loss(y, y_hat)
    return loss_mse


def loss_poisson_function(y, y_hat):
    """Custom loss function for Poisson model."""
    y = torch.exp(y)
    loss_poisson = torch.nn.functional.poisson_nll_loss(y, y_hat)
    return loss_poisson


def loss_mse_pearson_function(y, y_hat, mse_w=0.2):
    loss_pearson = loss_cosine_function(y, y_hat)
    loss_mse = loss_mse_function(y, y_hat)
    loss = loss_pearson + mse_w * loss_mse
    return loss


def loss_mse_cosine_function(y, y_hat):
    loss_cosine = loss_cosine_function(y, y_hat)
    loss_mse = loss_mse_function(y, y_hat)
    loss = loss_cosine + loss_mse
    return loss
