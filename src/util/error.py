import torch
import torch.nn as nn

def RMSE(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error
    """
    # Add small term to avoid divison by zero
    epsilon = 1e-8

    return torch.sqrt(nn.functional.mse_loss(y_hat, y) + epsilon)
    
def NRMSE(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Normalized Root Mean Squared Error
    """
    return RMSE(y_hat, y) / (y.max() - y.min())

def NLL(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Negative Log Likelihood
    """
    # Small term to avoid division by zero
    eps = 1e-8
    
    stddev = torch.clamp(stddev, min=eps)
    variance = torch.pow(stddev, 2)
    return torch.sum(0.5 * (torch.log(2 * torch.pi * variance) + ((y - mean) ** 2) / variance))

def MNLL(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean Negative Log Likelihood
    """
    return NLL(mean, stddev, y) / y.size(0)
