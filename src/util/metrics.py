import torch
import torch.nn as nn
from scipy.stats import norm

def RMSE(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error
    """
    # Add small term to avoid divison by zero
    eps = 1e-16

    return torch.sqrt(nn.functional.mse_loss(y_hat, y) + eps)
    
def NRMSE(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Normalized Root Mean Squared Error
    """
    if y.max() == y.min():
        return torch.zeros_like(y)
    return RMSE(y_hat, y) / (y.max() - y.min())

def NLL(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Negative Log Likelihood
    Returns the sum of the NLL of the batch\n
    """
    # Small term to avoid division by zero
    eps = 1e-16
    
    stddev = torch.clamp(stddev, min=eps)
    variance = torch.pow(stddev, 2)
    
    nll = 0.5 * (torch.log(2 * torch.pi * variance) + ((y - mean) ** 2) / variance)
    return torch.sum(nll)

def MNLL(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean Negative Log Likelihood
    """
    return NLL(mean, stddev, y) / y.size(0)

def NMNLL(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Normalized Mean Negative Log Likelihood
    """
    if y.max() == y.min():
        return torch.zeros_like(y)
    return MNLL(mean, stddev, y) / (y.max() - y.min())

def KL(mean1: torch.Tensor, stddev1: torch.Tensor, mean2: torch.Tensor, stddev2: torch.Tensor) -> torch.Tensor:
    """
    Kullback-Leibler Divergence\n
    Returns the sum of the KL of the batch\n
    Based of:
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    # Small term to avoid division by zero
    eps = 1e-16
    
    stddev1 = torch.clamp(stddev1, min=eps)
    stddev2 = torch.clamp(stddev2, min=eps)
    
    variance1 = torch.pow(stddev1, 2)
    variance2 = torch.pow(stddev2, 2)
    
    kl = torch.log(stddev2 / stddev1) + (variance1**2 + (mean1 - mean2)**2) / (2 * variance2) - 0.5
    return torch.sum(kl)

def MKL(mean1: torch.Tensor, stddev1: torch.Tensor, mean2: torch.Tensor, stddev2: torch.Tensor) -> torch.Tensor:
    """
    Mean Kullback-Leibler Divergence
    """
    return KL(mean1, stddev1, mean2, stddev2) / mean1.size(0)

def NMKL(mean1: torch.Tensor, stddev1: torch.Tensor, mean2: torch.Tensor, stddev2: torch.Tensor) -> torch.Tensor:
    """
    Normalized Mean Kullback-Leibler Divergence
    """
    if mean1.max() == mean1.min():
        return torch.zeros_like(mean1)
    return MKL(mean1, stddev1, mean2, stddev2) / (mean1.max() - mean1.min())

def CRPS(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Continuous Ranked Probability Score\\
    Returns the sum of the CRPS of the batch\n
    Based of: https://towardsdatascience.com/crps-a-scoring-function-for-bayesian-machine-learning-models-dd55a7a337a8
    """
    # Small term to avoid division by zero
    eps = 1e-16
    
    stddev = torch.clamp(stddev, min=eps)
    omega = (y - mean) / stddev
    crps = torch.sigmoid(omega * (2 * norm.cdf(omega) - 1) + 2 * norm.pdf(omega) - torch.pi**-0.5)
    
    return torch.sum(crps)

def MCRPS(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean Continuous Ranked Probability Score
    """
    return CRPS(mean, stddev, y) / mean.size(0)

def NMCRPS(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Normalized Mean Continuous Ranked Probability Score
    """
    if y.max() == y.min():
        return torch.zeros_like(y)
    return MCRPS(mean, stddev, y) / (y.max() - y.min())
