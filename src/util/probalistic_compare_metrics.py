import torch
import torch.nn as nn
from scipy.stats import norm

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
