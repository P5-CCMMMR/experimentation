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
