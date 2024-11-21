import torch
from .metric import DeterministicMetric
import torch.nn as nn

class RMSE(DeterministicMetric):
    @staticmethod
    def get_title():
        return "RMSE Loss: " 

    @staticmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Root Mean Squared Error
        """
        # Add small term to avoid divison by zero
        eps = 1e-16
    
        return torch.sqrt(nn.functional.mse_loss(y_hat, y) + eps)
    
class NRMSE(RMSE):
    @staticmethod
    def get_title():
        return "NRMSE Loss: " 

    @staticmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error
        """
        if y.max() == y.min():
            return torch.tensor(0.0)
        return RMSE.calc(y_hat, y) / (y.max() - y.min())