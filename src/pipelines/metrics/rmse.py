import torch
from .metric import DeterministicMetric
import torch.nn as nn

class RMSE(DeterministicMetric):
    def get_key():
        return "rmse"

    def get_title():
        return "RMSE Loss: " 

    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Root Mean Squared Error
        """
        # Add small term to avoid divison by zero
        eps = 1e-16
    
        return torch.sqrt(nn.functional.mse_loss(y_hat, y) + eps)
    
class NRMSE(RMSE):
    def get_key():
        return "nrmse"

    def get_title():
        return "NRMSE Loss: " 

    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error
        """
        # Add small term to avoid divison by zero
        eps = torch.tensor(1e-16)
        range = self.max - self.min
        denominator = max(eps, range)
   
        return super().calc(y_hat, y) / denominator