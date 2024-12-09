import torch
from .metric import DeterministicMetric
import torch.nn as nn

class MAE(DeterministicMetric):
    @staticmethod
    def get_key():
        return "mae"

    @staticmethod
    def get_title():
        return "MAE Loss: " 

    @staticmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean Absolute Error
        """
        return nn.functional.mse_loss(y_hat, y)
    
class NMAE(MAE):
    @staticmethod
    def get_key():
        return "nmae"

    @staticmethod
    def get_title():
        return "NMAE Loss: " 

    @staticmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Mean Absolute Error
        """
        # Add small term to avoid divison by zero
        eps = torch.tensor(1e-16)
        range = y.max() - y.min()
        denominator = max(eps, range)
   
        return MAE.calc(y_hat, y) / denominator