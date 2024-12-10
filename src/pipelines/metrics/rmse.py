import torch
from .metric import DeterministicMetric
import torch.nn as nn
#from torchmetrics import NormalizedRootMeanSquaredError

class RMSE(DeterministicMetric):
    @staticmethod
    def get_key():
        return "rmse"

    @staticmethod
    def get_title():
        return "RMSE Loss: " 

    @staticmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Root Mean Squared Error
        Eps is there according to: https://discuss.pytorch.org/t/rmse-loss-function/16540/6
        """
        # Add small term to avoid divison by zero
        eps = 1e-16    
        return torch.sqrt(nn.functional.mse_loss(y_hat, y) + eps)
    
class NRMSE(RMSE):
    @staticmethod
    def get_key():
        return "nrmse"

    @staticmethod
    def get_title():
        return "NRMSE Loss: " 

    @staticmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error
        """
        eps = torch.tensor(1e-16)
        range = y.max() - y.min()
        denominator = max(eps, range)
        return RMSE.calc(y_hat, y) / denominator