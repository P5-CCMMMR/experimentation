import torch
from .metric import DeterministicMetric

class MAXE(DeterministicMetric):
    def get_key():
        return "maxe"

    def get_title():
        return "MAXE Loss: " 

    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Maximum Absolute Error
        """
        return torch.max(torch.abs(y_hat - y))
    
class NMAXE(MAXE):
    def get_key():
        return "nmaxe"

    def get_title():
        return "NMAXE Loss: " 

    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Maximum Absolute Error
        """
        # Add small term to avoid divison by zero
        eps = torch.tensor(1e-16)
        range = self.max - self.min
        denominator = max(eps, range)
   
        return super().calc(y_hat, y) / denominator