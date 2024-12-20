import torch
from .metric import ProbabilisticMetric
import torch.nn as nn
from scipy.stats import norm

class TLSCV(ProbabilisticMetric):
    @staticmethod
    def get_key():
        return "lscv"
    
    @staticmethod
    def get_title():
        return "Total LSCV: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Total Logarithmic Score for Continuous Variables\n
        Returns sum of the logarithmic score for batch\n
        Based of: https://en.wikipedia.org/wiki/Scoring_rule#Logarithmic_score_for_continuous_variables
        """

        eps = 1e-16

        mean_np = mean.cpu().numpy()
        stddev_np = stddev.cpu().numpy()
        y_np = y.cpu().numpy()
        
        pdf_vals = norm.pdf(y_np, mean_np, stddev_np)
        pdf_tensor = torch.clamp(torch.tensor(pdf_vals, dtype=torch.float32, device=y.device), min=eps)
        
        return torch.sum(-torch.log(pdf_tensor))
    
class MLSCV(TLSCV):
    @staticmethod
    def get_key():
        return "mlscv"

    @staticmethod
    def get_title():
        return "MLSCV: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean Logarithmic Score for Continuous Variables
        """
        return super().calc(mean, stddev, y) / y.size(0)
    
class NMLSCV(MLSCV):
    @staticmethod
    def get_key():
        return "nmlscv"

    @staticmethod
    def get_title():
        return "NMLSCV: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Mean Logarithmic Score for Continuous Variables
        """
        # Add small term to avoid divison by zero
        eps = torch.tensor(1e-16)
        range = self.max - self.min
        denominator = max(eps, range)
        
        return super().calc(mean, stddev, y) / denominator
    
class DMRLSCV(TLSCV):
    @staticmethod
    def get_key():
        return "dmrlscv"
    
    @staticmethod
    def get_title():
        return "DMRLSCV: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Denormalized Mean Logarithmic Score for Continuous Variables
        """
        mean = self._denormalize_temp(mean)
        stddev = self._denormalize_stddev(stddev)
        y = self._denormalize_temp(y)
        return super().calc(mean, stddev, y)