import torch
from .metric import ProbabilisticMetric
import torch.nn as nn
from scipy.stats import norm

# ! need change have tensor to numpy error
class LSCV(ProbabilisticMetric):
    @staticmethod
    def get_title():
        return "LSCV Loss: "

    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic Score for Continuous Variables\n
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
    

class MLSCV(LSCV):
    @staticmethod
    def get_title():
        return "MLSCV Loss: "

    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean Logarithmic Score for Continuous Variables
        """
        return LSCV.calc(mean, stddev, y) / y.size(0)
    
    
class NMLSCV(MLSCV):
    @staticmethod
    def get_title():
        return "NMLSCV Loss: "
    
    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Mean Logarithmic Score for Continuous Variables
        """
        if y.max() == y.min():
            return torch.zeros_like(y)
        return MLSCV.calc(mean, stddev, y) / (y.max() - y.min())