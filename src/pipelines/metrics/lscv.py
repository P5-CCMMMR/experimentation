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
        Based of: https://en.wikipedia.org/wiki/Scoring_rule#Logarithmic_score_for_continuous_variables
        """
        y = y.cpu()
        mean = mean.cpu()
        stddev = stddev.cpu()
        return torch.sum(-torch.log(torch.tensor(norm.pdf(y, mean, stddev))))
    

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