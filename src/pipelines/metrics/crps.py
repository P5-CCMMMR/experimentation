import torch
from .metric import ProbabilisticMetric
from scipy.stats import norm

class CRPS(ProbabilisticMetric):
    @staticmethod
    def get_title():
        return "CRPS Loss: "

    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Continuous Ranked Probability Score\\
        Returns the sum of the CRPS of the batch\n
        Based of: https://towardsdatascience.com/crps-a-scoring-function-for-bayesian-machine-learning-models-dd55a7a337a8
        """
        # Small term to avoid division by zero
        eps = 1e-16

        stddev = torch.clamp(stddev, min=eps)
        omega = (y - mean) / stddev
        omega_np = omega.cpu().numpy()
        
        cdf_vals = norm.cdf(omega_np)
        pdf_vals = norm.pdf(omega_np)
        
        cdf_tensor = torch.tensor(cdf_vals, dtype=torch.float32, device=y.device)
        pdf_tensor = torch.tensor(pdf_vals, dtype=torch.float32, device=y.device)
        
        crps = stddev * (omega * (2 * cdf_tensor - 1) + 2 * pdf_tensor - torch.pi**-0.5)

        return torch.sum(crps)
    

class MCRPS(CRPS):
    @staticmethod
    def get_title():
        return "MCRPS Loss: "

    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean Continuous Ranked Probability Score
        """
        return CRPS.calc(mean, stddev, y) / mean.size(0)
    
class NMCRPS(CRPS):
    @staticmethod
    def get_title():
        return "NMCRPS Loss: "

    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Mean Continuous Ranked Probability Score
        """
        if y.max() == y.min():
            return torch.zeros_like(y)
        return MCRPS.calc(mean, stddev, y) / (y.max() - y.min())