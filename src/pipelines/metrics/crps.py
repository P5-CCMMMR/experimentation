import torch
from .metric import ProbabilisticMetric
from scipy.stats import norm

class TCRPS(ProbabilisticMetric):
    def get_key():
        return "tcrps"

    def get_title():
        return "TCRPS: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Total Continuous Ranked Probability Score\\
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
    

class MCRPS(TCRPS):
    def get_key():
        return "mcrps"

    def get_title():
        return "MCRPS: "
    
    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Mean Continuous Ranked Probability Score
        """
        return super().calc(mean, stddev, y) / mean.size(0)
    
class NMCRPS(TCRPS):
    def get_key():
        return "nmcrps"

    def get_title():
        return "NMCRPS: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalized Mean Continuous Ranked Probability Score
        """
        # Add small term to avoid divison by zero
        eps = torch.tensor(1e-16)
        range = self.max - self.min
        denominator = max(eps, range)
        
        return super().calc(mean, stddev, y) / denominator