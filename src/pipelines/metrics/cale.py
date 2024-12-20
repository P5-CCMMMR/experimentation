import torch
from .metric import ProbabilisticMetric

class CALE(ProbabilisticMetric):
    @staticmethod
    def get_key():
        return "cale"
    
    @staticmethod
    def get_title():
        return "CALE: "

    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calibration Error (CALE)\\
        Returns the error of the calibration of the model\\
        Based on: https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf
        """
        within_1_std = (y > mean - stddev) & (y < mean + stddev)
        within_2_std = (y > mean - 2 * stddev) & (y < mean + 2 * stddev)
        within_3_std = (y > mean - 3 * stddev) & (y < mean + 3 * stddev)

        stddev_counts = torch.tensor([
            within_1_std.sum().item(),
            within_2_std.sum().item(),
            within_3_std.sum().item()
        ], dtype=torch.float32)
                        
        size = mean.size(0)
        
        predicted_within_std = stddev_counts / size
        
        # Using three sigma rule
        expected_within_std = torch.tensor([0.6827, 0.9545, 0.9973], dtype=torch.float32)
        
        errors = torch.pow(torch.abs(predicted_within_std - expected_within_std), 2)

        return errors.mean()
    