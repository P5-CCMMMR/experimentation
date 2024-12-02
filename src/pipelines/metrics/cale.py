import torch
from .metric import ProbabilisticMetric

class CALE(ProbabilisticMetric):
    @staticmethod
    def get_title():
        return "CALE: "

    @staticmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calibration Error (CALE)\\
        Returns the percentage of predictions that fall within 1, 2 and 3 standard deviations of the mean
        """
        within_1_std = (y > mean - stddev) & (y < mean + stddev)
        within_2_std = (y > mean - 2 * stddev) & (y < mean + 2 * stddev)
        within_3_std = (y > mean - 3 * stddev) & (y < mean + 3 * stddev)

        stddev1_count = within_1_std.sum().item()
        stddev2_count = within_2_std.sum().item()
        stddev3_count = within_3_std.sum().item()
                        
        size = mean.size(0)
        
        return (stddev1_count / size, stddev2_count / size, stddev3_count / size)