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
        stddev1_count, stddev2_count, stddev3_count = 0, 0, 0
        for (m, s, pred) in zip(mean, stddev, y):
            if pred > m - 3*s and pred < m + 3*s:
                stddev3_count += 1
                if pred > m - 2*s and pred < m + 2*s:
                    stddev2_count += 1
                    if pred > m - s and pred < m + s:
                        stddev1_count += 1
                        
        size = len(mean)
        return (stddev1_count / size, stddev2_count / size, stddev3_count / size)
    