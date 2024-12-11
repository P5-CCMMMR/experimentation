from abc import ABC, abstractmethod
from enum import Enum

import torch

class Metric(ABC):
    def __init__(self, df):
        self.max = df.max()
        self.min = df.min()

    @abstractmethod
    def get_key():
        pass

    @abstractmethod
    def is_probabilistic():
        pass

    @abstractmethod
    def is_deterministic():
        pass

    @abstractmethod
    def get_title():
        pass
    
class ProbabilisticMetric(Metric):
    @abstractmethod
    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def is_probabilistic():
        return True

    def is_deterministic():
        return False

class DeterministicMetric(Metric):
    @abstractmethod
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def is_probabilistic():
        return False

    def is_deterministic():
        return True
