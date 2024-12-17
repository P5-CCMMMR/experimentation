from abc import ABC, abstractmethod
from enum import Enum

import torch

class Metric(ABC):
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max
    
    def _denormalize_temp(self, prediction):
        return prediction * (self.max - self.min) + self.min
    
    @staticmethod
    @abstractmethod
    def get_key(self):
        pass

    @staticmethod
    @abstractmethod
    def is_probabilistic(self):
        pass

    @staticmethod
    @abstractmethod
    def is_deterministic(self):
        pass

    @staticmethod
    @abstractmethod 
    def get_title(self):
        pass
    
class ProbabilisticMetric(Metric):
    @abstractmethod
    def calc(self, mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def is_probabilistic():
        return True

    @staticmethod
    def is_deterministic():
        return False
    
    def _denormalize_stddev(self, stddev):
        return stddev * (self.max - self.min)

class DeterministicMetric(Metric):
    @abstractmethod
    def calc(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def is_probabilistic():
        return False
    
    @staticmethod
    def is_deterministic():
        return True
