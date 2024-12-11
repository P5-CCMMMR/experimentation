from abc import ABC, abstractmethod
from enum import Enum

import torch

class Metric(ABC):
    @staticmethod
    @abstractmethod
    def is_probabilistic():
        pass

    @staticmethod 
    @abstractmethod
    def is_deterministic():
        pass
    
    @staticmethod
    @abstractmethod
    def get_key():
        pass

    @staticmethod 
    @abstractmethod
    def get_title():
        pass
    
class ProbabilisticMetric(Metric):
    @staticmethod
    @abstractmethod
    def calc(mean: torch.Tensor, stddev: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def is_probabilistic():
        return True

    @staticmethod 
    def is_deterministic():
        return False

class DeterministicMetric(Metric):
    @staticmethod
    @abstractmethod
    def calc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def is_probabilistic():
        return False

    @staticmethod 
    def is_deterministic():
        return True
