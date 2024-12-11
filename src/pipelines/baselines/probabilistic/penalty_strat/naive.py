import numpy as np
from .strat import Strat

class Naive(Strat):
    @staticmethod
    def calc(h: int, T: int) -> float:
        return np.sqrt(h)
    