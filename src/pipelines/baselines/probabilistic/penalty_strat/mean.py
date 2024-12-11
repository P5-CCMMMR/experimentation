import numpy as np
from .strat import Strat

class Mean(Strat):
    @staticmethod
    def calc(h: int, T: int) -> float:
        return 0 if T < 1 else np.sqrt(1 + (1/T))
    