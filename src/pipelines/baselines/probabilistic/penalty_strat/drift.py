import numpy as np
from .strat import Strat

class Drift(Strat):
    @staticmethod
    def calc(h: int, T: int) -> float:
        return 0 if T < 1 else np.sqrt(h * (1 + h/(T - 1)))
    