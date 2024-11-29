import pandas as pd
from abc import ABC, abstractmethod

class Normalizer(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def denormalize(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
