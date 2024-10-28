from normalizer import Normalizer
import pandas as pd

class MinMaxNormalizer(Normalizer):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.min_vals = None
        self.max_vals = None
    
    def normalize(self):
        data = self.df
        self.min_vals = data.min(axis=0)
        self.max_vals = data.max(axis=0)
        normalized_data = (data - self.min_vals) / (self.max_vals - self.min_vals)
        return normalized_data, self.min_vals, self.max_vals
    
    
    def denormalize(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df
        denormalized_data = data * (self.max_vals - self.min_vals) + self.min_vals
        return denormalized_data
