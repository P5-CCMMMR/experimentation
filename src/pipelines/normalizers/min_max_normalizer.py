from .normalizer import Normalizer
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
        return normalized_data
    
    def denormalize(self, df: pd.DataFrame, target_column = None) -> pd.DataFrame:
        if target_column is None:
            return df * (self.max_vals - self.min_vals) + self.min_vals
        return df * (self.max_vals[target_column] - self.min_vals[target_column]) + self.min_vals[target_column]
    
    def get_min_vals(self):
        return self.min_vals
    
    def get_max_vals(self):
        return self.max_vals
