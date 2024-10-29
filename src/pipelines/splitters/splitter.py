import pandas as pd
from abc import ABC, abstractmethod

class Splitter(ABC):
    def __init__(self, train_split, val_split, test_split):
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.total_split = test_split + val_split + train_split

    @abstractmethod
    def get_train(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_val(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_test(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
