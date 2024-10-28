from splitter import Splitter
import pandas as pd

class StdSplitter(Splitter):
    def get_train(self, df: pd.DataFrame) -> pd.DataFrame:
        train_len = int(len(df) * (self.train_split / self.total_split))
        return df.iloc[:train_len]
    
    def get_val(self, df: pd.DataFrame) -> pd.DataFrame:
        train_len = int(len(df) * (self.train_split / self.total_split))
        val_len = int(len(df) * (self.val_split / self.total_split))
        return df[train_len: train_len + val_len]
    
    def get_test(self, df: pd.DataFrame) -> pd.DataFrame:
        train_len = int(len(df) * (self.train_split / self.total_split))
        val_len = int(len(df) * (self.val_split / self.total_split))
        return df[train_len + val_len:]
