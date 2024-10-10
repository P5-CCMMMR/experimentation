import pandas as pd
from src.data_preprocess.data_splitter import DataSplitter 

class TvtDataSplitter(DataSplitter):
    def __init__(self, df: pd.DataFrame, time_col_name: str, power_col_name: str):
        super().__init__(df,time_col_name,power_col_name)

    def generate_train_data(self):
        train_data_len = int(len(self.df) * (self.train_days / self.split_days))
        return self.df[0:train_data_len]
    
    def generate_val_data(self):
        train_data_len = int(len(self.df) * (self.train_days / self.split_days))
        val_data_len = int(len(self.df) * (self.val_days / self.split_days))
        return self.df[train_data_len: train_data_len + val_data_len]
    
    def generate_test_data(self):
        train_data_len = int(len(self.df) * (self.train_days / self.split_days))
        val_data_len = int(len(self.df) * (self.val_days / self.split_days))
        return self.df[train_data_len + val_data_len:]