import pandas as pd
from datetime import datetime

class DataSplitter():
    def __init__(self, df: pd.DataFrame, time_col_name: str, power_col_name: str):
        self.df = df
        self.split_days = 0
        self.time_col_name = time_col_name
        self.power_col_name = power_col_name
        self.train = None
        self.val = None
        self.test = None

    def set_data_split(self, train_days, val_days, test_days):
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.split_days = self.train_days + self.val_days + self.test_days
    
    def generate_train_data(self):
        raise RuntimeError("Non implemented method")
    
    def generate_val_data(self):
        raise RuntimeError("Non implemented method")
    
    def generate_test_data(self):        
        raise RuntimeError("Non implemented method")
    
    def get_lt_power(self, power_watt: int, consecutive_points: int = 1):
        temp_df = self.df.copy()
        consec_rows = []
        new_df = pd.DataFrame()

        for _, row in temp_df.iterrows():
            if row[self.power_col_name] < power_watt:
                if consec_rows and consec_rows[len(consec_rows) - 1][self.time_col_name] - row[self.time_col_name] > self.average_time_diff:
                    consec_rows = []
                consec_rows.append(row)
            else:
                consec_rows = []

            if len(consec_rows) >= consecutive_points:
                for row in consec_rows:
                    new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
                consec_rows = []
        
        return new_df
    
    def get_mt_power(self, power_watt: int, consecutive_points: int = 0):
        temp_df = self.df.copy()
        consec_rows = []
        new_df = pd.DataFrame()

        for _, row in temp_df.iterrows():
            if row[self.power_col_name] > power_watt:
                if consec_rows and consec_rows[len(consec_rows) - 1][self.time_col_name] - row[self.time_col_name] > self.average_time_diff:
                    consec_rows = []
                consec_rows.append(row)
            else:
                consec_rows = []

            if len(consec_rows) >= consecutive_points:
                for row in consec_rows:
                    new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
                consec_rows = []
        
        return new_df