import pandas as pd
from datetime import datetime
from src.data_preprocess.data_splitter import DataSplitter 

class DayDataSplitter(DataSplitter):
    def __init__(self, df: pd.DataFrame, time_col_name: str, power_col_name: str):
        super().__init__(df, time_col_name, power_col_name)

    def generate_train_data(self):
        return self.__get_days_of_split(self.df, 0, self.train_days - 1, self.split_days)
    
    def generate_val_data(self):
        return self.__get_days_of_split(self.df, self.train_days, self.train_days + self.val_days - 1, self.split_days)
    
    def generate_test_data(self):
        return self.__get_days_of_split(self.df, self.train_days + self.val_days,  self.split_days, self.split_days)

    def __get_days_of_split(self, df, start_day, end_day, split_days):
        if end_day > self.split_days:
             raise ValueError("Days parameter larger than the day interval")
        temp_df = df.copy()
        global_day = 0
        last_day = None

        filtered_indices = []
        for idx, row in temp_df.iterrows():
            date = pd.to_datetime(row[0]).date()
            if date != last_day:
                last_day = date
                global_day += 1
            
                if global_day == split_days:
                    global_day = 0
            if global_day >= start_day and global_day <= end_day:
                filtered_indices.append(idx)
        
        last_df = temp_df.loc[filtered_indices]

        return last_df
   