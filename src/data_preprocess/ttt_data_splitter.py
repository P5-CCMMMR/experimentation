import pandas as pd
from datetime import datetime

class TttDataSplitter():
    def __init__(self, df: pd.DataFrame, time_col_name: str, power_col_name: str):
        self.df = df
        self.split_days = 0
        self.time_col_name = time_col_name
        self.power_col_name = power_col_name
        self.__get_average_time_diff()
        self.train = None
        self.val = None
        self.test = None

    
    def set_data_split(self, train_days, val_days, test_days):
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.split_days = self.train_days + self.val_days + self.test_days
    
    def generate_train_data(self):
        return self.__get_first_of_split(self.df, self.train_days, self.split_days)
    
    def generate_val_data(self):
        return self.__get_last_of_split(self.df, self.val_days, self.split_days)
    
    def generate_test_data(self):
        last_part =  self.test_days + self.val_days
        temp_df = self.__get_last_of_split(self.df, last_part, self.split_days)
        return self.__get_last_of_split(temp_df, self.test_days, last_part)

    def __get_first_of_split(self, df, days, split_days):
        if days > self.split_days:
             raise ValueError("Days parameter larger than the day interval")
        temp_df = df.copy()
        global_day = [0]
        last_day = [None]

        first_df = temp_df[temp_df[self.time_col_name].apply(lambda x: self.__filter_days(days, split_days, global_day, last_day, pd.to_datetime(x)))]

        return first_df
        
    def __get_last_of_split(self, df, days, split_days):
        if days > self.split_days:
             raise ValueError("Days parameter larger than the day interval")
        temp_df = df.copy()
        global_day = [0]
        last_day = [None]
        days = self.split_days - days

        last_df = temp_df[temp_df[self.time_col_name].apply(lambda x: not self.__filter_days(days, split_days, global_day, last_day, pd.to_datetime(x)))]

        return last_df
    

   

    def __get_average_time_diff(self):
        # Convert the 'timestamp' column to datetime
        self.df[self.time_col_name] = pd.to_datetime(self.df[self.time_col_name])

        # Calculate the difference between consecutive timestamps
        self.df['time_diff'] = self.df[self.time_col_name].diff()

        # Calculate the average time difference
        self.average_time_diff  = self.df['time_diff'].mean()  

        self.df = self.df.drop(columns=['time_diff'])

