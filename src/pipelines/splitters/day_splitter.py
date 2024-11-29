from src.pipelines.splitters.splitter import Splitter
import pandas as pd

class DaySplitter(Splitter):
    def __init__(self, time_col_name: str, power_col_name: str, train_split, val_split, test_split):
        super().__init__(train_split, val_split, test_split)
        self.time_col_name = time_col_name
        self.power_col_name = power_col_name

    def get_train(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.__get_days_of_split(df, 0, self.train_split - 1)
    
    def get_val(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.__get_days_of_split(df, self.train_split, self.train_split + self.val_split - 1)
    
    def get_test(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.__get_days_of_split(df, self.train_split + self.val_split, self.total_split - 1)

    def __get_days_of_split(self, df, start_day, end_day):
        if end_day > self.total_split:
            raise ValueError("Days parameter larger than the day interval")
        temp_df = df.copy()
        day_counter = 0
        last_day = None

        filtered_indices = []
        for idx, row in temp_df.iterrows():
            date = pd.to_datetime(row[self.time_col_name]).date()
            if last_day == None: last_day = date

            if date != last_day:
                last_day = date
                day_counter += 1
            
                if day_counter == self.total_split:
                    day_counter = 0
            if start_day <= day_counter <= end_day:
                filtered_indices.append(idx)
        
        last_df = temp_df.loc[filtered_indices]

        return last_df
