import pandas as pd
from datetime import datetime
import numpy as np

class DataSplitter():
    def __init__(self, df: pd.DataFrame, time_col_name: str, power_col_name: str):
        self.df = df
        self.split_days = 0
        self.time_col_name = time_col_name
        self.power_col_name = power_col_name
        self.__get_average_time_diff()

    def __filter_days(self, days_to_include: int, global_day: int, last_day: int, current_date: datetime):
        if current_date.date() != last_day[0]:
            last_day[0] = current_date.date()
            global_day[0] += 1
        
            if global_day[0] == self.split_days:
                global_day[0] = 0
        return global_day[0] < days_to_include

    def get_first_of_split(self, days):
        if days > self.split_days:
             raise ValueError("Days parameter larger than the day interval")
        temp_df = self.df.copy()
        global_day = [0]
        last_day = [None]

        first_df = temp_df[temp_df[self.time_col_name].apply(lambda x: self.__filter_days(days, global_day, last_day, pd.to_datetime(x)))]

        return first_df
        
    def get_last_of_split(self, days):
        if days > self.split_days:
             raise ValueError("Days parameter larger than the day interval")
        temp_df = self.df.copy()
        global_day = [0]
        last_day = [None]
        days = self.split_days - days

        last_df = temp_df[temp_df[self.time_col_name].apply(lambda x: not self.__filter_days(days, global_day, last_day, pd.to_datetime(x)))]

        return last_df


    def set_split_interval(self, split_days: int):
        self.split_days = split_days
    
    def get_lt_power(self, power_watt: int, consecutive_points: int = 1):
        temp_df = self.df.copy()
        consec_rows = []
        new_df = pd.DataFrame()

        for index, row in temp_df.iterrows():
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

        for index, row in temp_df.iterrows():
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

    def __get_average_time_diff(self):
        # Convert the 'timestamp' column to datetime
        self.df[self.time_col_name] = pd.to_datetime(self.df[self.time_col_name])

        # Calculate the difference between consecutive timestamps
        self.df['time_diff'] = self.df[self.time_col_name].diff()

        # Calculate the average time difference
        self.average_time_diff  = self.df['time_diff'].mean()  

        self.df = self.df.drop(columns=['time_diff'])



# Example usage
if __name__ == "__main__":
    # Generate a date range with 15-minute intervals
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=96*10, freq='15T'),  # 96 intervals per day, for 10 days
        'PowerConsumption': np.random.randint(0, 100, size=96*10)  # Random power consumption values between 0 and 100
    }
    df = pd.DataFrame(data)

    splitter = DataSplitter(df, 'timestamp', 'PowerConsumption')
    splitter.set_split_interval(10)

    train_df = splitter.get_first_of_split(3)
    test_df = splitter.get_last_of_split(7)

    print("Train DataFrame:")
    print(train_df)

    print("\nTest DataFrame:")
    print(test_df)

    lt_power_df = splitter.get_lt_power(50, 3)  # Adjusted power threshold for more data points
    print("\nRows with power less than 50 for at least 3 consecutive points within average time difference:")
    print(lt_power_df)

    mt_power_df = splitter.get_mt_power(50, 3)  # Adjusted power threshold for more data points
    print("\nRows with power greater than 50 for at least 3 consecutive points within average time difference:")
    print(mt_power_df)