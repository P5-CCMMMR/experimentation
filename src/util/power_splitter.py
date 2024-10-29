import pandas as pd

class PowerSplitter():
    def __init__(self, df: pd.DataFrame, time_col_name: str, power_col_name: str):
        self.df = df
        self.time_col_name = time_col_name
        self.power_col_name = power_col_name
        self.average_time_diff = None
        self.__get_average_time_diff()

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
    
    def __get_average_time_diff(self):
        # Convert the 'timestamp' column to datetime
        self.df[self.time_col_name] = pd.to_datetime(self.df[self.time_col_name])

        # Calculate the difference between consecutive timestamps
        self.df['time_diff'] = self.df[self.time_col_name].diff()

        # Calculate the average time difference
        self.average_time_diff  = self.df['time_diff'].mean()  

        self.df = self.df.drop(columns=['time_diff'])
