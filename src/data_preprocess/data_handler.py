import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, config, data_splitter_class):
        self.train_data_path = config["train_data_path"]
        self.test_data_path = config["test_data_path"]
        self.on_data_path = config["on_data_path"]
        self.off_data_path = config["off_data_path"]
        self.data_path = config["data_path"]

        self.timestamp_col = config["timestamp_col"]
        self.power_col = config["power_col"]
        self.training_days = config["training_days"]
        self.test_days = config["test_days"]
        self.off_limit = config["off_limit_w"]
        self.on_limit = config["on_limit_w"]
        
        self.data_splitter_class = data_splitter_class

        self.df = None
        self.on_df = None
        self.off_df = None
        self.train_df = None
        self.test_df = None

    def load_dataset(self, file_path):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            return None

    def save_dataset(self, df, file_path):
        df.to_csv(file_path, index=False)

    def get_on_off_data(self):
        off_df = self.load_dataset(self.off_data_path)
        on_df = self.load_dataset(self.on_data_path)
        if off_df is not None or on_df is not None:
            return on_df, off_df
        
        if self.test_df == None:
            self.get_train_test_data();

        off_df = self.ds.get_lt_power(self.off_limit, 3)
        on_df = self.ds.get_mt_power(self.on_limit, 3)
        
        self.save_dataset(self.off_df, self.off_data_path)
        self.save_dataset(self.on_df, self.on_data_path)

        return on_df, off_df

    def get_train_test_data(self):
        if self.train_df is not None or self.test_df is not None:
            return self.train_df, self.test_df
        
        self.train_df = self.load_dataset(self.train_data_path)
        self.test_df = self.load_dataset(self.test_data_path)
        if self.train_df is not None or self.test_df is not None:
            return self.train_df, self.test_df

        if self.df is None:
            self.df = self.load_dataset(self.data_path)
        
        ds = self.data_splitter_class(self.df, self.timestamp_col, self.power_col)

        ds.set_split_interval(self.test_days + self.training_days)
        self.train_df= ds.get_first_of_split(self.training_days)
        self.test_df = ds.get_last_of_split(self.test_days)

        self.save_dataset(self.train_df, self.train_data_path)
        self.save_dataset(self.test_df, self.test_data_path)

        return self.train_df, self.test_df

    def split_dataframe_by_continuity(self, df, time_difference: int, sequence_min_len: int):
        sequences = []
        temp_sequence = []
        last_time = None
        for _, row in df.iterrows():
            if last_time is not None and pd.to_datetime(row[self.timestamp_col]) - last_time != pd.Timedelta(minutes=time_difference):
                if len(temp_sequence) > sequence_min_len:
                    sequences.append(np.array(temp_sequence))
                temp_sequence = []
            temp_sequence.append(row)
            last_time = pd.to_datetime(row[self.timestamp_col])
        if temp_sequence:
            sequences.append(np.array(temp_sequence))
        return sequences