import pandas as pd
from datetime import datetime

def split_data_train_and_test(df, training_days, test_days, time_row_name):
    train_test_split_days = training_days + test_days

    def filter_days(global_day: int, last_day: int, current_date: datetime):
        if current_date.date() != last_day[0]:
            last_day[0] = current_date.date()
            global_day[0] += 1
            if global_day[0] == train_test_split_days:
                global_day[0] = 0
        return global_day[0] < training_days

    global_day = [0]
    last_day = [None]

    traindf = df[df[time_row_name].apply(lambda x: filter_days(global_day, last_day, pd.to_datetime(x)))]

    global_day = [0]
    last_day = [None]

    testdf = df[df[time_row_name].apply(lambda x: not filter_days(global_day, last_day, pd.to_datetime(x)))]

    return traindf, testdf