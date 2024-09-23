from datetime import datetime
import pandas as pd
from src.util.hyper_parameters import TIMESTAMP, DATASETFOLDER, training_days, test_days

TRAIN_DATA_PATH = DATASETFOLDER + "/NIST_cleaned_train.csv"
TEST_DATA_PATH = DATASETFOLDER + "/NIST_cleaned_test.csv"
DATA_PATH = DATASETFOLDER + "/NIST_cleaned.csv"

try:
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
except FileNotFoundError as e:
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError as e:
        print(DATA_PATH + " not found")


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

    traindf = df[df[TIMESTAMP].apply(lambda x: filter_days(global_day, last_day, pd.to_datetime(x)))]

    global_day = [0]
    last_day = [None]

    testdf = df[df[TIMESTAMP].apply(lambda x: not filter_days(global_day, last_day, pd.to_datetime(x)))]

    traindf.to_csv(TRAIN_DATA_PATH, index=False)
    testdf.to_csv(TEST_DATA_PATH, index=False)

    train_data = traindf
    test_data = testdf