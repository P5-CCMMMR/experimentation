import argparse
import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import multiprocessing
import torch
from torch.utils.data import DataLoader
from src.util.normalize import normalize, denormalize
from src.network.models.normal_lstm import NormalLSTM
from src.network.lit_lstm import LitLSTM
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.data_preprocess.usage_timeseries_dataset import UsageTimeSeriesDataset
from src.util.plot import plot_results
from src.data_preprocess.data import DataSplitter

matplotlib.use("Agg")

MODEL_ITERATIONS = 0
TARGET_COLUMN = 1
NUM_WORKERS = multiprocessing.cpu_count()

# Hyper parameters
training_batch_size = 64
test_batch_size = 105
hidden_size = 32
n_epochs = 10
seq_len = 96
learning_rate = 0.005
num_layers = 1
dropout = 0

# Data Parameters
nist = {
    "training_days"       : 18, 
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : 100,
    "on_limit_w"          : 1500,   
    "consecutive_points"  : 3,
    "train_data_path"     : "src/data_preprocess/dataset/train/NIST.csv",
    "test_data_path"      : "src/data_preprocess/dataset/train/NIST.csv",
    "on_data_path"        : "src/data_preprocess/dataset/on/NIST.csv",
    "off_data_path"       : "src/data_preprocess/dataset/off/NIST.csv",
    "data_path"           : "src/data_preprocess/dataset/NIST_cleaned.csv"
}

dengiz = {
    "training_days"       : 18, 
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : None,   # Yet to be known
    "on_limit_w"          : None,   # Yet to be known
    "consecutive_points"  : 3,
    "train_data_path"     : "src/data_preprocess/dataset/train/Dengiz.csv",
    "test_data_path"      : "src/data_preprocess/dataset/test/Dengiz.csv",
    "on_data_path"        : "src/data_preprocess/dataset/on/Dengiz.csv",
    "off_data_path"       : "src/data_preprocess/dataset/off/Dengiz.csv",
    "data_path"           : "src/data_preprocess/dataset/Dengiz_cleaned.csv"
}

used_dataset       = nist
training_days      = used_dataset["training_days"]
test_days          = used_dataset["test_days"]
validation_days    = used_dataset["validation_days"]
off_limit          = used_dataset["off_limit_w"]
on_limit           = used_dataset["on_limit_w"]
consecutive_points = used_dataset["consecutive_points"]

# General Constant
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"

# Paths
TRAIN_DATA_PATH = used_dataset[ "train_data_path"]
TEST_DATA_PATH  = used_dataset["test_data_path"]
ON_DATA_PATH    = used_dataset["on_data_path"]
OFF_DATA_PATH   = used_dataset["off_data_path"]
DATA_PATH       = used_dataset["data_path"]


def main():
    seq_len = 4

    train_data, test_data = getTrainTestDataset()
    modelTrainingAndEval(train_data, test_data, MODEL_ITERATIONS)
    on_df, off_df = getOnOffDataset()
    model = NormalLSTM(hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()
    getMafe(on_df, model, seq_len)
    getMafe(off_df, model, seq_len)

def getOnOffDataset():
    try:
        off_df = pd.read_csv(OFF_DATA_PATH)
        on_df = pd.read_csv(ON_DATA_PATH)
        print("On/Off data loaded successfully")
    except FileNotFoundError:
        print("On/Off data files not found, generating new data")
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            raise RuntimeError(DATA_PATH + " not found")
        
        ds = DataSplitter(df, TIMESTAMP, POWER)
        off_df = ds.get_lt_power(off_limit, 3)
        on_df = ds.get_mt_power(on_limit, 3)

        off_df.to_csv(OFF_DATA_PATH , index=False)
        on_df.to_csv(ON_DATA_PATH, index=False)
        print("On/Off data generated and saved")

    return on_df, off_df


def getTrainTestDataset():
    try:
        df = pd.read_csv(DATA_PATH)
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        try:
            ds = DataSplitter(df, TIMESTAMP, POWER)
            ds.set_split_interval(test_days + training_days)
            train_data = ds.get_first_of_split(training_days)
            test_data = ds.get_last_of_split(test_days)

            train_data.to_csv(TRAIN_DATA_PATH , index=False)
            test_data.to_csv(TEST_DATA_PATH, index=False)

        except FileNotFoundError:
            raise RuntimeError(DATA_PATH + " not found")
    return train_data, test_data

def modelTrainingAndEval(train_data, test_data, iterations):
    train_data = train_data.values
    test_data = test_data.values

    test_timestamps = pd.to_datetime(test_data[:,0])

    train_data, _, _                        = normalize(train_data[:,1:].astype(float))
    test_data, test_min_vals, test_max_vals = normalize(test_data[:,1:].astype(float))
    best_loss = None
    for _ in range(0, iterations):
        model = NormalLSTM(hidden_size, num_layers, dropout)
        lit_lstm = LitLSTM(model, learning_rate)
        trainer = L.Trainer(max_epochs=n_epochs)
        train_dataset = TimeSeriesDataset(train_data, seq_len, TARGET_COLUMN)
        train_loader = DataLoader(train_dataset, batch_size=training_batch_size, num_workers=NUM_WORKERS)
        test_dataset = TimeSeriesDataset(test_data, seq_len, TARGET_COLUMN)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=NUM_WORKERS)


        trainer.fit(lit_lstm, train_loader)
        test_results = trainer.test(lit_lstm, test_loader)

        predictions, actuals = lit_lstm.get_results()

        test_loss = test_results[0].get('test_loss_epoch', None) if test_results else None

        if best_loss is None or best_loss > test_loss :
            print("NEW BEST")
            plot_results(predictions, actuals, test_timestamps, test_min_vals, test_max_vals, TARGET_COLUMN)
            best_loss = test_loss 
            torch.save(model.state_dict(), 'model.pth')

def splitDateframeByContinuity(df, time_difference: int, sequence_min_len: int): 
    sequences = []
    temp_sequence = []
    last_time = None
    for index, row in df.iterrows():
        if last_time is not None and pd.to_datetime(row[TIMESTAMP]) - last_time != pd.Timedelta(minutes=time_difference):
            if len(temp_sequence) > sequence_min_len:
                sequences.append(np.array(temp_sequence))
            temp_sequence = []
        temp_sequence.append(row)
        last_time = pd.to_datetime(row[TIMESTAMP])
    if temp_sequence:
        sequences.append(np.array(temp_sequence))
    return sequences

def getMafe(df, model, seq_len):
    df_data = splitDateframeByContinuity(df, 15, 3)

    flex_predictions = []
    flex_actual_values = []

    error = 0
    temp_boundery = 0.5

    print("data arrays: " + str(len(df_data)))
    for data in df_data:
        in_temp_predictions = multiTimestepForecasting(model, data, len(data), seq_len)

        in_temp_actual = data[4: ,2:3]
        last_in_temp = data[3][2]

        lower_boundery = last_in_temp - temp_boundery
        upper_boundery = last_in_temp + temp_boundery

        actual_flex = flexPredict(in_temp_actual, lower_boundery, upper_boundery, error)
        predicted_flex = flexPredict(in_temp_predictions, lower_boundery, upper_boundery, error)

  
        flex_predictions.append(predicted_flex)
        flex_actual_values.append(actual_flex)

        # need to check why the prediction allways perfect, and why its either all the data its flexible or no data

def multiTimestepForecasting(model, data, timesteps, sequence_len):
    if (len(data) < sequence_len): 
        return []
    predictions = []
    seq, min_vals, max_vals = normalize(data[0:sequence_len, 1:].astype(float)) 
    last_out_temp = seq[3][2] # magick numbers doesn't work if data not in this format (Power, inTemp, outTemp)
    last_out_power = seq[3][0]
    
    for _ in range(0, timesteps - sequence_len):
        dataset = UsageTimeSeriesDataset(seq, sequence_len)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS)

        for batch in dataloader:
            outputs = model(batch)
            predictions.append(outputs.item())

        seq = seq[1:sequence_len]
        new_row = np.array([[last_out_power, predictions[len(predictions) - 1], last_out_temp]])
        seq = np.append(seq, new_row, axis=0)
    return denormalize(predictions, min_vals[1:2], max_vals[1:2])

def flexPredict(forecasts, lower_bound, upper_bound, error):
    flex_iter = 0


    for forecast in forecasts:
        if lower_bound + error <= forecast and upper_bound - error >= forecast:    
            flex_iter = flex_iter + 1
        else:
            break
    
    return flex_iter

main()