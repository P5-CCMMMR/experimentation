import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import multiprocessing
import torch
from torch.utils.data import DataLoader
from src.util.flex_predict import flexPredict
from src.util.multi_timestep_forecast import multiTimestepForecasting
from src.util.normalize import normalize
from src.network.models.normal_lstm import NormalLSTM
from src.network.lit_lstm import LitLSTM
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.plot import plot_results
from src.data_preprocess.data_handler import DataHandler
from src.data_preprocess.data_splitter import DataSplitter

matplotlib.use("Agg")

MODEL_ITERATIONS = 10
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
time_horizon = 4

# Data Parameters
nist = {
    "training_days"       : 18, 
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : 100,
    "on_limit_w"          : 1500,   
    "consecutive_points"  : 3,

    "train_data_path"     : "src/data_preprocess/dataset/train/NIST.csv",
    "test_data_path"      : "src/data_preprocess/dataset/test/NIST.csv",
    "on_data_path"        : "src/data_preprocess/dataset/on/NIST.csv",
    "off_data_path"       : "src/data_preprocess/dataset/off/NIST.csv",
    "data_path"           : "src/data_preprocess/dataset/NIST_cleaned.csv",

    "power_col"           : "PowerConsumption",
    "timestamp_col"       : "Timestamp"
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
    "data_path"           : "src/data_preprocess/dataset/Dengiz_cleaned.csv",

    "power_col"           : "PowerConsumption",
    "timestamp_col"       : "Timestamp"
}


def main():
    temp_boundery = 0.5
    seq_len = 4
    error = 0

    mnist = DataHandler(nist, DataSplitter)

    # train_data, test_data = mnist.get_train_test_data()
    # modelTrainingAndEval(train_data, test_data, MODEL_ITERATIONS)

    model = NormalLSTM(hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()

    on_df, off_df = mnist.get_on_off_data()

    on_data_arr = mnist.split_dataframe_by_continuity(on_df, 15, seq_len)
    off_data_arr = mnist.split_dataframe_by_continuity(off_df, 15, seq_len)

    getMafe(on_data_arr, model, seq_len, error, temp_boundery)
    getMafe(off_data_arr, model, seq_len, error, temp_boundery)


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


def getMafe(data_arr, model, seq_len, error, boundary):
    flex_predictions = []
    flex_actual_values = []

    for data in data_arr:
        for i in range(0, len(data), seq_len):
            if len(data) < i + seq_len * 2:
                break

            in_temp_idx = 2

            input_data = data[i: i + seq_len]

            # get the actual result data by first gettin the next *seq* data steps forward, 
            # and taking only the in_temp_id column to get the actual result indoor temperatures
            result_actual = data[i + seq_len : i + (seq_len * 2), in_temp_idx:in_temp_idx + 1] 

            result_predictions = multiTimestepForecasting(model, input_data, seq_len)

            last_in_temp = input_data[len(input_data) - 1][2]

            lower_boundery = last_in_temp - boundary
            upper_boundery = last_in_temp + boundary

            actual_flex = flexPredict(result_actual, lower_boundery, upper_boundery, error)
            predicted_flex = flexPredict(result_predictions, lower_boundery, upper_boundery, error)

    
            flex_predictions.append(predicted_flex)
            flex_actual_values.append(actual_flex)

        # need to check why the prediction allways perfect, and why its either all the data its flexible or no data
    flex_difference = [a - b for a, b in zip(flex_predictions, flex_actual_values)]
    print(flex_difference)



main()