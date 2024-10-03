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
    train_data, test_data = getTrainTestDataset()
    modelTrainingAndEval(train_data, test_data, MODEL_ITERATIONS)
    on_df, off_df = getOnOffDataset()
    flexPredictEval(on_df, off_df)

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

def flexPredictEval(df, model):
    df_sequences = splitDateframeByContinuity(df, 15, 3)

    predictions = []

    for seq in df_sequences:
        #print(seq)
        #print(len(seq))
        mutiTimestepForecasting(model, seq, len(seq))
        break

# 
def mutiTimestepForecasting(model, seq, timesteps):
    predictions = []
    print(seq[0:4,1:].astype(float))
    data, min_vals, max_vals = normalize(seq[0:4,1:].astype(float))
    last_out_temp = data[3]
    print(last_out_temp)
    print(data)
    data = data[0:4:]
    print(data[0])
#    for i in range(0, timesteps):
#        dataset = TimeSeriesDataset(data, 4, TARGET_COLUMN)
#        loader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS)
#        data = data[1:4:].append()

#    off_data, off_min_vals, off_max_vals = normalize(off_df.values[:,1:].astype(float))
#    on_data, on_min_vals, on_max_vals  = normalize(on_df.values[:,1:].astype(float))
#
#    off_dataset = TimeSeriesDataset(off_data, 4, TARGET_COLUMN)
#    off_loader = DataLoader(off_dataset, batch_size=test_batch_size, num_workers=NUM_WORKERS)
#
#    # Check the content of the test_loader
#    last_batch = None
#    for batch in off_loader:
#        last_batch = batch
#
#
#    model = NormalLSTM(hidden_size, num_layers, dropout)
#    model.load_state_dict(torch.load('model.pth', weights_only=True))
#    model.eval()
#    model.cuda()
#
#    if last_batch is not None:
#        features, labels = last_batch
#        features = features.cuda()
#        with torch.no_grad():
#            predictions = model(features)
#        
#        predictions_np = predictions.cpu().numpy()
#        print("Predictions:")
#        print(denormalize(predictions_np, off_min_vals[1], off_max_vals[1]))
#        print("actual values: ")
#        print(denormalize(labels, off_min_vals[1], off_max_vals[1]))
#    else:
#        print("No batches found in the DataLoader.")



main()