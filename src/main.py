import argparse
import lightning as L
import matplotlib
import pandas as pd
import multiprocessing

import torch
from torch.utils.data import DataLoader
from src.util.normalize import normalize
from src.network.normal_lstm import NormalLSTM
from src.network.lit_lstm import LitLSTM
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.plot import plot_results
from src.data_preprocess.data import split_data_train_and_test

matplotlib.use("Agg")

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

training_days = 18
test_days = 20 - training_days

TIMESTAMP = "Timestamp"
TRAIN_DATA_PATH = "dataset/NIST_cleaned_train.csv"
TEST_DATA_PATH = "dataset/NIST_cleaned_test.csv"
DATA_PATH = "dataset/NIST_cleaned.csv"

def main(iterartions):
    try:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        try:
            df = pd.read_csv(DATA_PATH)
            train_data, test_data = split_data_train_and_test(df, training_days, test_days, TIMESTAMP)
        except FileNotFoundError:
            raise RuntimeError(DATA_PATH + " not found")

    train_data = train_data.values
    test_data = test_data.values

    test_timestamps = pd.to_datetime(test_data[:,0])

    train_data, _, _ = normalize(train_data[:,1:].astype(float))
    test_data, test_min_vals, test_max_vals = normalize(test_data[:,1:].astype(float))

    best_loss = None
    
    for i in range(0, iterartions):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LSTM model training and testing.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations to run the training and testing loop.')
    args = parser.parse_args()
    main(args.iterations)

