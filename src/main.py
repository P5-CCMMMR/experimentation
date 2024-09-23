import lightning as L
import matplotlib
import pandas as pd
import multiprocessing

from torch.utils.data import DataLoader

from src.data_preprocess.data import split_data_train_and_test
from src.network.normal_lstm import NormalLSTM
from src.network.lit_lstm import LitLSTM
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.plot import plot_results

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

def main():
    try:
        df = pd.read_csv(DATA_PATH)
        train_data, test_data = split_data_train_and_test(df, training_days, test_days, TIMESTAMP)
    except FileNotFoundError as e:
        print(DATA_PATH + " not found")

    dfv = df.values
    train_len = int(len(dfv) * 0.8)

    data = dfv[train_len:, 1:].astype(float)
    test_min_vals = data.min(axis=0)
    test_max_vals = data.max(axis=0)

    model = NormalLSTM(hidden_size, num_layers, dropout)
    lit_lstm = LitLSTM(model, learning_rate)
    trainer = L.Trainer(max_epochs=n_epochs)
    train_dataset = TimeSeriesDataset(train_data, seq_len, TARGET_COLUMN)
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, num_workers=NUM_WORKERS)
    trainer.fit(lit_lstm, train_loader)

    test_dataset = TimeSeriesDataset(test_data, seq_len, TARGET_COLUMN)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=NUM_WORKERS)

    trainer.test(lit_lstm, test_loader)
    predictions, actuals = lit_lstm.get_results()
    test_timestamps = pd.to_datetime(dfv[train_len:, 0])

    plot_results(predictions, actuals, test_timestamps, test_min_vals, test_max_vals)

main()


