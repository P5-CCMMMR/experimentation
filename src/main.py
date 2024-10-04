import argparse
import lightning as L
import matplotlib
import pandas as pd
import multiprocessing
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.util.normalize import normalize
from src.network.models.mc_dropout_lstm import MCDropoutLSTM
from src.network.models.mc_dropout_gru import MCDropoutGRU
from src.network.lit_model import LitModel
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.plot import plot_results
from src.data_preprocess.data import split_data_train_and_test
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from src.util.conditional_early_stopping import ConditionalEarlyStopping

matplotlib.use("Agg")

TARGET_COLUMN = 1
NUM_WORKERS = multiprocessing.cpu_count()

# Hyper parameters
training_batch_size = 128
test_batch_size = 64
hidden_size = 24
n_epochs = 125
seq_len = 96
learning_rate = 0.005
swa_learning_rate = 0.01
num_layers = 2
dropout = 0.55
test_sample_nbr = 200

training_days = 18
test_days = 20 - training_days

TIMESTAMP = "Timestamp"
TRAIN_DATA_PATH = "src/data_preprocess/dataset/NIST_cleaned_train.csv"
TEST_DATA_PATH = "src/data_preprocess/dataset/NIST_cleaned_test.csv"
DATA_PATH = "src/data_preprocess/dataset/NIST_cleaned.csv"

def main(iterations):
    #try:
    #    train_data = pd.read_csv(TRAIN_DATA_PATH)
    #    test_data = pd.read_csv(TEST_DATA_PATH)
    #except FileNotFoundError:
    #    try:
    #        df = pd.read_csv(DATA_PATH)
    #        train_data, test_data = split_data_train_and_test(df, training_days, test_days, TIMESTAMP)
    #    except FileNotFoundError:
    #
    # raise RuntimeError(DATA_PATH + " not found")
    
    # TODO: function used to generate sine wave data for testing
    def generate_sine_wave_data(num_samples, num_features, noise=0):
        x = np.linspace(0, 360, num_samples)
        data = np.sin(x) + noise * np.random.randn(num_samples)
        data = data.reshape(-1, 1) 
        return np.hstack([data] * num_features)

    #df = generate_sine_wave_data(10000, 4, 0.1)

    df = pd.read_csv(DATA_PATH)
    train_len = int(len(df) * 0.8)
    val_len = int(len(df) * 0.1)
    
    train_data = df[:train_len]
    val_data = df[train_len:train_len+val_len]
    test_data = df[train_len+val_len:]
    
    train_data = train_data.values
    val_data = val_data.values
    test_data = test_data.values

    test_timestamps = pd.to_datetime(test_data[:,0])

    train_data, _, _ = normalize(train_data[:,1:].astype(float))
    val_data, _, _ = normalize(val_data[:,1:].astype(float))
    test_data, test_min_vals, test_max_vals = normalize(test_data[:,1:].astype(float))

    best_loss = None
    
    for _ in range(iterations):
        model = MCDropoutGRU(hidden_size, num_layers, dropout)
        lit_model = LitModel(model, learning_rate, test_sample_nbr)
        trainer = L.Trainer(max_epochs=n_epochs, callbacks=[StochasticWeightAveraging(swa_lrs=swa_learning_rate), ConditionalEarlyStopping(threshold=0.2)])
        train_dataset = TimeSeriesDataset(train_data, seq_len, TARGET_COLUMN)
        train_loader = DataLoader(train_dataset, batch_size=training_batch_size, num_workers=NUM_WORKERS)
        val_dataset = TimeSeriesDataset(val_data, seq_len, TARGET_COLUMN)
        val_loader = DataLoader(val_dataset, batch_size=training_batch_size, num_workers=NUM_WORKERS)
        test_dataset = TimeSeriesDataset(test_data, seq_len, TARGET_COLUMN)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=NUM_WORKERS)

        trainer.fit(lit_model, train_loader, val_loader)
        test_results = trainer.test(lit_model, test_loader)

        predictions, actuals = lit_model.get_results()

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
