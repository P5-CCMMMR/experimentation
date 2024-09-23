import lightning as L
import torch.utils.data as data
import matplotlib
import pandas as pd

from src.data_preprocess.create_sequnces import create_sequences
from src.data_preprocess.data import split_data_train_and_test
from src.network.SimpleLSTM import SimpleLSTM
from src.network.lstm_trainer import LSTMTrainer

matplotlib.use("Agg")

# Hyper params
hidden_size = 32
epochs = 1
learning_rate = 0.005
seq_len = 4
batch_size = 64
num_workers = 8

training_days = 18
test_days = 20 - training_days

TIMESTAMP = "Timestamp"
TRAIN_DATA_PATH = "dataset/NIST_cleaned_train.csv"
TEST_DATA_PATH = "dataset/NIST_cleaned_test.csv"
DATA_PATH = "dataset/NIST_cleaned.csv"


def main():
    try:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError as e:
        try:
            df = pd.read_csv(DATA_PATH)
            train_data, test_data = split_data_train_and_test(df, training_days, test_days, TIMESTAMP)
        except FileNotFoundError as e:
            print(DATA_PATH + " not found")

    model = SimpleLSTM(hidden_size, batch_size)

    training_xs, training_ys = create_sequences(train_data, seq_len, TIMESTAMP)
    training_loader = data.DataLoader(data.TensorDataset(training_xs, training_ys), batch_size=batch_size, drop_last=True, num_workers=8)

    test_xs, test_ys = create_sequences(train_data, seq_len, TIMESTAMP)
    test_loader = data.DataLoader(data.TensorDataset(test_xs, test_ys), batch_size=batch_size, drop_last=True, num_workers=8)
    timestamps = test_data.values[:, 0]

    trainer = LSTMTrainer(model, learning_rate, timestamps)
    lightning_trainer = L.Trainer(max_epochs=epochs)
    lightning_trainer.fit(trainer, training_loader)
    lightning_trainer.test(trainer, test_loader)

main()