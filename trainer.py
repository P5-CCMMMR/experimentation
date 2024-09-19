import lightning as L
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

# Hyper params
hidden_size = 32
epochs = 1
learning_rate = 0.005
seq_len = 4
batch_size = 64
num_workers = 8

TIMESTAMP = "Timestamp"
TRAIN_DATA_PATH = "dataset/NIST_cleaned_train.csv"
TEST_DATA_PATH = "dataset/NIST_cleaned_test.csv"
DATA_PATH = "dataset/NIST_cleaned.csv"

training_days = 18
test_days = 20 - training_days

def create_sequences(features: pd.DataFrame, seq_len: int):
    features[TIMESTAMP] = pd.to_datetime(features[TIMESTAMP])
    grouped = features.groupby(features[TIMESTAMP].dt.date)

    xs, ys = [], []
    for _, group in grouped:
        group_features = group.drop(columns=[TIMESTAMP]).values
        num_sequences = len(group_features) - seq_len
        num_features = group_features.shape[1]

        day_xs = np.zeros((num_sequences, seq_len, num_features), dtype=np.float32)
        day_ys = np.zeros((num_sequences, 1), dtype=np.float32)
        for i in range(num_sequences):
            day_xs[i] = group_features[i:i + seq_len]
            day_ys[i] = group_features[i + seq_len, 1]

        xs.append(day_xs)
        ys.append(day_ys)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return torch.tensor(xs), torch.tensor(ys)

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size: int, batch_size: int = 1):
        super(SimpleLSTM, self).__init__()
        self.input_size = 3
        self.output_size = 1
        self.num_layers = 1
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)

class LSTMTrainer(L.LightningModule):
    def __init__(self, model, learning_rate, timestamps):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.timestamps = pd.to_datetime(timestamps)
        self.writer = SummaryWriter()
        self.all_predictions = []
        self.all_actuals = []
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch):
        x, y = batch
        prediction = self(x)
        loss = torch.sqrt(self.criterion(prediction, y))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


# EVALUATE
    def test_step(self, batch):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Accumulate predictions and actuals
        self.all_predictions.extend(prediction.detach().cpu().numpy().flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())

        return loss

    def on_test_epoch_end(self):
        min_length = min(len(self.timestamps), len(self.all_predictions), len(self.all_actuals))
        timestamps = self.timestamps[:min_length]
        predictions = self.all_predictions[:min_length]
        actuals = self.all_actuals[:min_length]

        plt.figure(figsize=(40, 20), dpi=300)
        plt.plot(timestamps, predictions, label="Predictions")
        plt.plot(timestamps, actuals, label="Actual")
        plt.xlabel("Time")
        plt.ylabel("Indoor Temperature")
        plt.title("Predictions vs Actual")
        plt.legend()
        plt.grid()
        plt.gcf().autofmt_xdate()

        self.writer.add_figure("predictions", plt.gcf(), global_step=self.current_epoch)
        plt.close()

model = SimpleLSTM(hidden_size, batch_size)

# DATA
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

training_xs, training_ys = create_sequences(train_data, seq_len)
training_loader = data.DataLoader(data.TensorDataset(training_xs, training_ys), batch_size=batch_size, drop_last=True, num_workers=8)

test_xs, test_ys = create_sequences(train_data, seq_len)
test_loader = data.DataLoader(data.TensorDataset(test_xs, test_ys), batch_size=batch_size, drop_last=True, num_workers=8)
timestamps = test_data.values[:, 0]

trainer = LSTMTrainer(model, learning_rate, timestamps)
lightning_trainer = L.Trainer(max_epochs=epochs)
lightning_trainer.fit(trainer, training_loader)
lightning_trainer.test(trainer, test_loader)
