import lightning as L
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

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

def RMSE(y_hat: float, y: float):
        # Add small term to avoid divison by zero
        epsilon = 1e-8
        return torch.sqrt(nn.functional.mse_loss(y_hat, y) + epsilon)

class NormalLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(NormalLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        input_size = 3
        output_size = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out).squeeze()

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len: int, target_column: int):
        self.data = data
        self.seq_length = seq_len
        self.target_column = target_column

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, self.target_column]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class LitLSTM(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.all_predictions = []
        self.all_actuals = []
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = RMSE(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def test_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = RMSE(y_hat, y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)

        self.all_predictions.extend(y_hat.detach().cpu().numpy().flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
    
    def get_results(self):
        return self.all_predictions, self.all_actuals

def normalize(data: np.ndarray):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals), min_vals, max_vals

def plot_results(predictions, actuals, timestamps, min_vals, max_vals):
    writer = SummaryWriter()

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    predictions = predictions * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
    actuals = actuals * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
    timestamps = timestamps[:len(predictions)]

    plt.plot(timestamps, predictions, label="Prediction")
    plt.plot(timestamps, actuals, label="Actual")
    plt.xlabel("Time")
    plt.ylabel("Indoor Temperature")
    plt.title("Predictions vs Actuals")
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()

    writer.add_figure("predictions", plt.gcf())
    writer.close()

dfv = pd.read_csv("NIST_cleaned.csv").values
train_len = int(len(dfv) * 0.8)
train_data, _, _ = normalize(dfv[:train_len, 1:].astype(float))
test_data, test_min_vals, test_max_vals = normalize(dfv[train_len:, 1:].astype(float))

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