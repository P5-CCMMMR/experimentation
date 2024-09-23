import lightning as L
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

# Hyper parameters
training_batch_size = 64
test_batch_size = 105
hidden_size = 32
n_epochs = 10
seq_len = 16
learning_rate = 0.005
num_layers = 2
dropout = 0.2
num_workers = 13

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
    def __init__(self, data, seq_len: int):
        self.data = data
        self.seq_length = seq_len

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class LitLSTM(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, test_timestamps):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.test_timestamps = test_timestamps
    
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

    def on_test_epoch_start(self):
        self.writer = SummaryWriter()
        self.all_predictions = []
        self.all_actuals = []

    def on_test_epoch_end(self):
        min_length = min(len(self.test_timestamps), len(self.all_predictions))
        timestamps = self.test_timestamps[:min_length]
        predictions = self.all_predictions
        actuals = self.all_actuals

        plt.plot(timestamps, predictions, label="Predictions")
        plt.plot(timestamps, actuals, label="Actual")
        plt.xlabel("Time")
        plt.ylabel("Indoor Temperature")
        plt.title("Predictions vs actuals")
        plt.legend()
        plt.grid()
        plt.gcf().autofmt_xdate()

        self.writer.add_figure("predictions", plt.gcf(), global_step=self.current_epoch)

def normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)

dfv = pd.read_csv("NIST_cleaned.csv").values
train_len = int(len(dfv) * 0.8)
train_data = normalize(dfv[:train_len, 1:].astype(float))
test_data = normalize(dfv[train_len:, 1:].astype(float))
test_timestamps = pd.to_datetime(dfv[train_len:, 0])

model = NormalLSTM(hidden_size, num_layers, dropout)
lit_lstm = LitLSTM(model, learning_rate, test_timestamps)
trainer = L.Trainer(max_epochs=n_epochs)
train_dataset = TimeSeriesDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=False, num_workers=num_workers)
trainer.fit(lit_lstm, train_loader)

test_dataset = TimeSeriesDataset(test_data, seq_len)
test_dataset = TimeSeriesDataset(test_data, seq_len)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers)
trainer.test(lit_lstm, test_loader)
