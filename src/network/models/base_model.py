import lightning as L
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.error import RMSE
from src.util.constants import NUM_WORKERS, TARGET_COLUMN

class BaseModel(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int, train_data, val_data, test_data):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.all_predictions: list[float] = []
        self.all_actuals: list[float] = []
        
        self.setup_data_loaders(train_data, val_data, test_data)
    
    def setup_data_loaders(self, train_data, val_data, test_data):
        train_dataset = TimeSeriesDataset(train_data, self.seq_len, TARGET_COLUMN)
        val_dataset = TimeSeriesDataset(val_data, self.seq_len, TARGET_COLUMN)
        test_dataset = TimeSeriesDataset(test_data, self.seq_len, TARGET_COLUMN)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = RMSE(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = RMSE(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def test_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = RMSE(y_hat, y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)
        
        self.all_predictions.extend(y_hat.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
    
    def get_results(self):
        return self.all_predictions, self.all_actuals
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
    
    def plot_results(predictions, actuals, timestamps, min_vals, max_vals, target_column):
        writer = SummaryWriter()

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Rescale predictions and actuals 
        predictions = predictions * (max_vals[target_column] - min_vals[target_column]) + min_vals[target_column]
        actuals = actuals * (max_vals[target_column] - min_vals[target_column]) + min_vals[target_column]
        timestamps = timestamps[:len(predictions)]

        plt.plot(timestamps, predictions, label="Prediction")
        plt.plot(timestamps, actuals, label="Actual")

        plt.xlabel("Time")
        plt.ylabel("Indoor Temperature")
        plt.title("Predictions vs Actuals")
        plt.legend()
        plt.grid()
        plt.gcf().autofmt_xdate()

class RNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        input_size = 3
        output_size = 1
        self.rnn = nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

class GRU(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        input_size = 3
        output_size = 1
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

class LSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(LSTM, self).__init__()
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