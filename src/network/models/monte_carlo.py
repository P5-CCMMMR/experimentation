import lightning as L
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.error import RMSE
from src.util.constants import NUM_WORKERS, TARGET_COLUMN

class MCLit(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, test_sample_nbr: int, seq_len: int, batch_size: int, train_data, val_data, test_data):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.test_sample_nbr = test_sample_nbr
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.all_predictions: tuple[list[float], list[float]] = ([], [])
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
        mean_prediction, std_prediction = self.__predict_with_mc_dropout(x)
        loss = RMSE(torch.tensor(mean_prediction, device=y.device), y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)

        self.all_predictions[0].extend(mean_prediction.flatten())
        self.all_predictions[1].extend(std_prediction.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
    
    def __predict_with_mc_dropout(self, x):
        self.model.train()
        predictions = []

        with torch.no_grad():
            for _ in range(self.test_sample_nbr):
                y_hat = self.model(x)
                predictions.append(y_hat.cpu().numpy())

        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        return mean_prediction, std_prediction
    
    def get_results(self):
        return self.all_predictions, self.all_actuals
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader

class MCDropoutRNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(MCDropoutRNN, self).__init__()
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

class MCDropoutGRU(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(MCDropoutGRU, self).__init__()
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

class MCDropoutLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(MCDropoutLSTM, self).__init__()
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
    