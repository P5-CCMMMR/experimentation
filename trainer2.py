import lightning as L
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super(SimpleLSTM, self).__init__()
        self.input_size = 3
        self.output_size = 1
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)
    
class LitLSTM(L.LightningModule):
    def __init__(self, model, timestamps):
        super().__init__()
        self.model = model
        self.timestamps = pd.to_datetime(timestamps)
        self.writer = SummaryWriter()
        self.all_predictions = []
        self.all_actuals = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = nn.functional.mse_loss(prediction, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x)
        loss = nn.functional.mse_loss(prediction, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        self.all_predictions.extend(prediction.detach().cpu().numpy().flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())

    def on_test_epoch_end(self):
        min_length = min(len(self.timestamps), len(self.all_predictions), len(self.all_actuals))
        timestamps = self.timestamps[:min_length]
        predictions = self.all_predictions[:min_length]
        actuals = self.all_actuals[:min_length]

        plt.plot(timestamps, predictions, label="Predictions")
        plt.plot(timestamps, actuals, label="Actual")
        plt.xlabel("Time")
        plt.ylabel("Indoor Temperature")
        plt.title("Predictions vs actuals")
        plt.legend()
        plt.grid()
        plt.gcf().autofmt_xdate()

        self.writer.add_figure("predictions", plt.gcf(), global_step=self.current_epoch)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=4):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

dfv = pd.read_csv("NIST_cleaned.csv").values
train_len = int(len(dfv) * 0.7)

train_data = dfv[:train_len, 1:].astype(float)
test_data = dfv[train_len:, 1:].astype(float)
test_timestamps = dfv[train_len:, 0]

model = SimpleLSTM(64)
lit_lstm = LitLSTM(model, test_timestamps)
trainer = L.Trainer(max_epochs=10)
train_dataset = TimeSeriesDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
trainer.fit(lit_lstm, train_loader)

test_dataset = TimeSeriesDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=32)
trainer.test(lit_lstm, train_loader)
