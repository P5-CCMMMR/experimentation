import lightning as L
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

# Hyper params
hidden_size = 32
epochs = 1
learning_rate = 0.005
seq_len = 4
batch_size = 64
num_workers = 8

def create_sequences(features: np.ndarray, seq_len: int):
    num_sequences = len(features) - seq_len
    num_features = features.shape[1]

    xs = np.zeros((num_sequences, seq_len, num_features), dtype=np.float32)
    ys = np.zeros((num_sequences, 1), dtype=np.float32)
    for i in range(num_sequences):
        xs[i] = features[i:i + seq_len]
        ys[i] = features[i + seq_len, 1]

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
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x)
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

dfv = pd.read_csv("NIST_cleaned.csv").values
train_size = int(len(dfv) * 0.8)
train_data = dfv[:train_size]
test_data = dfv[train_size:]

training_features = train_data[:, 1:]
training_xs, training_ys = create_sequences(training_features, seq_len)
training_loader = data.DataLoader(data.TensorDataset(training_xs, training_ys), batch_size=batch_size, drop_last=True, num_workers=8)

test_features = test_data[:, 1:]
test_xs, test_ys = create_sequences(test_features, seq_len)
test_loader = data.DataLoader(data.TensorDataset(test_xs, test_ys), batch_size=batch_size, drop_last=True, num_workers=8)
timestamps = test_data[:, 0]

trainer = LSTMTrainer(model, learning_rate, timestamps)
lightning_trainer = L.Trainer(max_epochs=epochs)
lightning_trainer.fit(trainer, training_loader)
lightning_trainer.test(trainer, test_loader)
