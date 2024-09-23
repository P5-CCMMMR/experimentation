import lightning as L
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

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
