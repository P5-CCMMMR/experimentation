import lightning as L
import torch
import torch.nn as nn
from src.util.error import RMSE

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
