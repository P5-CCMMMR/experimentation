import lightning as L
import torch
import torch.nn as nn
import numpy as np
from src.util.error import RMSE, NRMSE

class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, test_sample_nbr: int):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.test_sample_nbr = test_sample_nbr
        self.all_predictions: tuple[list[float], list[float]] = ([], [])
        self.all_actuals: list[float] = []
    
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
    