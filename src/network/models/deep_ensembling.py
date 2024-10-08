import torch
import torch.nn as nn
import numpy as np
from src.network.models.base_model import ProbabilisticBaseModel
from src.util.error import RMSE

class DeepEnsemblingModel(ProbabilisticBaseModel):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int, train_data, val_data, test_data, num_models: int = 5):
        super().__init__(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data)
        self.num_models = num_models
        self.models = [model for _ in range(num_models)]
            
    def training_step(self, batch):
        x, y = batch
        total_loss = 0
        
        for model in self.models:
            y_hat = model(x)
            loss = RMSE(y_hat, y)
            total_loss += loss
        
        avg_loss = total_loss / self.num_models
        self.log('train_loss', avg_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return avg_loss
    
    def validation_step(self, batch):
        x, y = batch
        mean_prediction, _ = self.__predict_with_deep_ensembling(x)
        loss = RMSE(torch.tensor(mean_prediction, device=y.device), y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss
  
    def test_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.__predict_with_deep_ensembling(x)
        loss = RMSE(torch.tensor(mean_prediction, device=y.device), y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)
        
        self.all_predictions[0].extend(mean_prediction.flatten())
        self.all_predictions[1].extend(std_prediction.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
        
    def __predict_with_deep_ensembling(self, x):
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                y_hat = model(x)
                predictions.append(y_hat.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
        