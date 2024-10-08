import lightning as L
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.util.error import RMSE
from src.network.models.base_model import ProbabilisticBaseModel
from src.util.constants import TARGET_COLUMN

class MCModel(ProbabilisticBaseModel):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int, train_data, val_data, test_data, test_sample_nbr: int):
        super().__init__(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data)
        self.test_sample_nbr = test_sample_nbr

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
    
    def plot_results(self, timestamps, min_vals, max_vals):
        writer = SummaryWriter()

        mean_predictions, std_predictions = self.all_predictions
        mean_predictions = np.array(mean_predictions)
        std_predictions = np.array(std_predictions)
        actuals = np.array(self.all_actuals)

        # Rescale predictions and actuals
        mean_predictions = mean_predictions * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
        std_predictions = std_predictions * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN])
        actuals = actuals * (max_vals[TARGET_COLUMN] - min_vals[TARGET_COLUMN]) + min_vals[TARGET_COLUMN]
        timestamps = timestamps[:len(mean_predictions)]

        plt.plot(timestamps, mean_predictions, label="Prediction")
        plt.plot(timestamps, actuals, label="Actual")

        # Uncertainty bands
        plt.fill_between(timestamps, 
                        mean_predictions - 3 * std_predictions, 
                        mean_predictions + 3 * std_predictions, 
                        color='gray', alpha=0.2, label='3σ')
        
        plt.fill_between(timestamps,
                        mean_predictions - 2 * std_predictions,
                        mean_predictions + 2 * std_predictions,
                        color='gray', alpha=0.5, label='2σ')
        
        plt.fill_between(timestamps,
                        mean_predictions - std_predictions,
                        mean_predictions + std_predictions,
                        color='gray', alpha=0.8, label='σ')

        plt.xlabel("Time")
        plt.ylabel("Indoor Temperature")
        plt.title("Predictions vs Actuals with Uncertainty")
        plt.legend()
        plt.grid()
        plt.gcf().autofmt_xdate()

        # Save plot to TensorBoard
        writer.add_figure("predictions", plt.gcf())
        writer.close()
    