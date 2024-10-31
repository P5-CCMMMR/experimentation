import torch
from src.pipelines.wip_pipeline import Pipeline
import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.tuners.tuner_wrapper import TunerWrapper


class ProbabilisticPipeline(Pipeline):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                 optimizer: torch.optim.Optimizer, model: nn.Module, trainer: L.Trainer,
                 tuner_class: TunerWrapper,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                 train_error_func, val_error_func, test_error_func):
        super().__init__(learning_rate, seq_len, batch_size,
                 optimizer, model, trainer,
                 tuner_class,
                 train_loader, val_loader, test_loader,
                 test_timesteps, normalizer,
                 train_error_func, val_error_func, test_error_func)
        self.all_predictions: tuple[list[float], list[float]] = ([], []) # type: ignore

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.train_error_func(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.val_error_func(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.forward(x)
        loss = self.test_error_func(torch.tensor(mean_prediction, device=y.device), torch.tensor(std_prediction, device=y.device), y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)
        
        self.all_predictions[0].extend(mean_prediction.flatten())
        self.all_predictions[1].extend(std_prediction.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())
    
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = ProbabilisticPipeline
    
