import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_preprocess.timeseries_dataset import TimeSeriesDataset
from src.util.error import NRMSE
from src.util.constants import NUM_WORKERS, TARGET_COLUMN

INPUT_SIZE = 3

class BaseModel(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int, train_data, val_data, test_data):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.horizon_len = model.get_horizon_len()
        self.batch_size = batch_size
        self.all_predictions: list[float] = []
        self.all_actuals: list[float] = []
        
        self.setup_data_loaders(train_data, val_data, test_data)
    
    def setup_data_loaders(self, train_data, val_data, test_data):
        train_dataset = TimeSeriesDataset(train_data, self.seq_len, self.horizon_len, TARGET_COLUMN)
        val_dataset = TimeSeriesDataset(val_data, self.seq_len, self.horizon_len, TARGET_COLUMN)
        test_dataset = TimeSeriesDataset(test_data, self.seq_len, self.horizon_len, TARGET_COLUMN)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
        
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = NRMSE(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = NRMSE(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def test_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = NRMSE(y_hat, y)
        self.log('test_loss', loss, on_step=True, logger=True, prog_bar=True)
        
        self.all_predictions.extend(y_hat.detach().cpu().numpy().flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
    
    def get_predictions(self):
        return self.all_predictions
    
    def get_actuals(self):
        return self.all_actuals
    
    def forward(self, batch):
        x = batch
        return self.model(x)
        
class ProbabilisticBaseModel(BaseModel):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int, train_data, val_data, test_data):
        super().__init__(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data)
        self.all_predictions: tuple[list[float], list[float]] = ([], []) # type: ignore
