import torch.nn as nn
from src.network.models.base_model import BaseModel

class DeepEnsemblingModel(BaseModel):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int, train_data, val_data, test_data, num_models: int):
        super().__init__(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data)
        self.num_models = num_models