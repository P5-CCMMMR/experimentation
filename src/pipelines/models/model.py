from abc import ABC, abstractmethod
import torch.nn as nn

class Model(nn.Module, ABC):
    def __init__(self, input_len: int, horizon_len: int, dropout: float):
        super(Model, self).__init__()
        self.input_len = input_len
        self.output_size = horizon_len
        self.dropout = dropout

    @abstractmethod
    def forward(self, x):
        pass
    
    def get_horizon_len(self):
        return self.output_size