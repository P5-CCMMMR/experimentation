from abc import ABC, abstractmethod
import torch.nn as nn
import copy

class Model(nn.Module, ABC):
    def __init__(self, hidden_size: int, num_layers: int, input_len: int, horison_len: int, dropout: float):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = horison_len
        self.input_len = input_len
        self.dropout = dropout

    @abstractmethod
    def forward(self, x):
        pass
    
    def get_horizon_len(self):
        return self.output_size
    
    def deep_copy(self):
        return copy.deepcopy(self)