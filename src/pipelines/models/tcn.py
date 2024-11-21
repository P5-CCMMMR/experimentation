import torch.nn as nn
from pytorch_tcn import TCN as TCNModel
from .model import Model

class TCN(Model):
    def __init__(self, num_channels: list, kernel_size: int, input_len: int, horizon_len: int, dropout: float):
        super(TCN, self).__init__(input_len, horizon_len, dropout)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.tcn = TCNModel(self.input_len, num_channels, kernel_size=kernel_size, dropout=dropout, causal=True, input_shape='NLC')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1], horizon_len)
        
    def forward(self, x):
        out = self.tcn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
        
    def copy(self):
        return type(self) (
            self.num_channels, 
            self.kernel_size, 
            self.input_len,
            self.output_size, 
            self.dropout
        )