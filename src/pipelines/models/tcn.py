import torch.nn as nn
from pytorch_tcn import TCN as TCNModel
from .model import Model
from math import ceil

class TCN(Model):
    def __init__(self, hidden_size: int, num_layers: int, input_len: int, horizon_len: int, dropout: float, seq_len: int):
        super(TCN, self).__init__(input_len, horizon_len, dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = dropout

        assert num_layers > 0, "Number of layers must be greater than 0"

        # Use same hidden size for all residual blocks
        num_channels = [hidden_size] * num_layers
        
        # Calculate kernel size based of num_layers and seq_len. 
        # Kernel size should be big enough so that the receptive field is at least as big as the sequence length
        kernel_size = ceil((2**(num_layers+1) + seq_len - 3) / (2**(num_layers+1) - 2))
        
        self.tcn = TCNModel(self.input_len, num_channels, kernel_size=kernel_size, dropout=dropout, causal=True, input_shape='NLC', use_skip_connections=True, activation='leaky_relu')
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon_len)
        
    def forward(self, x):
        out = self.tcn(x)
        out = self.dropout_layer(out[:, -1, :])
        return self.fc(out)
        
    def copy(self):
        return type(self) (
            self.hidden_size,
            self.num_layers, 
            self.input_len,
            self.output_size, 
            self.dropout,
            self.seq_len
        )