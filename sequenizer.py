import torch
import numpy as np
from device import device

def create_sequences(features: np.ndarray, seq_len: int):
    num_sequences = len(features) - seq_len
    num_features = features.shape[1]

    xs = np.zeros((num_sequences, seq_len, num_features), dtype=np.float32)
    ys = np.zeros((num_sequences, 1), dtype=np.float32)
    for i in range(num_sequences):
        xs[i] = features[i:i + seq_len]
        ys[i] = features[i + seq_len, 1]

    return torch.tensor(xs).to(device), torch.tensor(ys).to(device)