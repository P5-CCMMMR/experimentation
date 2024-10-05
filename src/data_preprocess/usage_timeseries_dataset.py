import torch
from torch.utils.data import Dataset

class UsageTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len: int):
        self.data = data
        self.seq_length = seq_len

        # Ensure there is enough data
        if len(self.data) < self.seq_length:
            raise ValueError("Data length must be greater than or equal to sequence length")


    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        return torch.tensor(x, dtype=torch.float32)
