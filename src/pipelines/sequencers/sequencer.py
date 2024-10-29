from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class Sequencer(ABC, Dataset):
    def __init__(self, data, seq_len: int, horizon_len: int, target_column: int):
        self.data = data
        self.seq_len = seq_len
        self.horizon_len = horizon_len
        self.target_column = target_column

    @abstractmethod
    def __len__(self):
        pass 

    @abstractmethod
    def __getitem__(self, idx):
        pass
