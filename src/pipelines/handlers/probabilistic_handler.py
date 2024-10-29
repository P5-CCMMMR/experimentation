from abc import ABC
import torch.nn as nn
from .handler import Handler


class ProbabilisticHandler(Handler, ABC):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int,
                 optimizer_list: list, train_loader, val_loader, test_loader,
                 train_error_func, val_error_func, test_error_func):
        super().__init__(model, learning_rate, seq_len, batch_size,
                 optimizer_list, train_loader, val_loader, test_loader,
                 train_error_func, val_error_func, test_error_func)
        self.all_predictions: tuple[list[float], list[float]] = ([], []) # type: ignore