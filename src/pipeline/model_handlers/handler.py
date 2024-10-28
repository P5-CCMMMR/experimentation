import lightning as L
import torch.nn as nn
from abc import ABC, abstractmethod

class Handler(L.LightningModule, ABC):
    def __init__(self, model: nn.Module, learning_rate: float, seq_len: int, batch_size: int,
                 optimizer_list: list, train_loader, val_loader, test_loader,
                 train_error_func, val_error_func, test_error_func):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.horizon_len = model.get_horizon_len()
        self.batch_size = batch_size

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_error_func = train_error_func
        self.val_error_func = val_error_func
        self.test_error_func = test_error_func

        self.all_predictions = []
        self.all_actuals = []

        self.optimizer_list = optimizer_list

    @abstractmethod
    def training_step(self, batch):
        pass
  
    @abstractmethod
    def validation_step(self, batch):
        pass

    @abstractmethod
    def test_step(self, batch):
        pass

    class Builder:
        def __init__(self):
            self.model = None
            self.learning_rate = None
            self.seq_len = None
            self.batch_size = None
            self.optimizer_list = []
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None
            self.train_error_func = None
            self.val_error_func = None
            self.test_error_func = None

        def set_model(self, model: nn.Module):
            self.model = model
            return self

        def set_learning_rate(self, learning_rate: float):
            self.learning_rate = learning_rate
            return self

        def set_seq_len(self, seq_len: int):
            self.seq_len = seq_len
            return self

        def set_batch_size(self, batch_size: int):
            self.batch_size = batch_size
            return self

        def add_optimizer(self, optimizer):
            self.optimizer_list.append(optimizer)
            return self

        def set_train_dataloader(self, train_loader):
            self.train_loader = train_loader
            return self

        def set_val_dataloader(self, val_loader):
            self.val_loader = val_loader
            return self

        def set_test_dataloader(self, test_loader):
            self.test_loader = test_loader
            return self
        
        def set_error(self, error_func):
            self.train_error_func = error_func
            self.val_error_func = error_func
            self.test_error_func = error_func
            return self

        def set_train_error(self, error_func):
            self.train_error_func = error_func
            return self

        def set_val_error(self, error_func):
            self.val_error_func = error_func
            return self
        
        def set_test_error(self, error_func):
            self.test_error_func = error_func
            return self
        
        def _check_none(self, **kwargs):
            for key, value in kwargs.items():
                if value is None:
                    raise ValueError(f"{key} cannot be None")

        def build(self):
            self._check_none(
                model=self.model,
                learning_rate=self.learning_rate,
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                train_error_func=self.train_error_func,
                val_error_func=self.val_error_func,
                test_error_func=self.test_error_func
            )

            return Handler(
                model=self.model,
                learning_rate=self.learning_rate,
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                optimizer_list=self.optimizer_list,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                train_error_func=self.train_error_func,
                val_error_func=self.val_error_func,
                test_error_func=self.test_error_func
            )