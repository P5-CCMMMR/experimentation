import copy
import copy
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from src.pipelines.optimizers.optimizer import OptimizerWrapper
from src.pipelines.cleaners.cleaner import Cleaner
from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.splitters.splitter import Splitter
from src.pipelines.sequencers.sequencer import Sequencer
from src.pipelines.models.model import Model
from src.pipelines.trainers.trainerWrapper import TrainerWrapper
from src.pipelines.tuners.tuner_wrapper import TunerWrapper

class Pipeline(L.LightningModule, ABC):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                 optimizer: torch.optim.Optimizer, model: nn.Module, trainer_wrapper: TrainerWrapper,
                 tuner_class: TunerWrapper,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                 train_error_func, val_error_func, test_error_func_arr,
                 target_column):
        super().__init__()
        self.seq_len = seq_len

        self.batch_size = batch_size
        if model is not None:
            self.horizen_len = model.get_horizon_len()

        self.learning_rate = learning_rate

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.timesteps = test_timesteps

        self.train_error_func = train_error_func
        self.val_error_func = val_error_func
        self.test_error_func_arr = test_error_func_arr

        self.optimizer = optimizer
        self.normalizer = normalizer
        self.model = model
        self.trainer_wrapper = trainer_wrapper
        if trainer_wrapper is not None:
            self.trainer = trainer_wrapper.get_trainer()

        self.tuner_class = tuner_class
        self.tuner = None

        self.target_column = target_column

        self.all_predictions = []
        self.all_actuals = []

        self.val_loss_arr = []
        self.train_loss_arr = []

        self.epoch_train_loss_arr = []
        self.epoch_val_loss_arr = []

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.train_error_func.calc(y_hat, y)
        self.train_loss_arr.append(loss.cpu())
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.val_error_func.calc(y_hat, y)
        self.val_loss_arr.append(loss.cpu())
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_loss_arr) <= 0: return
        self.epoch_val_loss_arr.append(sum(self.val_loss_arr) / len(self.val_loss_arr))
        self.log('val_loss', (sum(self.val_loss_arr) / len(self.val_loss_arr)))
        self.val_loss_arr = []
        if len(self.train_loss_arr) <= 0: return
        self.epoch_train_loss_arr.append(sum(self.train_loss_arr) / len(self.train_loss_arr))
        self.train_loss_arr = []
    
    @abstractmethod
    def test_step(self, batch):
        pass

    def fit(self): # Cancer train keyword taken by L.module
        self.tuner = self.tuner_class(self.trainer, self)
        self.tuner.tune()
        self.trainer.fit(self)

    def test(self):
        self.trainer.test(self)

        self.all_predictions = self.normalizer.denormalize(np.array(self.all_predictions), self.target_column)
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)
    
    def forward(self, x):
        return self.model(x).squeeze()

    def configure_optimizers(self):
        return self.optimizer.get_optimizer()

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader

    def get_predictions(self):
        return self.all_predictions
    
    def get_actuals(self):
        return self.all_actuals
    
    def get_validation_loss(self):
        return self.epoch_val_loss_arr
    
    def get_training_loss(self):
        return self.epoch_train_loss_arr
    
    def get_timestamps(self):
        return self.timesteps
        
    class Builder:
        def __init__(self):
            self.learning_rate = 0.001
            self.seq_len = 4
            self.batch_size = 1

            self.train_loader = None
            self.val_loader = None
            self.test_loader = None

            self.train_error_func = None
            self.val_error_func = None
            self.test_error_func_arr = []

            self.normalizer_class = None
            self.tuner_class = None

            self.optimizer = None
            self.cleaner = None
            self.splitter = None
            self.model = None
            self.trainer = None

            self.df_arr = []
            self.pipeline_class = Pipeline

        def add_data(self, df):
            self.df_arr.append(df)
            return self
        
        def add_data_arr(self, df_arr):
            for df in df_arr:
                self.df_arr.append(df)
            return self
        
        def set_cleaner(self, cleaner):
            if not isinstance(cleaner, Cleaner):
                raise ValueError("Cleaner given not extended from Cleaner class")
            self.cleaner = cleaner
            return self
        
        def set_normalizer_class(self, normalizer_class):
            if not issubclass(normalizer_class, Normalizer):
                raise ValueError("Normalizer sub class given not extended from Normalizer class")
            self.normalizer_class = normalizer_class
            return self
           
        def set_splitter(self, splitter):
            if not isinstance(splitter, Splitter):
                raise ValueError("Splitter given not extended from Splitter class")
            self.splitter = splitter
            return self

        def set_sequencer_class(self, sequencer_class):
            if not issubclass(sequencer_class, Sequencer):
                raise ValueError("Sequencer sub class given not extended from Sequencer class")
            self.sequencer_class = sequencer_class
            return self

        def set_model(self, model):
            if not isinstance(model, Model):
                raise ValueError("Model instance given not extended from Model class")
            self.model = model
            return self

        def set_optimizer(self, optimizer):
            if not isinstance(optimizer, OptimizerWrapper):
                raise ValueError("Optimizer instance given not extended from torch.optim class")

            self.optimizer = optimizer
            return self
        
        def set_trainer(self, trainer_wrapper):
            if not isinstance(trainer_wrapper, TrainerWrapper):
                raise ValueError("TrainerWrapper instance given not extended from TrainerWrapper class")
            self.trainer_wrapper = trainer_wrapper
            return self

        def set_tuner_class(self, tuner_class):
            if not issubclass(tuner_class, TunerWrapper):
                raise ValueError("TunerWrapper sub class given not extended from TunerWrapper class")
            self.tuner_class = tuner_class
            return self

        def set_learning_rate(self, learning_rate: float):
            self.learning_rate = learning_rate
            return self

        def set_seq_len(self, seq_len: int):
            self.seq_len = seq_len
            return self
        
        def set_target_column(self, target_column):
            self.target_column = target_column
            return self

        def set_batch_size(self, batch_size: int):
            self.batch_size = batch_size
            return self
        
        def set_worker_num(self, worker_num):
            self.worker_num = worker_num
            return self

        def set_optimizer(self, optimizer: torch.optim.Optimizer):
            self.optimizer = optimizer
            return self
        
        @abstractmethod
        def set_error(self, error_func):
            pass
        
        @abstractmethod
        def set_train_error(self, error_func):
            pass

        @abstractmethod
        def set_val_error(self, error_func):
            pass
        
        @abstractmethod
        def add_test_error(self, error_func):
            pass

        def _check_none(self, **kwargs):
            for key, value in kwargs.items():
                if value is None:
                    raise ValueError(f"{key} cannot be None")
                
        def _get_loaders(self):
            train_dfs = []
            val_dfs = []
            test_dfs = []

            for df in self.df_arr:
                cleaned_df = self.cleaner.clean(df)
                train_dfs.append(self.splitter.get_train(cleaned_df))
                val_dfs.append(self.splitter.get_val(cleaned_df))
                test_dfs.append(self.splitter.get_test(cleaned_df))
                
            train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
            val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
            test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

            test_timestamps = pd.to_datetime(test_df.values[:,0]) if not test_df.empty else pd.DatetimeIndex([])

            if not train_df.empty:
                train_df.iloc[:, 0] = pd.to_datetime(train_df.iloc[:, 0]).astype(int) / 10**9
                train_normalizer = self.normalizer_class(train_df.values.astype(float)) 
                train_df = train_normalizer.normalize()
                train_segmenter = self.sequencer_class(train_df[0], self.seq_len, self.model.get_horizon_len(), self.target_column)
                train_loader = DataLoader(train_segmenter, batch_size=self.batch_size, num_workers=self.worker_num)
            else:
                train_loader = DataLoader([], batch_size=self.batch_size, num_workers=self.worker_num)
                train_normalizer = self.normalizer_class(np.array([]))

            if not val_df.empty:
                val_df.iloc[:, 0] = pd.to_datetime(val_df.iloc[:, 0]).astype(int) / 10**9
                val_normalizer = self.normalizer_class(val_df.values.astype(float)) 
                val_df = val_normalizer.normalize()
                val_segmenter = self.sequencer_class(val_df[0], self.seq_len, self.model.get_horizon_len(), self.target_column)
                val_loader = DataLoader(val_segmenter, batch_size=self.batch_size, num_workers=self.worker_num)
            else:
                val_loader = DataLoader([], batch_size=self.batch_size, num_workers=self.worker_num)
                val_normalizer = self.normalizer_class(np.array([]))

            if not test_df.empty:
                test_df.iloc[:, 0] = pd.to_datetime(test_df.iloc[:, 0]).astype(int) / 10**9
                test_normalizer = self.normalizer_class(test_df.values.astype(float)) 
                test_df = test_normalizer.normalize()
                test_segmenter = self.sequencer_class(test_df[0], self.seq_len, self.model.get_horizon_len(), self.target_column)
                test_loader = DataLoader(test_segmenter, batch_size=self.batch_size, num_workers=self.worker_num)
            else:
                test_loader = DataLoader([], batch_size=self.batch_size, num_workers=self.worker_num)
                test_normalizer = self.normalizer_class(np.array([]))
            
            return train_loader, val_loader, test_loader, test_timestamps, test_normalizer

        def build(self):
            self._check_none(trainer_wrapper=self.trainer_wrapper, model=self.model, optimizer=self.optimizer)
        
            train_loader, val_loader, test_loader, test_timestamps, test_normalizer = self._get_loaders()

            pipeline = self.pipeline_class(self.learning_rate,
                                          self.seq_len, 
                                          self.batch_size,
                                          self.optimizer,
                                          self.model,
                                          self.trainer_wrapper,
                                          self.tuner_class,
                                          train_loader,
                                          val_loader,
                                          test_loader,
                                          test_timestamps,
                                          test_normalizer,
                                          self.train_error_func,
                                          self.val_error_func,
                                          self.test_error_func_arr,
                                          self.target_column)

            return pipeline
    
