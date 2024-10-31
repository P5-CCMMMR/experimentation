import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from src.pipelines.cleaners.cleaner import Cleaner
from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.splitters.splitter import Splitter
from src.pipelines.sequencers.sequencer import Sequencer
from src.pipelines.models.model import Model
from src.pipelines.tuners.tuner_wrapper import TunerWrapper

class Pipeline(L.LightningModule, ABC):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                 optimizer: torch.optim.Optimizer, model: nn.Module, trainer: L.Trainer,
                 tuner_class: TunerWrapper,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                 train_error_func, val_error_func, test_error_func,
                 target_column):
        super().__init__()
        self.seq_len = seq_len
        self.horizon_len = model.get_horizon_len()
        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.timesteps = test_timesteps

        self.train_error_func = train_error_func
        self.val_error_func = val_error_func
        self.test_error_func = test_error_func

        self.optimizer = optimizer
        self.normalizer = normalizer
        self.model = model
        self.trainer = trainer

        self.tuner_class = tuner_class
        self.tuner = None

        self.target_column = target_column

        self.df_arr = []

        self.all_predictions = []
        self.all_actuals = []

    @abstractmethod
    def training_step(self, batch):
        pass
  
    @abstractmethod
    def validation_step(self, batch):
        pass

    @abstractmethod
    def test_step(self, batch):
        pass

    def fit(self): #Cancer train keyword taken by L.module
        self.tuner = self.tuner_class(self.trainer, self)
        self.tuner.tune()
        self.trainer.fit(self)

    def test(self):
        if self.tuner is None:
            raise RuntimeError("Need to train before testing")
        self.trainer.test(self)

        self.all_predictions = self.normalizer.denormalize(np.array(self.all_predictions), self.target_column)
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return self.optimizer

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
            self.test_error_func = None

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
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise ValueError("Optimizer instance given not extended from torch.optim class")

            self.optimizer = optimizer
            return self
        
        def set_trainer(self, trainer):
            if not isinstance(trainer, L.Trainer):
                raise ValueError("Trainer instance given not extended from Trainer class")
            self.trainer = trainer
            return self

        def set_tuner(self, tuner_class):
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
                
        def _get_loaders(self):
            # implement 
            #for df in self.df_arr:
            df = self.cleaner.clean(self.df_arr[0]) #! needs to be changed to a solution concatting the dataset

            train_df = self.splitter.get_train(df)
            val_df = self.splitter.get_val(df)
            test_df = self.splitter.get_test(df)

            test_timestamps = pd.to_datetime(test_df.values[:,0])

            train_normalizer = self.normalizer_class(train_df.values[:,1:].astype(float)) 
            val_normalizer = self.normalizer_class(val_df.values[:,1:].astype(float)) 
            test_normalizer = self.normalizer_class(test_df.values[:,1:].astype(float)) 

            train_df = train_normalizer.normalize()
            val_df = val_normalizer.normalize()
            test_df = test_normalizer.normalize()

            horizon_len = self.model.get_horizon_len()

            train_segmenter = self.sequencer_class(train_df[0], self.seq_len, horizon_len, self.target_column)
            val_segmenter = self.sequencer_class(val_df[0], self.seq_len, horizon_len, self.target_column)
            test_segmenter = self.sequencer_class(test_df[0], self.seq_len, horizon_len, self.target_column)

            train_loader = DataLoader(train_segmenter, batch_size=self.batch_size, num_workers=self.worker_num)
            val_loader = DataLoader(val_segmenter, batch_size=self.batch_size, num_workers=self.worker_num)
            test_loader = DataLoader(test_segmenter, batch_size=self.batch_size, num_workers=self.worker_num)
            
            return train_loader, val_loader, test_loader, test_timestamps, test_normalizer

        def Build(self):
            train_loader, val_loader, test_loader, test_timestamps, test_normalizer = self._get_loaders()

            pipeline = self.pipeline_class(self.learning_rate,
                                          self.seq_len, 
                                          self.batch_size,
                                          self.optimizer,
                                          self.model,
                                          self.trainer,
                                          self.tuner_class,
                                          train_loader,
                                          val_loader,
                                          test_loader,
                                          test_timestamps,
                                          test_normalizer,
                                          self.train_error_func,
                                          self.val_error_func,
                                          self.test_error_func,
                                          self.target_column)

            return pipeline
    
