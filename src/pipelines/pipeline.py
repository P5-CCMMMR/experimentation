import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.pipelines.cleaners.cleaner import Cleaner
from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.splitters.splitter import Splitter
from src.pipelines.sequencers.sequencer import Sequencer
from src.pipelines.models.model import Model
from src.pipelines.handlers.handler import Handler
from src.util.plot import plot_results
from src.pipelines.tuners.tuner_wrapper import TunerWrapper

from src.util.error import RMSE

# TODO 
# [x] 1. Get the basic thing running
# [x] 2. Use proper builder pattern
# [x] 3. Stop using constants
# [x] 4. Find a solution for modulating optimizers
# [ ] 5. Make train, test & val steps into modules
# [ ] 6. Make ensemble work
# [ ] 7. Make it so normalization is done after split so that there is no connection between the different parts
# [ ] 8. figure out better method for plotting

class Pipeline: # AKA Handler Builder
    def __init__(self):
        self.model_arr = []
        self.df_arr = []

        self.worker_num = 1

        self.optimizer = None
        self.cleaner = None
        self.splitter = None
        self.normalizer_class = None
        self.model = None
        self.handler_class = None
        self.trainer = None
        self.tuner_class = None
        self.batch_size = None
        self.seq_len = None
        

    def add_data(self, df):
        self.df_arr.append(df)
        return self

    def set_clean(self, cleaner):
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

    def set_handler_class(self, handler_class):
        if not issubclass(handler_class, Handler):
            raise ValueError("Handler sub class given not extended from Handler class")
        self.handler_class = handler_class
        return self

    def set_optimizer(self, optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("Optimizer instance given not extended from torch.optim class")

        self.optimizer = optimizer
        return self

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
        return self
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    
    def set_worker_num(self, worker_num):
        self.worker_num = worker_num
        return self
    
    def set_target_column(self, target_column):
        self.target_column = target_column
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

    def run(self):
        for df in self.df_arr:
            df = self.cleaner.clean(df)

            train_df = self.splitter.get_train(df)
            val_df = self.splitter.get_val(df)
            test_df = self.splitter.get_test(df)

            test_timestamps = pd.to_datetime(test_df.values[:,0])

            train_df = self.splitter.get_train(df)
            val_df = self.splitter.get_val(df)
            test_df = self.splitter.get_test(df)

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

            builder = self.handler_class.Builder() \
                .set_model(self.model.deep_copy()) \
                .set_seq_len(self.seq_len) \
                .set_batch_size(self.batch_size) \
                .set_train_dataloader(train_loader) \
                .set_val_dataloader(val_loader) \
                .set_test_dataloader(test_loader) \
                .set_error(RMSE) \
                
            if (self.optimizer is not None): 
                builder.set_optimizer(self.optimizer)

            handler = builder.build()
            
            tuner = self.tuner_class(self.trainer, handler)
            tuner.tune()

            self.trainer.fit(handler)
            self.trainer.test(handler)

            plot_results([handler.get_predictions()], 
                         handler.get_actuals(), 
                         test_timestamps, 
                         test_normalizer.get_min_vals(),  
                         test_normalizer.get_max_vals())
            
            self.handler_arr.append(handler)

        return self.handler_arr
