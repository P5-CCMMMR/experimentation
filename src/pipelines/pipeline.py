from concurrent.futures import ThreadPoolExecutor, as_completed
import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader
import copy

from src.pipelines.cleaners.cleaner import Cleaner
from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.splitters.splitter import Splitter
from src.pipelines.sequencers.sequencer import Sequencer
from src.pipelines.models.model import Model
from src.pipelines.handlers.handler import Handler
from src.util.plot import plot_results
from src.pipelines.tuners.tuner_wrapper import TunerWrapper

class Pipeline: # AKA Handler Builder
    def __init__(self):
        self.handler_arr = []
        self.df_arr = []

        self.worker_num = 1
        self.num_ensembles = 1
        self.inference_samples = 1
        self.batch_size = 1
        self.seq_len = 4

        self.optimizer = None
        self.cleaner = None
        self.splitter = None
        self.normalizer_class = None
        self.model = None
        self.handler_class = None
        self.trainer = None
        self.tuner_class = None

        self.train_error_func = None
        self.val_error_func = None
        self.test_error_func = None
        

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
    
    def set_inference_samples(self, inference_samples):
        self.inference_samples = inference_samples
        return self
    
    def set_num_ensembles(self, num_ensembles):
        self.num_ensembles = num_ensembles
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
            
            tuner_arr = []
            trainer_arr = []
            ensemble_handler_arr = []

            for _ in range(self.num_ensembles):
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
                    .set_train_error(self.train_error_func) \
                    .set_val_error(self.val_error_func) \
                    .set_test_error(self.test_error_func) \
                
                if (self.inference_samples is not None):
                    builder.set_inference_samples(self.inference_samples)

                if (self.optimizer is not None): 
                    builder.set_optimizer(self.optimizer)

                handler = builder.build()
                ensemble_handler_arr.append(handler)

                trainer = copy.deepcopy(self.trainer)
                trainer_arr.append(trainer)
                
                tuner = self.tuner_class(trainer, handler)
                tuner.tune()
                tuner_arr.append(tuner)

            with ThreadPoolExecutor(max_workers=min(self.num_ensembles, self.worker_num)) as executor:
                futures = [executor.submit(lambda trainer, lit_model: (trainer.fit(lit_model), trainer.test(lit_model)), trainer, lit_model) for trainer, lit_model in zip(trainer_arr, ensemble_handler_arr)]
                for future in as_completed(futures):
                    future.result()

            all_predictions = []     
            for handler in ensemble_handler_arr:
                all_predictions.append(handler.get_predictions()) 

            all_actuals = ensemble_handler_arr[0].get_actuals()

            self.handler_arr.append(ensemble_handler_arr)

            plot_results(all_predictions, 
                         all_actuals, 
                         test_timestamps, 
                         test_normalizer.get_min_vals(),  
                         test_normalizer.get_max_vals(),
                         self.target_column)
            

        return self.handler_arr
    
