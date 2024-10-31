from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.tuners.tuner_wrapper import TunerWrapper
from src.pipelines.pipeline import Pipeline

from torch.utils.data import DataLoader


class EnsemblePipeline(Pipeline):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                optimizer: torch.optim.Optimizer, model: nn.Module, trainer: L.Trainer,
                tuner_class: TunerWrapper,
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                train_error_func, val_error_func, test_error_func,
                pipeline_arr: list, num_ensembles: int,
                target_column: int):
        super().__init__(learning_rate, seq_len, batch_size,
                            optimizer, model, trainer,
                            tuner_class,
                            train_loader, val_loader, test_loader,
                            test_timesteps, normalizer,
                            train_error_func, val_error_func, test_error_func,
                            target_column)
        self.pipeline_arr = pipeline_arr
        self.num_ensembles = num_ensembles


    def training_step(self, batch):
        raise NotImplementedError("Training_step not Meant to be used for ensemble")
    
    def validation_step(self, batch):
        raise NotImplementedError("Validation_step not Meant to be used for ensemble")

    def test_step(self, batch):
        raise NotImplementedError("Test_step not Meant to be used for ensemble")
    
    def fit(self): 
        with ThreadPoolExecutor(max_workers=self.num_ensembles) as executor:
            futures = [executor.submit(pipeline.fit) for pipeline in self.pipeline_arr]
            for future in as_completed(futures):
                future.result()
    
    def test(self):
        with ThreadPoolExecutor(max_workers=self.num_ensembles) as executor:
            futures = [executor.submit(pipeline.test) for pipeline in self.pipeline_arr]
            for future in as_completed(futures):
                future.result()

        self.all_actuals = self.pipeline_arr[0].get_actuals()
        for pipeline in self.pipeline_arr:
            self.all_predictions.append(pipeline.get_predictions())

        if (isinstance(self.all_predictions[0], tuple)):
            self.all_predictions = self._ensemble_probabilistic_predictions(self.all_predictions)
        else:
            self.all_predictions = self._ensemble_deterministic_predictions(self.all_predictions)
        
    def forward(self, x):
        predictions = []

        with ThreadPoolExecutor(max_workers=self.num_ensembles) as executor:
            futures = [executor.submit(pipeline.forward, x) for pipeline in self.pipeline_arr]
            for future in as_completed(futures):
                predictions.append(future.result())

        if (isinstance(predictions[0], tuple)):
            predictions = self._ensemble_probabilistic_predictions(predictions)
        else:
            predictions = self._ensemble_deterministic_predictions(predictions)
        return predictions 
            
    def _ensemble_probabilistic_predictions(self, predictions):
        mean_predictions = []
        std_predictions = []
        
        for i in range(len(predictions[0][0])):
            mean_row = []
            std_row = []
            
            for j in range(len(predictions)):
                mean_row.append(predictions[j][0][i])
                std_row.append(predictions[j][1][i])
        
            mean_mixture = np.mean(mean_row)
            std_mixture = np.sqrt(np.sum([n**2 for n in std_row] + [n**2 for n in mean_row]) / len(std_row) - mean_mixture**2)
            
            mean_predictions.append(mean_mixture)
            std_predictions.append(std_mixture)
            
        return mean_predictions, std_predictions

    def _ensemble_deterministic_predictions(self, predictions):
        mean_predictions = []
        std_predictions = []

        for i in range(len(predictions[0])):
            row = []
            for j in range(len(predictions)):
                row.append(float(predictions[j][i]))
                
            mean_prediction = np.mean(row)
            std_prediction = np.std(row)
            
            mean_predictions.append(mean_prediction)
            std_predictions.append(std_prediction)
            
        return mean_predictions, std_predictions

 
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = EnsemblePipeline
            self.sub_pipeline_class = None
            self.pipeline_arr = []

        def set_num_ensembles(self, num_ensembles):
            self.num_ensembles = num_ensembles
            return self
        
        def set_pipeline_class(self, sub_pipeline_class):
            if not issubclass(sub_pipeline_class, Pipeline):
                raise ValueError("Pipeline sub class given not extended from Pipeline class")
            self.sub_pipeline_class = sub_pipeline_class
            return self
        
        def Build(self):
            train_loader, val_loader, test_loader, test_timestamps, test_normalizer = self._get_loaders()

            for _ in range(0, self.num_ensembles):
                self.pipeline_arr.append(self.sub_pipeline_class(self.learning_rate,
                                                                 self.seq_len, 
                                                                 self.batch_size,
                                                                 self.optimizer,
                                                                 self.model,
                                                                 copy.deepcopy(self.trainer),
                                                                 self.tuner_class,
                                                                 train_loader,
                                                                 val_loader,
                                                                 test_loader,
                                                                 test_timestamps,
                                                                 test_normalizer,
                                                                 self.train_error_func,
                                                                 self.val_error_func,
                                                                 self.test_error_func,
                                                                 self.target_column))
                         
            return self.pipeline_class(self.learning_rate,
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
                                       self.pipeline_arr,
                                       self.num_ensembles,
                                       self.target_column)
        
    


