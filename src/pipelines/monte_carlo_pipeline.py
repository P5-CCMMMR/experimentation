import copy
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.trainers.trainerWrapper import TrainerWrapper
from src.pipelines.tuners.tuner_wrapper import TunerWrapper
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline

class MonteCarloPipeline(ProbabilisticPipeline):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                    optimizer: torch.optim.Optimizer, model: nn.Module, trainer_wrapper: TrainerWrapper,
                    tuner_class: TunerWrapper,
                    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                    test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                    train_error_func, val_error_func, test_error_func,
                    target_column: int,
                    inference_samples: int):
        super().__init__(learning_rate, seq_len, batch_size,
                            optimizer, model, trainer_wrapper,
                            tuner_class,
                            train_loader, val_loader, test_loader,
                            test_timesteps, normalizer,
                            train_error_func, val_error_func, test_error_func,
                            target_column)
        self.inference_samples = inference_samples

    def forward(self, x):
        self.model.train()
        predictions = []

        with torch.no_grad():
            for _ in range(self.inference_samples):
                y_hat = self.model(x)
                predictions.append(y_hat.cpu().numpy())

        predictions = np.array(predictions).squeeze()
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        return mean_prediction, std_prediction
    
    def copy(self):
        new_model = self.model.copy()
        new_optimizer = self.optimizer.copy(new_model)
        new_trainer_wrapper = self.trainer_wrapper.copy()

        new_instance = MonteCarloPipeline(
            learning_rate=self.learning_rate,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            optimizer=new_optimizer,
            model=new_model,
            trainer_wrapper=new_trainer_wrapper,
            tuner_class=self.tuner_class,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            test_timesteps=self.timesteps,
            normalizer=self.normalizer,
            train_error_func=self.train_error_func,
            val_error_func=self.val_error_func,
            test_error_func=self.test_error_func,
            target_column=self.target_column,
            inference_samples=self.inference_samples
        )
        return new_instance
    
    

    class Builder(ProbabilisticPipeline.Builder):
        def __init__(self):
            super().__init__()
            self.inference_samples = 0
            self.pipeline_class = MonteCarloPipeline

        def set_inference_samples(self, inference_samples):
            self.inference_samples = inference_samples
            return self
        
        def build(self):
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
                                          self.test_error_func,
                                          self.target_column,
                                          self.inference_samples)
            return pipeline
        
    
