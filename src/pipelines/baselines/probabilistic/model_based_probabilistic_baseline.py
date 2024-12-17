import numpy as np
import torch
from src.pipelines.pipeline import Pipeline
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.trainers.trainerWrapper import TrainerWrapper

class ModelBasedProbabilisticBaseline(Pipeline):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                 optimizer: torch.optim.Optimizer, model: nn.Module, trainer_wrapper: TrainerWrapper,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                 train_error_func, val_error_func, test_error_func_arr,
                 target_column: int, test_power, test_outdoor, use_tuner: bool,
                 penalty_strat, horizon_len):
        super().__init__(learning_rate, seq_len, batch_size,
                 optimizer, model, trainer_wrapper,
                 train_loader, val_loader, test_loader,
                 test_timesteps, normalizer,
                 train_error_func, val_error_func, test_error_func_arr,
                 target_column, test_power, test_outdoor, use_tuner)
        self.all_predictions: tuple[list[float], list[float]] = ([], []) # type: ignore

        self.horizon_len = horizon_len
        self.penalty_strat = penalty_strat

        self.reset_forward_memory()

    def training_step(self, batch):
        raise NotImplementedError("Training_step not meant to be used for deterministic baseline")
    
    def validation_step(self, batch):
        raise NotImplementedError("Validation_step not meant to be used for deterministic baseline")
    
    def copy(self):
        raise NotImplementedError("Copy not meant to be used for deterministic baseline")
    
    def get_validation_loss(self):
        raise NotImplementedError("Get_validation_loss not meant to be used for deterministic baseline")
    
    def get_training_loss(self):
        raise NotImplementedError("Get_training_loss not meant to be used for deterministic baseline")

    def forward(self, x):
        return_mean_arr = []
        return_std_dev_arr = []
        
        for input in x:
            prediction_arr = self.forward_prediction_2d_arr[self.step_start_index]
            prediction_arr_length = len(prediction_arr)
        
            # update resduals
            residual_sum = 0
            if prediction_arr_length > 0:
                for i in range(0, self.horizon_len):
                    before_index = self.horizon_len - i
                    current_prediction_temp = prediction_arr[prediction_arr_length -  before_index]
                    current_actual_temp = input[len(input) -  before_index][self.target_column]
                    residual = (current_actual_temp - current_prediction_temp) ** 2
                    residual_sum += residual

                self.error_arr[self.step_start_index] += residual_sum.item()
 
            # make return arrays
            mean_arr = []
            std_dev_arr = []

            T = prediction_arr_length 
    
            predictions = self.model(x).squeeze().cpu().detach().numpy()
        
            sigma = 0 if T == 0 else np.sqrt(self.error_arr[self.step_start_index] / T) 
            for i in range(0, self.horizon_len):
                std_dev = sigma * self.penalty_strat.calc(i + 1, T)
                mean_arr.append(predictions[i])
                std_dev_arr.append(std_dev)

            # Prepare for next forward and next forward in this step
            self.forward_prediction_2d_arr[self.step_start_index].extend(mean_arr)
            self.step_start_index = (self.step_start_index + 1) % self.horizon_len

            return_mean_arr.append(mean_arr), return_std_dev_arr.append(std_dev_arr)
        return np.array(return_mean_arr[0]), np.array(return_std_dev_arr[0]) # This is sus but works for now so cba

    def reset_forward_memory(self):
        self.step_start_index = 0
        self.error_arr = [0] * self.horizen_len
        self.forward_prediction_2d_arr = []
        for _ in range(0, self.horizen_len):
            self.forward_prediction_2d_arr.append([])

    def test_step(self, batch):
        x, y = batch
        mean_prediction, std_prediction = self.forward(x)
        self.all_predictions[0].extend(mean_prediction.flatten())
        self.all_predictions[1].extend(std_prediction.flatten())
        self.all_actuals.extend(y.detach().cpu().numpy().flatten())

    
    def test(self):
        results = {}
        self.trainer.test(self)

        mean, stddev = self.all_predictions

        for func in self.test_error_func_arr:
            if func.is_deterministic():
                loss = func.calc(torch.tensor(self.all_predictions[0]), torch.tensor(self.all_actuals))
            if func.is_probabilistic():
                loss = func.calc(torch.tensor(self.all_predictions[0]), torch.tensor(self.all_predictions[1]), torch.tensor(self.all_actuals))
            results[func.get_key()] = loss.item()
            title = func.get_title()
            print(f"{title:<30} {loss:.6f}")


        self.all_predictions = (self.normalizer.denormalize(np.array(mean), self.target_column),
                                np.array(stddev) * (self.normalizer.max_vals[self.target_column] - self.normalizer.min_vals[self.target_column]))
        
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)
        
        return results
    
    class Builder(Pipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = ModelBasedProbabilisticBaseline

        def set_error(self, error_func):
            self.train_error_func = error_func
            self.val_error_func = error_func
            self.add_test_error(error_func)
            return self

        def set_train_error(self, error_func):
            self.train_error_func = error_func
            return self

        def set_val_error(self, error_func):
            self.val_error_func = error_func
            return self
        
        def set_horizon_len(self, horizon_len):
            self.horizon_len = horizon_len
            return self
        
        def set_penalty_strat(self, penalty_strat):
            self.penalty_strat = penalty_strat
            return self
        
        def add_test_error(self, error_func):
            self.test_error_func_arr.append(error_func)
            return self
        
        def build(self):
            self._init_loaders()

            pipeline = self.pipeline_class(self.learning_rate,
                                          self.seq_len, 
                                          self.batch_size,
                                          self.optimizer,
                                          self.model,
                                          self.trainer_wrapper,
                                          self.train_loader,
                                          self.val_loader,
                                          self.test_loader,
                                          self.test_timestamps,
                                          self.test_normalizer,
                                          self.train_error_func,
                                          self.val_error_func,
                                          self.test_error_func_arr,
                                          self.target_column,
                                          self.test_power,
                                          self.test_outdoor,
                                          self.use_tuner,
                                          self.penalty_strat, 
                                          self.horizon_len)
            return pipeline
        
