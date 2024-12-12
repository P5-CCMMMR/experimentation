import numpy as np
import torch
from src.pipelines.pipeline import Pipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline
from abc import ABC, abstractmethod

class ProbabilisticBaseline(ProbabilisticPipeline, ABC):
    def __init__(self, penalty_strat, target_column, test_loader, test_timestamps, test_normalizer, test_error_func_arr, horizon_len):
        super().__init__(None, None, None, None, None, None, None, None, test_loader, test_timestamps, test_normalizer, None, None, test_error_func_arr, target_column, False)

        self.horizen_len = horizon_len
        self.penalty_strat = penalty_strat

        self.reset_forward_memory()

    def reset_forward_memory(self):
        self.step_start_index = 0
        self.error_arr = [0] * self.horizen_len
        self.forward_prediction_2d_arr = []
        for _ in range(0, self.horizen_len):
            self.forward_prediction_2d_arr.append([])

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
    
    def save(self, path):
        raise NotImplementedError("Save not meant to be used for deterministic baseline")

    def load(self, path):
        raise NotImplementedError("Load not meant to be used for deterministic baseline")
    
    def fit(self):
        raise NotImplementedError("Fit not meant to be used for deterministic baseline")

    def test_step(self, batch):
        for batch in self.test_loader:
            x, y = batch
            mean_prediction, std_prediction = self.forward(x)
            self.all_predictions[0].extend(mean_prediction.flatten())
            self.all_predictions[1].extend(std_prediction.flatten())
            self.all_actuals.extend(y.detach().cpu().numpy().flatten())


    def test(self):
        loss_dict = {}

        for batch in self.test_loader:
            x, y = batch
            mean_prediction, std_prediction = self.forward(x)
            self.all_predictions[0].extend(mean_prediction.flatten())
            self.all_predictions[1].extend(std_prediction.flatten())
            self.all_actuals.extend(y.detach().cpu().numpy().flatten())

        mean, stddev = self.all_predictions

        func_arr = self.test_error_func_arr
        for func in func_arr:
            if func.is_deterministic():
                loss = func.calc(torch.tensor(self.all_predictions[0], device=y.device), torch.tensor(self.all_actuals))
            if func.is_probabilistic():
                loss = func.calc(torch.tensor(self.all_predictions[0], device=y.device), torch.tensor(self.all_predictions[1], device=y.device), torch.tensor(self.all_actuals))
            
            loss_dict[func.get_key()] = loss
            title = func.get_title()
            print(f"{title:<30} {loss:.6f}")

        self.all_predictions = (self.normalizer.denormalize(np.array(mean), self.target_column),
                                np.array(stddev) * (self.normalizer.max_vals[self.target_column] - self.normalizer.min_vals[self.target_column]))
        
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)
        
        return loss_dict

    @abstractmethod
    def forward(self, x):
        pass
 
    class Builder(ProbabilisticPipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = ProbabilisticBaseline

        def set_horizon_len(self, horizon_len):
            self.horizon_len = horizon_len
            return self
        
        def set_penalty_strat(self, penalty_strat):
            self.penalty_strat = penalty_strat
            return self
        
        def build(self):    
            self._init_loaders(self.horizon_len)
            return self.pipeline_class(self.penalty_strat, self.target_column, self.test_loader, self.test_timestamps, self.test_normalizer, self.test_error_func_arr, self.horizon_len)
        
    


