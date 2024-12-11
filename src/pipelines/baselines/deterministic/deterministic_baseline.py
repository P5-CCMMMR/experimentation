import numpy as np
import torch
from src.pipelines.pipeline import Pipeline
from src.pipelines.deterministic_pipeline import DeterministicPipeline
from abc import ABC, abstractmethod

class DeterministicBaseline(DeterministicPipeline, ABC):
    def __init__(self, target_column, test_loader, test_timestamps, test_normalizer, test_error_func_arr, horizon_len):
        super().__init__(None, None, None, None, None, None, None, None, None, test_loader, test_timestamps, test_normalizer, None, None, test_error_func_arr, target_column)
        self.horizen_len = horizon_len

    def training_step(self, batch):
        raise NotImplementedError("Training_step not meant to be used for deterministic baseline")
    
    def validation_step(self, batch):
        raise NotImplementedError("Validation_step not meant to be used for deterministic baseline")

    def test_step(self, batch):
        raise NotImplementedError("Test_step not meant to be used for deterministic baseline")
    
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

    def test(self):
        loss_dict = {}

        for batch in self.test_loader:
            x, y = batch
            self.all_predictions.append(self.forward(x.detach().cpu().numpy()))
            self.all_actuals.append(y.detach().cpu().numpy().flatten())

        self.all_predictions = self.normalizer.denormalize(np.array(self.all_predictions), self.target_column)
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)

        func_arr = self.test_error_func_arr
        for func in func_arr:
            loss = func.calc(torch.tensor(self.all_predictions), torch.tensor(self.all_actuals))
            loss_dict[func.get_key()] = loss
            title = func.get_title()
            print(f"{title:<30} {loss:.6f}")
        
        return loss_dict

    @abstractmethod
    def forward(self, x):
        pass
 
    class Builder(DeterministicPipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = DeterministicBaseline

        def set_horizon_len(self, horizon_len):
            self.horizon_len = horizon_len
            return self
        
        def build(self):    
            _, _, test_loader, test_timestamps, test_normalizer = self._get_loaders(self.horizon_len)
            return self.pipeline_class(self.target_column, test_loader, test_timestamps, test_normalizer, self.test_error_func_arr, self.horizon_len)
        
    


