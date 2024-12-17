import numpy as np
import torch
from src.pipelines.pipeline import Pipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline
from pytorch_lightning import seed_everything
import os

class EnsemblePipeline(ProbabilisticPipeline):
    def __init__(self, pipeline_arr, num_ensembles, horizon_len, test_error_func_arr, base_seed, 
                 target_column, test_loader, test_timesteps, normalizer, seed_multiplier=1, test_power=None, test_outdoor=None):
        super().__init__(None, None, None, None, None, None, None, None, test_loader, test_timesteps, normalizer, None, None, test_error_func_arr, target_column, test_power, test_outdoor, None)
        self.pipeline_arr = pipeline_arr
        self.num_ensembles = num_ensembles
        self.horizon_len = horizon_len
        self.base_seed = base_seed
        self.seed_multiplier = seed_multiplier

        self.timesteps = self.pipeline_arr[0].get_timestamps()

    def training_step(self, batch):
        raise NotImplementedError("Training_step not Meant to be used for ensemble")
    
    def validation_step(self, batch):
        raise NotImplementedError("Validation_step not Meant to be used for ensemble")

    def test_step(self, batch):
        raise NotImplementedError("Test_step not Meant to be used for ensemble")
    
    def copy(self):
        raise NotImplementedError("copy not Meant to be used for ensemble")
    
    def get_validation_loss(self):
        return [pipeline.get_validation_loss() for pipeline in self.pipeline_arr]
    
    def get_training_loss(self):
        return [pipeline.get_training_loss() for pipeline in self.pipeline_arr]
    
    def save(self, path):
        print(f"saving to {path}")
        os.makedirs(path, exist_ok=True)
        for i, pipeline in enumerate(self.pipeline_arr):
            torch.save(pipeline.state_dict(), f"{path}/sub_model{self._get_seed(i)}.pth")


    def load(self, path):
        for i, pipeline in enumerate(self.pipeline_arr):
            pipeline.load_state_dict(torch.load(f"{path}/sub_model{self._get_seed(i)}.pth", weights_only=True))
    
    def fit(self): 
        for i, pipeline in enumerate(self.pipeline_arr):
            if self.base_seed is not None:
                self._run_with_seed(pipeline.fit, self._get_seed(i))
            else:
                pipeline.fit()

    def test(self):
        results = {}

        for batch in self.test_loader:
            x, y = batch
            mean_prediction, std_prediction = self.forward(x)
            self.all_predictions[0].extend(mean_prediction)
            self.all_predictions[1].extend(std_prediction)
            self.all_actuals.extend(y.detach().cpu().numpy().flatten())

        mean, stddev = self.all_predictions

        func_arr = self.test_error_func_arr
        for func in func_arr:
            if func.is_deterministic():
                loss = func.calc(torch.tensor(self.all_predictions[0], device=y.device), torch.tensor(self.all_actuals))
            if func.is_probabilistic():
                loss = func.calc(torch.tensor(self.all_predictions[0], device=y.device), torch.tensor(self.all_predictions[1], device=y.device), torch.tensor(self.all_actuals))
            results[func.get_key()] = loss.item()
            title = func.get_title()
            print(f"{title:<30} {loss:.6f}")

        self.all_predictions = (self.normalizer.denormalize(np.array(mean), self.target_column),
                                np.array(stddev) * (self.normalizer.max_vals[self.target_column] - self.normalizer.min_vals[self.target_column]))
        
        self.all_actuals = self.normalizer.denormalize(np.array(self.all_actuals), self.target_column)

        return results
        
    def forward(self, x):
        predictions = []
        for i, pipeline in enumerate(self.pipeline_arr):
            if self.base_seed is not None:
                result = self._run_with_seed(pipeline.forward, self._get_seed(i), x)
            else:
                result = pipeline.forward(x)
                
            if isinstance(result, tuple):
                predictions.append(tuple(r.flatten() for r in result))
            else:
                predictions.append(result.flatten())

        if isinstance(predictions[0], tuple):
            predictions = self._ensemble_probabilistic_predictions(predictions)
        else:
            predictions = self._ensemble_deterministic_predictions(predictions)
        return predictions 

    def _run_with_seed(self, func, seed, *args, **kwargs):
        seed_everything(seed, workers=True, verbose=False)
        return func(*args, **kwargs)
     
    def _get_seed(self, idx):
        return self.base_seed + (idx * self.seed_multiplier)
            
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
    
    class Builder(ProbabilisticPipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = EnsemblePipeline
            self.pipeline_arr = []
            self.sub_pipeline = None
            self.num_ensembles = None
            self.base_seed = None

        def set_num_ensembles(self, num_ensembles):
            self.num_ensembles = num_ensembles
            
            if self.num_ensembles < 2:
                raise ValueError("Ensemble pipeline should have at least 2 models")
            return self
        
        def set_horizon_len(self, horizon_len):
            self.horizon_len = horizon_len
            return self

        def set_pipeline(self, sub_pipeline):
            if not isinstance(sub_pipeline, Pipeline):
                raise ValueError("Pipeline instance given not extended from Pipeline class")
            self.sub_pipeline = sub_pipeline
            return self
        
        def set_base_seed(self, base_seed):
            if not isinstance(base_seed, int):
                raise ValueError("Base seed should be an integer")
            self.base_seed = base_seed
            return self
        
        def set_seed_multiplier(self, seed_multiplier):
            if not isinstance(seed_multiplier, int):
                raise ValueError("Seed multiplier should be an integer")
            self.seed_multiplier = seed_multiplier
            return self
        
        
        def build(self):
            self._init_loaders(self.horizon_len)
            for _ in range(self.num_ensembles):
                self.pipeline_arr.append(self.sub_pipeline.copy())
                         
            return self.pipeline_class(pipeline_arr=self.pipeline_arr,
                                       target_column=self.target_column,
                                       test_loader=self.test_loader,
                                       test_timesteps=self.test_timestamps,
                                       normalizer=self.test_normalizer,
                                       num_ensembles=self.num_ensembles,
                                       horizon_len=self.horizon_len,
                                       test_error_func_arr=self.test_error_func_arr,
                                       base_seed=self.base_seed,
                                       seed_multiplier=self.seed_multiplier,
                                       test_power=self.test_power,
                                       test_outdoor=self.test_outdoor)
            