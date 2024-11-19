from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np
import torch
from src.pipelines.pipeline import Pipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline
import os

class EnsemblePipeline(ProbabilisticPipeline):
    def __init__(self, pipeline_arr, num_ensembles, horizon_len, test_error_func_arr):
        super().__init__(None, None, None, None, None, None, None, None, None, None, None, None, None, None, test_error_func_arr, None)
        self.pipeline_arr = pipeline_arr
        self.num_ensembles = num_ensembles
        self.horizen_len = horizon_len

        self.timesteps = self.pipeline_arr[0].get_timestamps()
        self.all_predictions = []
        self.all_actuals = []

    def training_step(self, batch):
        raise NotImplementedError("Training_step not Meant to be used for ensemble")
    
    def validation_step(self, batch):
        raise NotImplementedError("Validation_step not Meant to be used for ensemble")

    def test_step(self, batch):
        raise NotImplementedError("Test_step not Meant to be used for ensemble")
    
    def copy(self):
        raise NotImplementedError("copy not Meant to be used for ensemble")
    
    def save(self, path):
        print(f"saving to {path}")
        os.makedirs(path, exist_ok=True)
        for i, (pipeline) in enumerate(self.pipeline_arr):
            torch.save(pipeline.state_dict(), f"{path}/sub_model{i}.pth")


    def load(self, path):
        for i, (pipeline) in enumerate(self.pipeline_arr):
            pipeline.load_state_dict(torch.load(f"{path}/sub_model{i}.pth", weights_only=True))
    
    def fit(self): 
        with ThreadPoolExecutor(max_workers=self.num_ensembles) as executor:
            futures = [executor.submit(pipeline.fit) for pipeline in self.pipeline_arr]
            for future in as_completed(futures):
                future.result()

    #! Make test test the ensemble This is not tried out and need testing
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

        mean_arr = np.array(self.all_predictions[0]).reshape(-1, self.horizen_len) 
        stddev_arr = np.array(self.all_predictions[1]).reshape(-1, self.horizen_len) 
        
        all_y = []
        if len(self.all_predictions[0]) > len(self.all_actuals):
            for i in range(0, len(self.all_actuals)):
                all_y.append(self.all_actuals[i:i+self.horizen_len])
        else:
            all_y = np.array(self.all_actuals).reshape(-1, self.horizen_len) 

        func_arr = self.test_error_func_arr
        for func in func_arr:
            loss_arr = []
            for mean, stddev, y in zip(mean_arr, stddev_arr, all_y):     
                if func.is_deterministic():
                    temp_loss = func.calc(torch.tensor(mean, device=y.device), torch.tensor(y, device=y.device))
                elif func.is_probabilistic():
                    temp_loss = func.calc(torch.tensor(mean, device=y.device), torch.tensor(stddev, device=y.device), torch.tensor(y, device=y.device))
                if temp_loss.numel() == 1:
                    loss_arr.append(temp_loss)
                else:
                    print(f"temp_loss: {temp_loss}, mean: {mean}, y: {y}")

            title = func.get_title()
            avg_loss = (sum(loss_arr) / len(loss_arr)).item()
            print(f"{title:<30} {avg_loss:.6f}")

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

 
    class Builder(ProbabilisticPipeline.Builder):
        def __init__(self):
            super().__init__()
            self.pipeline_class = EnsemblePipeline
            self.pipeline_arr = []
            self.sub_pipeline = None
            self.num_ensembles = None

        def set_num_ensembles(self, num_ensembles):
            self.num_ensembles = num_ensembles
            return self
        
        def set_horizon_len(self, horizon_len):
            self.horizon_len = horizon_len
            return self

        def set_pipeline(self, sub_pipeline):
            if not isinstance(sub_pipeline, Pipeline):
                raise ValueError("Pipeline instance given not extended from Pipeline class")
            self.sub_pipeline = sub_pipeline
            return self
        
        def build(self):
            for _ in range(0, self.num_ensembles):
                self.pipeline_arr.append(self.sub_pipeline.copy())
                         
            return self.pipeline_class(self.pipeline_arr,
                                       self.num_ensembles,
                                       self.horizon_len,
                                       self.test_error_func_arr)
        
    


