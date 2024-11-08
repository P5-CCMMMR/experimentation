from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import numpy as np
from src.pipelines.pipeline import Pipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline

class EnsemblePipeline(ProbabilisticPipeline):
    def __init__(self, pipeline_arr, num_ensembles):
        super().__init__(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
        self.pipeline_arr = pipeline_arr
        self.num_ensembles = num_ensembles

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
            self.pipeline_arr = []
            self.sub_pipeline = None
            self.num_ensembles = None

        def set_num_ensembles(self, num_ensembles):
            self.num_ensembles = num_ensembles
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
                                       self.num_ensembles)
        
    


