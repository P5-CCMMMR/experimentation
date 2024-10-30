import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipelines.normalizers.normalizer import Normalizer
from src.pipelines.tuners.tuner_wrapper import TunerWrapper

from src.pipelines.wip_probabilistic_pipeline import ProbabilisticPipeline

class MonteCarloPipeline(ProbabilisticPipeline):
    def __init__(self, learning_rate: float, seq_len: int, batch_size: int,
                    optimizer: torch.optim.Optimizer, model: nn.Module, trainer: L.Trainer,
                    tuner_class: TunerWrapper,
                    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                    test_timesteps: pd.DatetimeIndex, normalizer: Normalizer,
                    train_error_func, val_error_func, test_error_func,
                    inference_samples: int):
        super().__init__(learning_rate, seq_len, batch_size,
                            optimizer, model, trainer,
                            tuner_class,
                            train_loader, val_loader, test_loader,
                            test_timesteps, normalizer,
                            train_error_func, val_error_func, test_error_func)
        self.inference_samples = inference_samples

    def forward(self, x):
        self.model.train()
        predictions = []

        with torch.no_grad():
            for _ in range(self.inference_samples):
                y_hat = self.model(x)
                predictions.append(y_hat.cpu().numpy())

        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        return mean_prediction, std_prediction
        

    class Builder(ProbabilisticPipeline.Builder):
        def __init__(self):
            super().__init__()
            self.inference_samples = 0
            self.pipeline_class = MonteCarloPipeline

        def set_inference_samples(self, inference_samples):
            self.inference_samples = inference_samples
            return self
        
        def Build(self):
            # implement 
            #for df in self.df_arr:
            df = self.cleaner.clean(self.df_arr[0]) #! needs to be changed to a solution concatting the dataset

            train_df = self.splitter.get_train(df)
            val_df = self.splitter.get_val(df)
            test_df = self.splitter.get_test(df)

            test_timestamps = pd.to_datetime(test_df.values[:,0])

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

            pipeline = self.pipeline_class(self.learning_rate,
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
                                          self.inference_samples)

            return pipeline
        
    
