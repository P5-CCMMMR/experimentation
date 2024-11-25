from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
num_samples = 10
gpus_per_trial = 0
import torch
import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import multiprocessing
from src.pipelines.trainers.trainerWrapper import TrainerWrapper
from src.util.conditional_early_stopping import ConditionalEarlyStopping
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from src.util.power_splitter import PowerSplitter

from src.util.plotly import plot_results, plot_loss

from src.pipelines.cleaners.temp_cleaner import TempCleaner
from src.pipelines.models.lstm import LSTM
from src.pipelines.models.gru import GRU
from src.pipelines.models.tcn import TCN
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.pipelines.sequencers.time_sequencer import TimeSequencer
from src.pipelines.sequencers.all_time_sequencer import AllTimeSequencer
from src.pipelines.splitters.std_splitter import StdSplitter
from src.pipelines.tuners.std_tuner_wrapper import StdTunerWrapper
from src.pipelines.optimizers.optimizer import OptimizerWrapper

from src.pipelines.metrics.crps import *
from src.pipelines.metrics.lscv import *
from src.pipelines.metrics.rmse import * 

from src.pipelines.deterministic_pipeline import DeterministicPipeline
from src.pipelines.monte_carlo_pipeline import MonteCarloPipeline
from src.pipelines.ensemble_pipeline import EnsemblePipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline

from src.util.evaluator import Evaluator

import torch.optim as optim

config = {
    "num_epochs": tune.choice([800,1000,1200]),
    "seq_len": tune.choice([80,96,114]),
    "batch_size": tune.choice([100,128,156]),
    "learning_rate": tune.choice([0.004,0.005,0.006]),
    "hidden_size": tune.choice([20,32,48]),
    "dropout": tune.choice([0.4,0.5,0.6]),
    "time_horizon": tune.choice([3,4,5])
}

matplotlib.use("Agg")

NUM_WORKERS = multiprocessing.cpu_count()
TARGET_COLUMN = 2
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"



# Data Split
train_days = 16
val_days = 2
test_days = 2

# ON / OFF Power Limits
off_limit_w = 88
on_limit_w = 963

consecutive_points = 3

clean_in_low = 10
clean_in_high = 30
clean_out_low = -50
clean_out_high = 50
clean_pow_low = 0
clean_delta_temp = 15








def nig():
    # Hyper parameters
    # Model
    # Model
    input_size = 4
    time_horizon = config["time_horizon"]
    hidden_size = config["hidden_size"]
    num_epochs = config["num_epochs"]
    seq_len = config["seq_len"]
    num_layers = 2
    
    # MC ONLY
    inference_samples = 50
    inference_dropout = 0.5

    # Training
    dropout = config["dropout"]
    gradient_clipping = 0
    early_stopping_threshold = 0.15

    num_ensembles = 5

    # Flexibility
    flex_confidence = 0.90
    temp_boundary = 0.1
    error = 0

    # Controlled by tuner
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    assert time_horizon > 0, "Time horizon must be a positive integer"
    
    df = pd.read_csv(nist_path)

    cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
    splitter = StdSplitter(train_days, val_days, test_days)
    metrics = {'loss': 'val_loss'}    
    model = LSTM(hidden_size, num_layers, input_size, time_horizon, dropout)
    trainer = TrainerWrapper(L.Trainer, 
                            max_epochs=num_epochs, 
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=False, mode='min', strict=True), TuneReportCallback(metrics, on="validation_end")], 
                            gradient_clip_val=gradient_clipping)
    trainer.fit(model, df)
    optimizer = OptimizerWrapper(optim.Adam, model, lr=learning_rate)

    model = DeterministicPipeline.Builder() \
            .add_data(df) \
            .set_cleaner(cleaner) \
            .set_normalizer_class(MinMaxNormalizer) \
            .set_splitter(splitter) \
            .set_sequencer_class(TimeSequencer) \
            .set_target_column(TARGET_COLUMN) \
            .set_model(model) \
            .set_optimizer(optimizer) \
            .set_batch_size(batch_size) \
            .set_seq_len(seq_len) \
            .set_worker_num(NUM_WORKERS) \
            .set_error(NRMSE) \
            .set_train_error(RMSE) \
            .set_trainer(trainer) \
            .set_tuner_class(StdTunerWrapper) \
            .build()

    model.fit()

trainable = tune.with_parameters(
    nig ,
    num_gpus=gpus_per_trial
)

analysis = tune.run(
    
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": gpus_per_trial
    },
    metric="loss",
    mode="min",
    config=config,
    num_samples=num_samples,
    name="nig"
)

print(analysis.best_config)