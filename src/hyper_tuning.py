from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.train import Checkpoint

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
from src.pipelines.splitters.day_splitter import DaySplitter
from src.pipelines.splitters.blocked_k_fold_splitter import BlockedKFoldSplitter
from src.pipelines.tuners.std_tuner_wrapper import StdTunerWrapper
from src.pipelines.optimizers.optimizer import OptimizerWrapper

from src.pipelines.metrics.crps import *
from src.pipelines.metrics.lscv import *
from src.pipelines.metrics.rmse import * 

from src.pipelines.deterministic_pipeline import DeterministicPipeline
from src.pipelines.monte_carlo_pipeline import MonteCarloPipeline
from src.pipelines.ensemble_pipeline import EnsemblePipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline

import torch.optim as optim
MODEL_PATH = 'model_saves/testing_model'
nist_path = "/home/vind/P5/experimentation/src/data_preprocess/dataset/NIST_cleaned.csv"

config = {
    "num_epochs": tune.randint(50,1000),
    "seq_len": tune.randint(16,672),
    "hidden_size": tune.randint(4,128),
    "dropout": tune.randint(0,1),
    "time_horizon": tune.randint(4,96),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "num_layers": tune.randint(1,4)
}

matplotlib.use("Agg")

NUM_WORKERS = multiprocessing.cpu_count()
TARGET_COLUMN = 2
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"

gradient_clipping = 0
num_samples = 2
gpus_per_trial = 0

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
    
# MC ONLY
inference_dropout = 0.5
inference_samples = 50

# Ensemble ONLY
num_ensembles = 5
flex_confidence = 0.90

input_size = 4
batch_size = 128
folds = 5

def train(config):
    df = pd.read_csv(nist_path)
    df = DaySplitter(TIMESTAMP, POWER, train_days + val_days, 0, test_days).get_train(df)
    # Hyper parameters
    # Model
    time_horizon = config["time_horizon"]
    hidden_size = config["hidden_size"]
    num_epochs = config["num_epochs"]
    seq_len = config["seq_len"]
    num_layers = config["num_layers"]
    
    # Training
    dropout = config["dropout"]

    # Controlled by tuner
    learning_rate = config["learning_rate"]

    assert time_horizon > 0, "Time horizon must be a positive integer"
    for i in range(0, folds):
        df = df.copy(deep=True)
        splitter = BlockedKFoldSplitter(folds=folds)
        splitter.set_val_index(i)
        cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
        metrics = {'loss': 'val_loss'}    
        model = LSTM(hidden_size, num_layers, input_size, time_horizon, dropout)
        trainer = TrainerWrapper(L.Trainer, 
                                max_epochs=num_epochs, 
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=False, mode='min', strict=True), TuneReportCallback(metrics, on="validation_end")], 
                                gradient_clip_val=gradient_clipping)
        optimizer = OptimizerWrapper(optim.Adam, model, lr=learning_rate)

        model = DeterministicPipeline.Builder() \
                .add_data(df) \
                .set_cleaner(cleaner) \
                .set_normalizer_class(MinMaxNormalizer) \
                .set_splitter(splitter) \
                .set_sequencer_class(AllTimeSequencer) \
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
    train
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
    name="train"
)

print(analysis.best_config)