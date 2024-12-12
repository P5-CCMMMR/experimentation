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

from src.util.plotly import plot_results, plot_loss, plot_pillar_diagrams

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
from src.pipelines.tuners.tuner import Tuner
from src.pipelines.optimizers.optimizer import OptimizerWrapper

from src.pipelines.metrics.crps import *
from src.pipelines.metrics.lscv import *
from src.pipelines.metrics.rmse import * 
from src.pipelines.metrics.mae import *
from src.pipelines.metrics.maxe import *
from src.pipelines.metrics.cale import *

from src.pipelines.deterministic_pipeline import DeterministicPipeline
from src.pipelines.monte_carlo_pipeline import MonteCarloPipeline
from src.pipelines.ensemble_pipeline import EnsemblePipeline
from src.pipelines.probabilistic_pipeline import ProbabilisticPipeline

from src.util.evaluator import Evaluator

import torch.optim as optim

matplotlib.use("Agg")

NUM_WORKERS = multiprocessing.cpu_count()
TARGET_COLUMN = 2
POWER_COLUMN = 1
OUTDOOR_COLUMN = 3
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"

# Hyper parameters
# Model
input_size = 4
time_horizon = 4
hidden_size = 64
num_epochs = 100
seq_len = 96
num_layers = 1
 
# MC ONLY
inference_samples = 50
inference_dropout = 0.25

# Training
dropout = 0
gradient_clipping = 0
early_stopping_threshold = 0.18

num_ensembles = 2

# Flexibility
flex_confidence = 0.90
temp_boundary = 0.1
error = 0

# Controlled by tuner
batch_size = 128
learning_rate = 0.005

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
