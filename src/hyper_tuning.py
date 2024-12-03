import numpy as np
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.train import Checkpoint

import lightning as L
import matplotlib
import pandas as pd
import multiprocessing
from src.pipelines.trainers.trainerWrapper import TrainerWrapper
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.pipelines.cleaners.temp_cleaner import TempCleaner
from src.pipelines.models.lstm import LSTM
from src.pipelines.models.gru import GRU
from src.pipelines.models.tcn import TCN
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.pipelines.sequencers.all_time_sequencer import AllTimeSequencer
from src.pipelines.splitters.day_splitter import DaySplitter
from src.pipelines.splitters.blocked_k_fold_splitter import BlockedKFoldSplitter
from src.pipelines.tuners.std_tuner_wrapper import StdTunerWrapper
from src.pipelines.optimizers.optimizer import OptimizerWrapper

from src.pipelines.metrics.crps import *
from src.pipelines.metrics.lscv import *
from src.pipelines.metrics.rmse import * 

from src.pipelines.deterministic_pipeline import DeterministicPipeline

import torch.optim as optim
nist_path = "/home/vind/P5/experimentation/src/data_preprocess/dataset/NIST_cleaned.csv"
uk_path = "/home/vind/P5/experimentation/src/data_preprocess/dataset/UKDATA_cleaned.csv"

config = {
    "seq_len": tune.qrandint(16,672, 8), 
    "hidden_size": tune.qrandint(24,128, 8),
    "dropout": tune.quniform(0, 0.8, 0.1),
    "num_layers": tune.randint(1,3),
    "arch_idx": tune.randint(0, 3) # LSTM, GRU, TCN
}
arch_arr = [LSTM, GRU, TCN]
arch_str_arr = ["LSTM", "GRU", "TCN"]

matplotlib.use("Agg")

NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2)
TARGET_COLUMN = 2
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"

gradient_clipping = 0
gpus_per_trial = 0.1

# Data Split
train_days = 16
val_days = 2
test_days = 2

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
num_epochs = 100
learning_rate = 0.005
time_horizon = 4
num_samples = 10

debug = False

if debug:
    num_epochs = 1
    num_samples = 1
    folds = 2

def train(config):
    df_arr = []
    for path in [nist_path, uk_path]:
        df = pd.read_csv(path)
        df = DaySplitter(TIMESTAMP, POWER, train_days + val_days, 0, test_days).get_train(df)
        df_arr.append(df)

    hidden_size = config["hidden_size"]
    seq_len = config["seq_len"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    arch_class = arch_arr[config["arch_idx"]]

    assert time_horizon > 0, "Time horizon must be a positive integer"
    for df in df_arr:
        for i in range(0, folds):
            df = df.copy(deep=True)
            splitter = BlockedKFoldSplitter(folds=folds)
            splitter.set_val_index(i)
            cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
            metrics = {'loss': 'val_loss'}    
            if arch_class == TCN:
                model = arch_class(hidden_size, num_layers, input_size, time_horizon, dropout, seq_len)
            else:
                model = arch_class(hidden_size, num_layers, input_size, time_horizon, dropout)
            
            trainer = TrainerWrapper(L.Trainer, 
                                    max_epochs=num_epochs, 
                                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=False, mode='min', strict=True), 
                                               TuneReportCallback(metrics, on="validation_end")], 
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
        "cpu": NUM_WORKERS,
        "gpu": gpus_per_trial
    },
    metric="loss",
    mode="min",
    config=config,
    num_samples=num_samples,
    name="train"
)

results = analysis.results_df.sort_values(by="loss")

output_file = "all_config.txt"

with open(output_file, "a") as f:
    f.write("-" * 100 + "\n")
    f.write(f"{'loss':<20} | {'architecture':<12} | {'seq_len':<10} | {'hidden_size':<12} | {'dropout':<8} | {'num_layers':<10} | {'iter':<5} | {'total_time':<10}\n")
    for i, row in results.iterrows():
        loss = row['loss']
        arch_idx = row['config/arch_idx']
        seq_len = row['config/seq_len']
        hidden_size = row['config/hidden_size']
        dropout = row['config/dropout']
        num_layers = row['config/num_layers']
        iter = row['training_iteration']
        total_time = row['time_total_s']
        f.write(f"{loss:<20} | {arch_str_arr[arch_idx]:<12} | {seq_len:<10} | {hidden_size:<12} | {round(dropout , 1):<8} | {num_layers:<10} | {iter:<5} | {total_time:<10}\n")

def clean_and_sort_file(filename):
    data = []


    with open(filename, "r") as f:
        for line in f:
            if line.startswith("-") or "loss" in line or not line.strip():
                continue
            try:
                parts = line.split("|", 1)
                loss = float(parts[0].strip())  
                rest_of_line = parts[1] if len(parts) > 1 else ""
                data.append((loss, rest_of_line.strip()))  
            except (ValueError, IndexError):
                continue

    data.sort(key=lambda x: x[0])

    with open(filename, "w") as f:
        f.write(f"{'-' * 100}\n")
        f.write(f"{'loss':<20} | {'architecture':<12} | {'seq_len':<10} | {'hidden_size':<12} | {'dropout':<8} | {'num_layers':<10} | {'iter':<5} | {'total_time':<10}\n")
        for loss, rest_of_line in data:
            f.write(f"{loss:<20} | {rest_of_line}\n")

clean_and_sort_file(output_file)