import argparse
import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import multiprocessing
from src.pipelines.trainers.trainerWrapper import TrainerWrapper
from src.util.conditional_early_stopping import ConditionalEarlyStopping
from src.util.evaluator import Evaluator
from src.util.plotly import plot_results
from src.util.power_splitter import PowerSplitter

from src.pipelines.cleaners.temp_cleaner import TempCleaner
from src.pipelines.models.lstm import LSTM
from src.pipelines.models.gru import GRU
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.pipelines.sequencers.time_sequencer import TimeSequencer
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

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import torch.optim as optim


matplotlib.use("Agg")

MODEL_PATH = 'model.pth'
NUM_WORKERS = multiprocessing.cpu_count()
TARGET_COLUMN = 2
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"

# Hyper parameters
config = {
    "num_epochs": tune.choice([800,1000,1200]),
    "seq_len": tune.choice([80,96,114]),
    "batch_size": tune.choice([100,128,156]),
    "learning_rate": tune.choice([0.004,0.005,0.006]),
    "hidden_size": tune.choice([20,32,48]),
    "dropout": tune.choice([0.4,0.5,0.6]),
    "time_horizon": tune.choice([3,4,5])
}

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

# Data Split
train_days = 16
val_days = 2
test_days = 2

# ON / OFF Power Limits
off_limit_w = 100
on_limit_w = 1500

consecutive_points = 3

nist_path = "src/data_preprocess/dataset/NIST_cleaned.csv"
ukdata_path = "src/data_preprocess/dataset/UKDATA_cleaned.csv"

clean_in_low = 10
clean_in_high = 30
clean_out_low = -50
clean_out_high = 50
clean_pow_low = 0
clean_delta_temp = 15

def main(d):
    assert time_horizon > 0, "Time horizon must be a positive integer"
    
    df = pd.read_csv(nist_path)

    cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
    splitter = StdSplitter(train_days, val_days, test_days)
    
    model = LSTM(hidden_size, num_layers, input_size, time_horizon, dropout)
    metrics = {'loss': 'val_loss', "acc": "NRMSE Loss: " }
    trainer = TrainerWrapper(L.Trainer, 
                             max_epochs=num_epochs, 
                             callbacks=[ConditionalEarlyStopping(threshold=early_stopping_threshold), TuneReportCallback(metrics, on="validation_end")],
                             gradient_clip_val=gradient_clipping, 
                             fast_dev_run=d)
    trainer.fit(model, df)
    optimizer = OptimizerWrapper(optim.Adam, model, lr=learning_rate)

    model = MonteCarloPipeline.Builder() \
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
        .set_trainer(trainer) \
        .set_tuner_class(StdTunerWrapper) \
        .set_inference_samples(inference_samples) \
        .set_inference_dropout(inference_dropout) \
        .build()


#    model = EnsemblePipeline.Builder() \
#        .set_pipeline(model) \
#        .set_num_ensembles(num_ensembles) \
#        .build()

    model.fit()

    def evaluate_model(model, df, splitter, cleaner, TIMESTAMP, POWER, on_limit_w, off_limit_w, consecutive_points, seq_len, time_horizon, TARGET_COLUMN, error, temp_boundary, confidence):
        predictions = model.get_predictions()

        if isinstance(predictions, tuple):
            predictions_2d_arr = tuple(np.array(pred).reshape(-1, time_horizon) for pred in predictions)
        else:
            predictions_2d_arr = np.array(predictions).reshape(-1, time_horizon)

        actuals_arr = np.array(model.get_actuals()).reshape(-1, time_horizon)[::time_horizon].flatten()
        timestep_arr = model.get_timestamps()

        if isinstance(predictions_2d_arr, tuple):
            for i in range(0, time_horizon):
                predictions_arr = tuple(np.array(pred)[i::time_horizon].flatten() for pred in predictions_2d_arr)
                plot_results(predictions_arr, actuals_arr[i:], timestep_arr[i:], time_horizon)
        else: 
            for i in range(0, time_horizon):
                predictions_arr = predictions_2d_arr[i::time_horizon].flatten()
                plot_results(predictions_arr, actuals_arr[i:], timestep_arr[i:], time_horizon)

        model.eval()

        ps = PowerSplitter(splitter.get_test(cleaner.clean(df)), TIMESTAMP, POWER)

        on_df = ps.get_mt_power(on_limit_w, consecutive_points)
        off_df = ps.get_lt_power(off_limit_w, consecutive_points)

        def normalize_and_convert_dates(data):
            data[:, 0] = pd.to_datetime(data[:, 0]).astype(int) / 10**9
            temp = MinMaxNormalizer(data.astype(float)).normalize()
            return temp[0]

        on_data = np.array(on_df)
        on_data = normalize_and_convert_dates(on_data)

        off_data = np.array(off_df)
        off_data = normalize_and_convert_dates(off_data)

        evaluator = Evaluator(model, error, temp_boundary)

        print("Calculating On set...")
        evaluator.init_predictions(on_data, seq_len, time_horizon, TARGET_COLUMN, confidence=confidence) 
        print(f"On Mafe: {evaluator.evaluate(lambda a, b: abs(a - b))}") 
        print(f"On Maofe: {evaluator.evaluate(lambda a, b: abs(max(a - b, 0)))}")
        print(f"On Maufe: {evaluator.evaluate(lambda a, b: abs(min(a - b, 0)))}")

        print("Calculating Off set...")
        evaluator.init_predictions(off_data, seq_len, time_horizon, TARGET_COLUMN, confidence=confidence)
        print(f"Off Mafe: {evaluator.evaluate(lambda a, b: abs(a - b))}")
        print(f"Off Maofe: {evaluator.evaluate(lambda a, b: abs(max(a - b, 0)))}")
        print(f"Off Maufe: {evaluator.evaluate(lambda a, b: abs(min(a - b, 0)))}")

    evaluate_model(model, df, splitter, cleaner, TIMESTAMP, POWER, on_limit_w, off_limit_w, consecutive_points, seq_len, time_horizon, TARGET_COLUMN, error, temp_boundary, 0.95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('-d', action='store_true', help='Debug mode')
    args = parser.parse_args()
    if args.d:
        print("DEBUG MODE")
    main(args.d)
