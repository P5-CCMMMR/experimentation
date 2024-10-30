import argparse
import lightning as L
import matplotlib
import pandas as pd
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import torch
from src.util.conditional_early_stopping import ConditionalEarlyStopping
from src.util.flex_error import get_mafe, get_prob_mafe
from src.util.power_splitter import PowerSplitter
from src.util.continuity_splitter import split_dataframe_by_continuity
from src.util.error import NRMSE, NLL

from src.pipelines.pipeline import Pipeline
from src.pipelines.cleaners.temp_cleaner import TempCleaner
from src.pipelines.handlers.deterministic_handler import DeterministicHandler
from src.pipelines.handlers.monte_carlo_handler import MonteCarloHandler
from src.pipelines.models.lstm import LSTM
from src.pipelines.models.gru import GRU
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.pipelines.sequencers.time_sequencer import TimeSequencer
from src.pipelines.splitters.std_splitter import StdSplitter
from src.pipelines.tuners.std_tuner_wrapper import StdTunerWrapper

matplotlib.use("Agg")

MODEL_PATH = 'model.pth'
NUM_WORKERS = 1 #multiprocessing.cpu_count()
TARGET_COLUMN = 1
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"


# Hyper parameters

    # Model
input_size = 3
time_horizon = 4
hidden_size = 32 * time_horizon
num_epochs = 20 #250 * time_horizon
seq_len = 96
num_layers = 2 
        # MC ONLY
inference_samples = 50

    # Training
swa_learning_rate = 0.01
dropout = 0.50
gradient_clipping = 0
early_stopping_threshold = 0.1  
num_ensembles = 1

# Controlled by tuner
batch_size = 128
learning_rate = 0.005

# Data Split
train_days = 16
val_days = 2
test_days = 2

# ON / OFF Power Limits
off_limit_w = 100
on_limit_w = 1500

consecutive_points = 3

nist_path = "src/data_preprocess/dataset/NIST_cleaned.csv"

clean_in_low = 10
clean_in_high = 30
clean_out_low = -50
clean_out_high = 50
clean_pow_low = 0
clean_delta_temp = 15

# TODO
# [x] 1. Get flexpredict working
# [ ] 2. Change handlers to be the pipelines
#           - Ensemble having to be implemented for every pipeline
#           -      
# [ ] 2. Make flexpredict modules and get_mafe a module holding flexpredict
# [ ] 3. Modulize plotter
# [ ] 4. Figure out how to correctly train on multiple dataset

def main(d):
    temp_boundery = 0.5
    error = 0
    probalistic = True
    df = pd.read_csv(nist_path)

    cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
    splitter = StdSplitter(train_days, val_days, test_days)
    model = GRU(hidden_size, num_layers, input_size, time_horizon, dropout)
    trainer = L.Trainer(max_epochs=num_epochs, 
                        callbacks=[StochasticWeightAveraging(swa_lrs=swa_learning_rate), 
                                   ConditionalEarlyStopping(threshold=early_stopping_threshold)], 
                        gradient_clip_val=gradient_clipping, 
                        fast_dev_run=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    models = Pipeline() \
        .add_data(df) \
        .set_clean(cleaner) \
        .set_normalizer_class(MinMaxNormalizer) \
        .set_splitter(splitter) \
        .set_sequencer_class(TimeSequencer) \
        .set_target_column(TARGET_COLUMN) \
        .set_model(model) \
        .set_optimizer(optimizer) \
        .set_handler_class(MonteCarloHandler) \
        .set_inference_samples(inference_samples) \
        .set_batch_size(batch_size) \
        .set_seq_len(seq_len) \
        .set_worker_num(NUM_WORKERS) \
        .set_error(NRMSE) \
        .set_test_error(NLL) \
        .set_trainer(trainer) \
        .set_num_ensembles(1) \
        .set_tuner(StdTunerWrapper) \
        .run()


    model = models[0][0]

    model.eval()

    ps = PowerSplitter(df, TIMESTAMP, POWER)

    on_df = ps.get_mt_power(on_limit_w, consecutive_points)
    off_df = ps.get_lt_power(off_limit_w, consecutive_points)

    on_data_arr = split_dataframe_by_continuity(on_df, 15, seq_len, TIMESTAMP)
    off_data_arr = split_dataframe_by_continuity(off_df, 15, seq_len, TIMESTAMP)

    if (probalistic):
        print(get_prob_mafe(on_data_arr, model, seq_len, error, temp_boundery, time_horizon, TARGET_COLUMN))
        print(get_prob_mafe(off_data_arr, model, seq_len, error, temp_boundery, time_horizon, TARGET_COLUMN))
    else:
        print(get_mafe(on_data_arr, model, seq_len, error, temp_boundery, time_horizon, TARGET_COLUMN))
        print(get_mafe(off_data_arr, model, seq_len, error, temp_boundery, time_horizon, TARGET_COLUMN))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('--d', action='store_true', help='Debug mode')
    args = parser.parse_args()
    if args.d:
        print("DEBUG MODE")
    main(args.d)
