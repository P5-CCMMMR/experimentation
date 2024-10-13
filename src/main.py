import argparse
import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import torch
import src.network.models.base_model as bm
import src.network.models.mc_model as mc
from src.util.flex_predict import flex_predict
from src.util.multi_timestep_forecast import multiTimestepForecasting
import src.util.normalize as norm
from src.data_preprocess.data_handler import DataHandler
from src.data_preprocess.tvt_data_splitter import TvtDataSplitter
from src.data_preprocess.day_data_splitter import DayDataSplitter
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from src.util.conditional_early_stopping import ConditionalEarlyStopping
from src.util.plot import plot_results
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.util.constants import NUM_WORKERS
from src.util.error import RMSE

matplotlib.use("Agg")

MODEL_PATH = 'model.pth'
MODEL_ITERATIONS = 10
TARGET_COLUMN = 1

# Hyper parameters
hidden_size = 32
num_epochs = 125
seq_len = 96
swa_learning_rate = 0.01
num_layers = 2
dropout = 0.50
gradient_clipping = 0
num_ensembles = 1

# MC ONLY
inference_samples = 50

# Controlled by tuner
batch_size = 128
learning_rate = 0.005

# Other
early_stopping_threshold = 0.25

# Data Parameters
nist = {
    "training_days"       : 16, 
    "val_days"            : 2,
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : 100,
    "on_limit_w"          : 1500,   
    "consecutive_points"  : 3,

    "train_data_path"     : "src/data_preprocess/dataset/train/NIST.csv",
    "val_data_path"       : "src/data_preprocess/dataset/val/NIST.csv",
    "test_data_path"      : "src/data_preprocess/dataset/test/NIST.csv",
    "on_data_path"        : "src/data_preprocess/dataset/on/NIST.csv",
    "off_data_path"       : "src/data_preprocess/dataset/off/NIST.csv",
    "data_path"           : "src/data_preprocess/dataset/NIST_cleaned.csv",

    "power_col"           : "PowerConsumption",
    "timestamp_col"       : "Timestamp"
}

dengiz = {
    "training_days"       : 18, 
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : None,   # Yet to be known
    "on_limit_w"          : None,   # Yet to be known
    "consecutive_points"  : 3,

    "train_data_path"     : "src/data_preprocess/dataset/train/Dengiz.csv",
    "test_data_path"      : "src/data_preprocess/dataset/test/Dengiz.csv",
    "on_data_path"        : "src/data_preprocess/dataset/on/Dengiz.csv",
    "off_data_path"       : "src/data_preprocess/dataset/off/Dengiz.csv",
    "data_path"           : "src/data_preprocess/dataset/Dengiz_cleaned.csv",

    "power_col"           : "PowerConsumption",
    "timestamp_col"       : "Timestamp"
}

# Other
early_stopping_threshold = 0.1  
time_horizon = 4

# General Constant
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"


def train_and_test_model(trainer, lit_model):
    trainer.fit(lit_model)
    trainer.test(lit_model)
    return lit_model.get_predictions(), lit_model.get_actuals()

def main(i, d):
    temp_boundery = 0.5
    seq_len = 4
    error = 0

    mnist_dh = DataHandler(nist, DayDataSplitter)

    model_training_and_eval(mnist_dh, i, d)

    model = bm.GRU(hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    on_df = mnist_dh.get_on_data()
    off_df = mnist_dh.get_off_data()

    on_data_arr = mnist_dh.split_dataframe_by_continuity(on_df, 15, seq_len)
    off_data_arr = mnist_dh.split_dataframe_by_continuity(off_df, 15, seq_len)

    print(get_mafe(on_data_arr, model, seq_len, error, temp_boundery))
    print(get_mafe(off_data_arr, model, seq_len, error, temp_boundery))

def model_training_and_eval(mnist_dh, iterations, debug):
    train_data = mnist_dh.get_train_data().values
    val_data   = mnist_dh.get_val_data().values
    test_data  = mnist_dh.get_test_data().values

    test_timestamps = pd.to_datetime(test_data[:,0])

    train_data, _, _ = norm.minmax_scale(train_data[:,1:].astype(float))
    val_data, _, _ = norm.minmax_scale(val_data[:,1:].astype(float))
    test_data, test_min_vals, test_max_vals = norm.minmax_scale(test_data[:,1:].astype(float))
    
    best_loss = None
    # TODO: change iterations to save folder with best ensembled model
    for _ in range(iterations):
        all_models = []
        for _ in range(num_ensembles):
            model = bm.GRU(hidden_size, num_layers, dropout)
            lit_model = mc.MCModel(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data, inference_samples)
            all_models.append(lit_model)
        
        trainers = [L.Trainer(max_epochs=num_epochs, callbacks=[StochasticWeightAveraging(swa_lrs=swa_learning_rate), ConditionalEarlyStopping(threshold=early_stopping_threshold)], gradient_clip_val=gradient_clipping, fast_dev_run=debug) for _ in range(num_ensembles)]
        
        tuners = [Tuner(trainer) for trainer in trainers]
        for tuner, lit_model in zip(tuners, all_models):
            tuner.lr_find(lit_model)
            tuner.scale_batch_size(lit_model, mode="binsearch")
            
        all_predictions = []
        all_actuals = None
        
        # Run ensembles in parallel
        with ThreadPoolExecutor(max_workers=min(num_ensembles, NUM_WORKERS)) as executor:
            futures = [executor.submit(train_and_test_model, trainer, lit_model) for trainer, lit_model in zip(trainers, all_models)]
            for future in as_completed(futures):
                predictions, actuals = future.result()
                all_predictions.append(predictions)
                if all_actuals is None:
                    all_actuals = actuals
            
        plot_results(all_predictions, all_actuals, test_timestamps, test_min_vals, test_max_vals)

        model = bm.GRU(hidden_size, num_layers, dropout)
        lit_model = bm.BaseModel(model, learning_rate, time_horizon, batch_size, train_data, val_data, test_data)
        trainer = L.Trainer(max_epochs=num_epochs, callbacks=[StochasticWeightAveraging(swa_lrs=swa_learning_rate), ConditionalEarlyStopping(threshold=early_stopping_threshold)], gradient_clip_val=gradient_clipping, fast_dev_run=debug)
        tuner = Tuner(trainer)
        tuner.lr_find(lit_model)
        tuner.scale_batch_size(lit_model, mode="binsearch")

        trainer.fit(lit_model)
        test_results = trainer.test(lit_model)

        test_loss = test_results[0].get('test_loss_epoch', None) if test_results else None

        if best_loss is None or best_loss > test_loss :
            print("NEW BEST")
#            lit_model.plot_results(test_timestamps, test_min_vals, test_max_vals)
            best_loss = test_loss 
            torch.save(model.state_dict(), MODEL_PATH)



def get_mafe(data_arr, model, seq_len, error, boundary):
    flex_predictions = []
    flex_actual_values = []

    for data in data_arr:
        for i in range(0, len(data), seq_len):
            if len(data) < i + seq_len * 2:
                break

            in_temp_idx = 2

            input_data = data[i: i + seq_len]

            # get the actual result data by first gettin the next *seq* data steps forward, 
            # and taking only the in_temp_id column to get the actual result indoor temperatures
            result_actual = data[i + seq_len : i + (seq_len * 2), in_temp_idx:in_temp_idx + 1] 

            result_predictions = multiTimestepForecasting(model, input_data, seq_len)

            last_in_temp = input_data[len(input_data) - 1][2]

            lower_boundery = last_in_temp - boundary
            upper_boundery = last_in_temp + boundary

            actual_flex = flex_predict(result_actual, lower_boundery, upper_boundery, error)
            predicted_flex = flex_predict(result_predictions, lower_boundery, upper_boundery, error)


            flex_predictions.append(predicted_flex)
            flex_actual_values.append(actual_flex)
        
    flex_predictions_tensor = torch.tensor(flex_predictions, dtype=torch.float32)
    flex_actual_values_tensor = torch.tensor(flex_actual_values, dtype=torch.float32)

    flex_difference = [RMSE(a, b) for a, b in zip(flex_predictions_tensor, flex_actual_values_tensor)]
    return (sum(flex_difference) / len(flex_difference)).item()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('--i', type=int, required=True, help='Number of iterations to run the training and testing loop.')
    parser.add_argument('--d', action='store_true', help='Debug mode')
    args = parser.parse_args()
    if args.d:
        print("DEBUG MODE")
    main(args.i, args.d)
