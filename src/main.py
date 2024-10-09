import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import torch
import src.network.models.base_model as bm
import src.network.models.mc_model as mc
from src.util.flex_predict import flexPredict
from src.util.multi_timestep_forecast import multiTimestepForecasting
from src.util.normalize import normalize
from src.data_preprocess.data_handler import DataHandler
from src.data_preprocess.ttt_data_splitter import TttDataSplitter
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from src.util.conditional_early_stopping import ConditionalEarlyStopping
from src.util.plot import plot_results
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.util.constants import NUM_WORKERS

matplotlib.use("Agg")

MODEL_ITERATIONS = 10
TARGET_COLUMN = 1

# Hyper parameters
hidden_size = 24
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
num_layers = 1
dropout = 0
time_horizon = 4

# Data Parameters
nist = {
    "training_days"       : 18, 
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : 100,
    "on_limit_w"          : 1500,   
    "consecutive_points"  : 3,

    "train_data_path"     : "src/data_preprocess/dataset/train/NIST.csv",
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

# Data Parameters
nist = {
    "training_days"       : 18, 
    "test_days"           : 2,
    "validation_days"     : 0,
    "off_limit_w"         : 100,
    "on_limit_w"          : 1500,   
    "consecutive_points"  : 3,
    "train_data_path"     : "src/data_preprocess/dataset/train/NIST.csv",
    "test_data_path"      : "src/data_preprocess/dataset/train/NIST.csv",
    "on_data_path"        : "src/data_preprocess/dataset/on/NIST.csv",
    "off_data_path"       : "src/data_preprocess/dataset/off/NIST.csv",
    "data_path"           : "src/data_preprocess/dataset/NIST_cleaned.csv"
}

used_dataset       = nist
training_days      = used_dataset["training_days"]
test_days          = used_dataset["test_days"]
validation_days    = used_dataset["validation_days"]
off_limit          = used_dataset["off_limit_w"]
on_limit           = used_dataset["on_limit_w"]
consecutive_points = used_dataset["consecutive_points"]

# General Constant
TIMESTAMP = "Timestamp"
POWER     = "PowerConsumption"

# Paths
TRAIN_DATA_PATH = used_dataset["train_data_path"]
TEST_DATA_PATH  = used_dataset["test_data_path"]
ON_DATA_PATH    = used_dataset["on_data_path"]
OFF_DATA_PATH   = used_dataset["off_data_path"]
DATA_PATH       = used_dataset["data_path"]

def train_and_test_model(trainer, lit_model):
    trainer.fit(lit_model)
    trainer.test(lit_model)
    return lit_model.get_predictions(), lit_model.get_actuals()

def main(iterations, debug):
    #df = pd.read_csv(DATA_PATH)
    #train_data, test_data = split_data_train_and_test(df, training_days, test_days, TIMESTAMP)

    temp_boundery = 0.5
    seq_len = 4
    error = 0

    mnist = DataHandler(nist, TttDataSplitter)

    modelTrainingAndEval(mnist, iterations)

    model = None # Insert loading and preperation of model

    on_df, off_df = mnist.get_on_off_data()

    on_data_arr = mnist.split_dataframe_by_continuity(on_df, 15, seq_len)
    off_data_arr = mnist.split_dataframe_by_continuity(off_df, 15, seq_len)

    getMafe(on_data_arr, model, seq_len, error, temp_boundery)
    getMafe(off_data_arr, model, seq_len, error, temp_boundery)

def modelTrainingAndEval(mnist, iterations):
    train_data = mnist.get_train_value().values
    val_data   = mnist.get_val_value().values
    test_data  = mnist.get_test_value().values

    test_timestamps = pd.to_datetime(test_data[:,0])

    train_data, _, _ = normalize(train_data[:,1:].astype(float))
    val_data, _, _ = normalize(val_data[:,1:].astype(float))
    test_data, test_min_vals, test_max_vals = normalize(test_data[:,1:].astype(float))
    
    #best_loss = None
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

        #model = bm.GRU(hidden_size, num_layers, dropout)
        #lit_model = bm.BaseModel(model, learning_rate, seq_len, batch_size, #train_data, val_data, test_data, inference_samples)
        #trainer = L.Trainer(max_epochs=num_epochs, callbacks=#[StochasticWeightAveraging(swa_lrs=swa_learning_rate), #ConditionalEarlyStopping(threshold=early_stopping_threshold)], #gradient_clip_val=gradient_clipping, fast_dev_run=debug)
        #tuner = Tuner(trainer)
        #tuner.lr_find(lit_model)
        #tuner.scale_batch_size(lit_model, mode="binsearch")

        #trainer.fit(lit_model)
        #test_results = trainer.test(lit_model)

        #test_loss = test_results[0].get('test_loss_epoch', None) if test_results else None

        #if best_loss is None or best_loss > test_loss :
         #   print("NEW BEST")
          #  lit_model.plot_results(test_timestamps, test_min_vals, test_max_vals)
           # best_loss = test_loss 
            #torch.save(model.state_dict(), 'model.pth')



def getMafe(data_arr, model, seq_len, error, boundary):
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

            actual_flex = flexPredict(result_actual, lower_boundery, upper_boundery, error)
            predicted_flex = flexPredict(result_predictions, lower_boundery, upper_boundery, error)

            flex_predictions.append(predicted_flex)
            flex_actual_values.append(actual_flex)

        # need to check why the prediction allways perfect, and why its either all the data its flexible or no data
    flex_difference = [a - b for a, b in zip(flex_predictions, flex_actual_values)]
    print(flex_difference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations to run the training and testing loop.')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    if args.debug:
        print("DEBUG MODE")
    main(args.iterations, args.debug)
