import argparse
import lightning as L
import matplotlib
import pandas as pd
import src.network.models.base_model as bm
import src.network.models.mc_model as mc
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
from src.util.flex_error import get_mafe, get_prob_mafe
from src.pipeline.models.lstm import LSTM

matplotlib.use("Agg")

MODEL_PATH = 'model.pth'
TARGET_COLUMN = 1

# Other
early_stopping_threshold = 0.1  
time_horizon = 4

# Hyper parameters
hidden_size = 32 * time_horizon
num_epochs = 250 * time_horizon
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

def train_and_test_model(trainer, lit_model):
    trainer.fit(lit_model)
    trainer.test(lit_model)
    return lit_model.get_predictions(), lit_model.get_actuals()

def main(i, d):
    temp_boundery = 0.5
    error = 0
    probalistic = True

    mnist_dh = DataHandler(nist, TvtDataSplitter)

    models = model_training_and_eval(mnist_dh, LSTM, probalistic, i, d)

    model = models[0]

    model.eval()

    on_df = mnist_dh.get_on_data()
    off_df = mnist_dh.get_off_data()

    on_data_arr = mnist_dh.split_dataframe_by_continuity(on_df, 15, seq_len)
    off_data_arr = mnist_dh.split_dataframe_by_continuity(off_df, 15, seq_len)

    if (probalistic):
        print(get_prob_mafe(on_data_arr, model, seq_len, error, temp_boundery, time_horizon))
        print(get_prob_mafe(off_data_arr, model, seq_len, error, temp_boundery, time_horizon))
    else:
        print(get_mafe(on_data_arr, model, seq_len, error, temp_boundery, time_horizon))
        print(get_mafe(off_data_arr, model, seq_len, error, temp_boundery, time_horizon))

def model_training_and_eval(mnist_dh, model_constructor, is_prob, iterations, debug):
    train_data = mnist_dh.get_train_data().values
    val_data   = mnist_dh.get_val_data().values
    test_data  = mnist_dh.get_test_data().values

    test_timestamps = pd.to_datetime(test_data[:,0])

    train_data, _, _ = norm.minmax_scale(train_data[:,1:].astype(float))
    val_data, _, _ = norm.minmax_scale(val_data[:,1:].astype(float))
    test_data, test_min_vals, test_max_vals = norm.minmax_scale(test_data[:,1:].astype(float))
    
    num_columns = train_data.shape[1]

    all_models = []
    for _ in range(num_ensembles):
        model = model_constructor(hidden_size, num_layers, num_columns, time_horizon, dropout)

        if (is_prob):
            lit_model = mc.MCModel(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data, inference_samples)
        else: 
            lit_model = bm.BaseModel(model, learning_rate, seq_len, batch_size, train_data, val_data, test_data)

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

    return all_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('--i', type=int, required=True, help='Number of iterations to run the training and testing loop.')
    parser.add_argument('--d', action='store_true', help='Debug mode')
    args = parser.parse_args()
    if args.d:
        print("DEBUG MODE")
    main(args.i, args.d)
