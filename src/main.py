import argparse
import lightning as L
import matplotlib
import pandas as pd
import torch
import src.network.models.base_model as bm
import src.network.models.mc_model as mc
from src.util.normalize import normalize
from src.data_preprocess.data import split_data_train_and_test
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from src.util.conditional_early_stopping import ConditionalEarlyStopping
from src.util.plot import plot_results
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.util.constants import NUM_WORKERS

matplotlib.use("Agg")

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

    df = pd.read_csv(DATA_PATH)
    train_len = int(len(df) * 0.8)
    val_len = int(len(df) * 0.1)
    
    train_data = df[:train_len]
    val_data = df[train_len:train_len+val_len]
    test_data = df[train_len+val_len:]
    
    train_data = train_data.values
    val_data = val_data.values
    test_data = test_data.values

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations to run the training and testing loop.')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    if args.debug:
        print("DEBUG MODE")
    main(args.iterations, args.debug)
