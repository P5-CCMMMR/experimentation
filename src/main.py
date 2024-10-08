import argparse
import lightning as L
import matplotlib
import pandas as pd
import torch
from src.util.normalize import normalize
from src.network.models.mc_dropout_lstm import MCDropoutLSTM
from src.network.models.mc_dropout_gru import MCDropoutGRU
from src.network.lit_model import LitModel
from src.util.plot import plot_results
from src.data_preprocess.data import split_data_train_and_test
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from src.util.conditional_early_stopping import ConditionalEarlyStopping

matplotlib.use("Agg")

TARGET_COLUMN = 1

# Hyper parameters
batch_size = 128
hidden_size = 24
n_epochs = 125
seq_len = 96
learning_rate = 0.005
swa_learning_rate = 0.01
num_layers = 4
dropout = 0.50
test_sample_nbr = 50


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
TRAIN_DATA_PATH = used_dataset[ "train_data_path"]
TEST_DATA_PATH  = used_dataset["test_data_path"]
ON_DATA_PATH    = used_dataset["on_data_path"]
OFF_DATA_PATH   = used_dataset["off_data_path"]
DATA_PATH       = used_dataset["data_path"]

def main(iterations):
    #try:
    #    train_data = pd.read_csv(TRAIN_DATA_PATH)
    #    test_data = pd.read_csv(TEST_DATA_PATH)
    #except FileNotFoundError:
    #    try:
    #        df = pd.read_csv(DATA_PATH)
    #        train_data, test_data = split_data_train_and_test(df, training_days, test_days, TIMESTAMP)
    #    except FileNotFoundError:
    #
    # raise RuntimeError(DATA_PATH + " not found")

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

    best_loss = None
    
    for _ in range(iterations):
        model = MCDropoutGRU(hidden_size, num_layers, dropout)
        lit_model = LitModel(model, learning_rate, test_sample_nbr, seq_len, batch_size, train_data, val_data, test_data)
        trainer = L.Trainer(max_epochs=n_epochs, callbacks=[StochasticWeightAveraging(swa_lrs=swa_learning_rate), ConditionalEarlyStopping(threshold=0.1)])
        tuner = Tuner(trainer)
        tuner.lr_find(lit_model)
        tuner.scale_batch_size(lit_model, mode="binsearch")

        trainer.fit(lit_model)
        test_results = trainer.test(lit_model)

        predictions, actuals = lit_model.get_results()

        test_loss = test_results[0].get('test_loss_epoch', None) if test_results else None

        if best_loss is None or best_loss > test_loss :
            print("NEW BEST")
            plot_results(predictions, actuals, test_timestamps, test_min_vals, test_max_vals, TARGET_COLUMN)
            best_loss = test_loss 
            torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model training and testing.")
    parser.add_argument('--iterations', type=int, required=True, help='Number of iterations to run the training and testing loop.')
    args = parser.parse_args()
    main(args.iterations)
