import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
from experiments.iter1.util.importer import *
MODEL_PATH = 'model_saves/testing_model'
nist_path = "../../src/data_preprocess/dataset/UKDATA_cleaned.csv"
num_layers = 1