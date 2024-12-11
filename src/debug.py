from src.util.evaluate import evaluate_model

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
from src.util.importer import *
uk_path = "src/data_preprocess/dataset/UKDATA_cleaned.csv"
nist_path = "src/data_preprocess/dataset/NIST_cleaned.csv"

assert time_horizon > 0, "Time horizon must be a positive integer"

batch_size = 1
    
df = pd.read_csv(uk_path).iloc[0:2000]

cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
splitter = DaySplitter(TIMESTAMP, POWER, train_days, val_days, test_days) 

uk_naive_model = NaiveProbabilisticBaseline.Builder() \
    .add_data(df) \
    .set_cleaner(cleaner) \
    .set_normalizer_class(MinMaxNormalizer) \
    .set_splitter(splitter) \
    .set_sequencer_class(AllTimeSequencer) \
    .set_target_column(TARGET_COLUMN) \
    .set_horizon_len(time_horizon) \
    .set_worker_num(NUM_WORKERS) \
    .set_seq_len(time_horizon) \
    .set_error(NRMSE) \
    .set_train_error(RMSE) \
    .add_test_error(NMCRPS) \
    .add_test_error(CALE) \
    .set_penalty_strat(Naive) \
    .build()

uk_naive_result_dict = uk_naive_model.test()

uk_naive_model.reset_forward_memory()

uk_naive_on, uk_naive_off = evaluate_model(uk_naive_model, df, splitter, cleaner, TIMESTAMP, POWER, on_limit_w, off_limit_w, consecutive_points, time_horizon, time_horizon, TARGET_COLUMN, error, temp_boundary, 0.95)