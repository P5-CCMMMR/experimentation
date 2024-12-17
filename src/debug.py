from src.util.evaluate import evaluate_model

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
from src.util.importer import *
uk_path = "src/data_preprocess/dataset/UKDATA_cleaned.csv"
nist_path = "src/data_preprocess/dataset/NIST_cleaned.csv"

assert time_horizon > 0, "Time horizon must be a positive integer"

batch_size = 1
    
df = pd.read_csv(uk_path)

cleaner = TempCleaner(clean_pow_low, clean_in_low, clean_in_high, clean_out_low, clean_out_high, clean_delta_temp)
splitter = DaySplitter(TIMESTAMP, POWER, train_days, val_days, test_days) 

uk_lag_model = LagDeterministicBaseline.Builder() \
    .add_data(df) \
    .set_cleaner(cleaner) \
    .set_normalizer_class(MinMaxNormalizer) \
    .set_splitter(splitter) \
    .set_sequencer_class(AllTimeSequencer) \
    .set_target_column(TARGET_COLUMN) \
    .set_horizon_len(time_horizon) \
    .set_worker_num(NUM_WORKERS) \
    .set_seq_len(time_horizon) \
    .set_batch_size(batch_size) \
    .set_error(NRMSE) \
    .set_train_error(RMSE) \
    .add_test_error(NMAE) \
    .add_test_error(NMAXE) \
    .build()

uk_lag_test_results = uk_lag_model.test()

uk_lag_eval_results = evaluate_model(uk_lag_model, df, splitter, cleaner, TIMESTAMP, POWER, on_limit_w, off_limit_w, consecutive_points, time_horizon, time_horizon, TARGET_COLUMN, error, temp_boundary, None)

uk_lag_results = {**uk_lag_test_results, **uk_lag_eval_results}
uk_lag_results["title"] = "Lag"