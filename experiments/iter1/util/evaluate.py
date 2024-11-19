import numpy as np
import pandas as pd
from src.util.power_splitter import PowerSplitter
from src.util.plotly import plot_results
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.util.evaluator import Evaluator


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