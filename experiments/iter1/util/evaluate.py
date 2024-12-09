import numpy as np
import pandas as pd
from src.pipelines.metrics.mafe import mafe, maofe, maufe
from src.pipelines.metrics.epfr import epfr
from src.util.power_splitter import PowerSplitter
from src.util.plotly import plot_results
from src.pipelines.normalizers.min_max_normalizer import MinMaxNormalizer
from src.util.evaluator import Evaluator

def plot_models(predictions_arr, time_horizon, time_stamps, actuals, titels=None):
    if not isinstance(predictions_arr, list):
        predictions_arr = [predictions_arr]

    is_prob = True if isinstance(predictions_arr[0], tuple) else False

    temp_pred_arr = []
    for preds in predictions_arr:
        if is_prob:
            temp_pred_arr.append(tuple(np.array(pred.flatten()).reshape(-1, time_horizon) for pred in preds))
        else:
            temp_pred_arr.append(np.array(preds.flatten()).reshape(-1, time_horizon))

    actuals_arr = np.array(actuals).reshape(-1, time_horizon)[::time_horizon].flatten()
    timestep_arr = time_stamps

    for i in range(0, time_horizon):
        temp_arr = []
        for preds_2d_arr in temp_pred_arr:
            if is_prob:
                temp_arr.append(tuple(np.array(pred)[i::time_horizon].flatten() for pred in preds_2d_arr))
            else:
                temp_arr.append(preds_2d_arr[i::time_horizon].flatten())
        plot_results(temp_arr, actuals_arr[i:], timestep_arr[i:], time_horizon, titles=titels)
   
def evaluate_model(model, df, splitter, cleaner, TIMESTAMP, POWER, on_limit_w, off_limit_w, consecutive_points, seq_len, time_horizon, TARGET_COLUMN, error, temp_boundary, confidence=0.95):
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
    on_mafe = evaluator.evaluate(mafe)
    on_maofe = evaluator.evaluate(maofe)
    on_maufe = evaluator.evaluate(maufe)
    on_epfr = evaluator.evaluate(epfr)
    print(f"On Mafe: {on_mafe}") 
    print(f"On Maofe: {on_maofe}")
    print(f"On Maufe: {on_maufe}")
    print(f"On EPFR: {on_epfr}")

    print("Calculating Off set...")
    evaluator.init_predictions(off_data, seq_len, time_horizon, TARGET_COLUMN, confidence=confidence)
    off_mafe = evaluator.evaluate(mafe)
    off_maofe = evaluator.evaluate(maofe)
    off_maufe = evaluator.evaluate(maufe)
    off_epfr = evaluator.evaluate(epfr)
    print(f"Off Mafe: {off_mafe}")
    print(f"Off Maofe: {off_maofe}")
    print(f"Off Maufe: {off_maufe}")
    print(f"Off EPFR: {off_epfr}")

    results = {
        'on mafe': on_mafe,
        'on maofe': on_maofe,
        'on maufe': on_maufe,
        'on epfr': on_epfr,
        'off mafe': off_mafe,
        'off maofe': off_maofe,
        'off maufe': off_maufe,
        'off epfr': off_epfr
    }

    return results

