import os
import pandas as pd
import numpy as np
import time
from itertools import combinations
from tqdm import tqdm
import logging
import logging.config
import yaml
from concurrent.futures import ProcessPoolExecutor
from src.config.constants import (
    LOGGING_FILE,
    VR_STAT_DIR,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(VR_STAT_DIR, exist_ok=True)

def load_timeseries_data(file_path):

    """Loads and preprocesses time series data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with datetime index and only numeric columns.
    """

    try:
        df = pd.read_csv(file_path)
        time_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df.dropna(subset=[time_col], inplace=True)
            df.set_index(time_col, inplace=True)
        else:
            raise ValueError(f"No date or time column found in file {file_path}")
        df = df.select_dtypes(include=[float, int])
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

def autocorr_vec(x, y, lag):

    """Computes the autocorrelation between two series with a specified lag.

    Args:
        x (np.ndarray): First time series.
        y (np.ndarray): Second time series.
        lag (int): Lag at which to compute the autocorrelation.

    Returns:
        float: Autocorrelation coefficient or NaN if invalid.
    """

    if len(x) <= lag:
        return np.nan
    return np.corrcoef(x[:-lag], y[lag:])[0, 1]

def calculate_vr_hat(series, N):

    """Calculates the VR_hat statistic for a given time series and lag.

    Args:
        series (np.ndarray): Time series data.
        N (int): Lag value.

    Returns:
        float: VR_hat statistic or NaN if data is insufficient.
    """

    if len(series) < N:
        return np.nan
    mu = np.mean(series)
    rhos = np.array([autocorr_vec(series, series, tau) for tau in range(1, N)])
    N_tau = np.arange(N - 1, 0, -1)
    VR_hat = 1 + (2 / N) * np.nansum(N_tau * rhos)
    return VR_hat

def calculate_b_tau(series, tau):

    """Calculates the B_tau value for a given time series and lag.

    Args:
        series (np.ndarray): Time series data.
        tau (int): Lag value.

    Returns:
        float: B_tau statistic or NaN if data is insufficient.
    """

    s_t = (series - np.mean(series)) ** 2
    if len(s_t) <= tau:
        return np.nan
    numerator = np.dot(s_t[:-tau], s_t[tau:])
    denominator = np.sum(s_t) ** 2
    b_tau = (len(series) * numerator) / denominator if denominator != 0 else np.nan
    return b_tau

def calculate_v_N(series, N):

    """Computes the v_N statistic used in the variance ratio test.

    Args:
        series (np.ndarray): Time series data.
        N (int): Lag value.

    Returns:
        float: v_N statistic or NaN if data is insufficient.
    """

    b_tau_values = np.array([calculate_b_tau(series, tau) for tau in range(1, N)])
    N_tau = np.arange(N - 1, 0, -1)
    v_N = (4 / N ** 2) * np.nansum((N_tau ** 2) * b_tau_values)
    return v_N

def calculate_cross_vr_hat(series1, series2, N):

    """Computes cross-series VR_hat statistic between two time series.

    Args:
        series1 (np.ndarray): First time series.
        series2 (np.ndarray): Second time series.
        N (int): Lag value.

    Returns:
        float: Cross VR_hat statistic or NaN if data is insufficient.
    """

    if len(series1) < N or len(series2) < N:
        return np.nan
    rhos = np.array([autocorr_vec(series1, series2, tau) for tau in range(1, N)])
    N_tau = np.arange(N - 1, 0, -1)
    VR_hat_cross = 1 + (2 / N) * np.nansum(N_tau * rhos)
    return VR_hat_cross

def run_vr_test_parallel(args):

    """Computes VR statistic and z-statistic for a pair of series (or single series) in parallel.

    Args:
        args (tuple): Contains data (pd.DataFrame), label (str), file_name (str),
                      Z_values (list), col1 (str), col2 (str or None).

    Returns:
        list: List of dictionaries with VR statistics results.
    """

    data, label, file_name, Z_values, col1, col2 = args
    result_rows = []
    series1 = data[col1].dropna().values
    series2 = data[col2].dropna().values if col2 is not None else None

    if col2 is None or col1 == col2:

        for N in Z_values:
            VR_hat_self = calculate_vr_hat(series1, N)
            v_N_self = calculate_v_N(series1, N)
            denom = np.sqrt(v_N_self / len(series1)) if v_N_self and len(series1) else np.nan
            z_N_self = (VR_hat_self - 1) / denom if denom else np.nan
            result_rows.append({
                "Filename": file_name,
                "Frequency": label,
                "Series 1": col1,
                "Series 2": col1,
                "N": N,
                "z_N": z_N_self
            })
    else:

        for N in Z_values:
            VR_hat_cross = calculate_cross_vr_hat(series1, series2, N)
            v_N_cross = calculate_v_N(series1, N) + calculate_v_N(series2, N)
            denom = np.sqrt(v_N_cross / min(len(series1), len(series2))) if v_N_cross and min(len(series1), len(series2)) else np.nan
            z_N_cross = (VR_hat_cross - 1) / denom if denom else np.nan
            result_rows.append({
                "Filename": file_name,
                "Frequency": label,
                "Series 1": col1,
                "Series 2": col2,
                "N": N,
                "z_N": z_N_cross
            })
    return result_rows

def run_vr_test(return_data, label, file_name, comparison_data, Z_values=[2, 5, 50], n_jobs=None):

    """Runs variance ratio tests for return types and appends results to shared data list.

    Args:
        return_data (pd.DataFrame): Data containing time series returns.
        label (str): Type of return (e.g., 'Return', 'Abs Return').
        file_name (str): Name of the file being processed.
        comparison_data (list): List to accumulate results.
        Z_values (list, optional): List of lag values for VR tests. Defaults to [2, 5, 50].
        n_jobs (int, optional): Number of parallel jobs. Defaults to None.

    Returns:
        None
    """

    numerical_columns = return_data.columns
    all_args = []
  
    for col1, col2 in combinations(numerical_columns, 2):
        all_args.append((return_data, label, file_name, Z_values, col1, col2))
 
    for col in numerical_columns:
        all_args.append((return_data, label, file_name, Z_values, col, None))
  
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for result in tqdm(executor.map(run_vr_test_parallel, all_args), total=len(all_args), desc=f"VR test {label} {file_name}", leave=False):
            comparison_data.extend(result)

def calc_vr_statistic_stats(source_dir, n_jobs=None):

    """Main function to compute VR statistics on all CSV files in the source directory.

    Args:
        source_dir (str): Directory containing CSV files.
        n_jobs (int, optional): Number of parallel jobs. Defaults to None.

    Returns:
        None
    """

    start_time = time.time()
    comparison_data = []
    file_list = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    for file_name in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(source_dir, file_name)
        try:
            df = load_timeseries_data(file_path)
            if df.empty:
                logger.warning(f"Skipping empty file: {file_name}")
                continue
            returns = np.log10(df / df.shift(1)).dropna()
            abs_returns = returns.abs()
            squared_returns = returns ** 2
            run_vr_test(returns, 'Return', file_name, comparison_data, n_jobs=n_jobs)
            run_vr_test(abs_returns, 'Abs Return', file_name, comparison_data, n_jobs=n_jobs)
            run_vr_test(squared_returns, 'Squared Return', file_name, comparison_data, n_jobs=n_jobs)
        except Exception as e:
            logger.error(f"Failed VR test for {file_name}: {e}", exc_info=True)
            continue
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by=['Filename', 'Series 1', 'Series 2'])
    comparison_csv_path = os.path.join(VR_STAT_DIR, 'cross_VR_statistic_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Comparison table saved to {comparison_csv_path}")
    total_time = time.time() - start_time
    logger.info(f"Time taken to run the VR-statistic code: {total_time:.2f} seconds")


