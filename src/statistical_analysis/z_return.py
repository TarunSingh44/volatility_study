import os
import pandas as pd
import numpy as np
import time
from itertools import combinations
from tqdm import tqdm
import logging
import logging.config
import yaml
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
    """
    Loads a CSV, finds the datetime column, sets it as index, and keeps only numeric columns.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame with datetime index and only numeric columns.
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

def calculate_vr_hat(series, N):
    """
    Calculates VR_hat for a time series and lag N.

    Args:
        series (np.ndarray): Numeric series.
        N (int): Lag value.

    Returns:
        float: VR_hat value.
    """
    mu = np.mean(series)
    rho_hat = [np.corrcoef(series[:-tau], series[tau:])[0, 1] for tau in range(1, N)]
    VR_hat = 1 + (2 / N) * np.sum([(N - tau) * rho for tau, rho in enumerate(rho_hat, start=1)])
    return VR_hat

def calculate_b_tau(series, tau):
    """
    Helper for VR variance: autocovariance at lag tau.

    Args:
        series (np.ndarray): Numeric series.
        tau (int): Lag.

    Returns:
        float: b_tau value.
    """
    s_t = (series - np.mean(series)) ** 2
    numerator = np.sum([s_t[t] * s_t[t + tau] for t in range(len(s_t) - tau)])
    denominator = np.sum(s_t) ** 2
    b_tau = (len(series) * numerator) / denominator
    return b_tau

def calculate_v_N(series, N):
    """
    Calculates v_N for VR variance.

    Args:
        series (np.ndarray): Numeric series.
        N (int): Lag.

    Returns:
        float: v_N value.
    """
    b_tau_values = [calculate_b_tau(series, tau) for tau in range(1, N)]
    v_N = (4 / N ** 2) * np.sum([(N - tau) ** 2 * b_tau for tau, b_tau in enumerate(b_tau_values, start=1)])
    return v_N

def calculate_cross_vr_hat(series1, series2, N):
    """
    Calculates cross VR_hat for two time series and lag N.

    Args:
        series1 (np.ndarray): Numeric series 1.
        series2 (np.ndarray): Numeric series 2.
        N (int): Lag.

    Returns:
        float: cross VR_hat value.
    """
    rho_hat_cross = [np.corrcoef(series1[:-tau], series2[tau:])[0, 1] for tau in range(1, N)]
    VR_hat_cross = 1 + (2 / N) * np.sum([(N - tau) * rho for tau, rho in enumerate(rho_hat_cross, start=1)])
    return VR_hat_cross

def run_vr_test(return_data, label, file_name, comparison_data, Z_values=[2, 5, 50]):
    """
    Computes VR statistics for all combinations of columns in return_data.

    Args:
        return_data (pd.DataFrame): DataFrame of returns/abs returns/squared returns.
        label (str): Label for type ("Return", "Abs Return", etc.).
        file_name (str): Filename of the input file.
        comparison_data (list): List to collect all results.
        Z_values (list): Lags to use.

    Returns:
        None
    """
    numerical_columns = return_data.columns
    total_combinations = len(list(combinations(numerical_columns, 2))) + len(numerical_columns)
    with tqdm(total=total_combinations, desc=f"VR test {label} {file_name}", ncols=100, leave=False) as pbar:
        # Cross series
        for col1, col2 in combinations(numerical_columns, 2):
            series1 = return_data[col1].dropna().values
            series2 = return_data[col2].dropna().values
            for N in Z_values:
                VR_hat_cross = calculate_cross_vr_hat(series1, series2, N)
                v_N_cross = calculate_v_N(series1, N) + calculate_v_N(series2, N)
                z_N_cross = (VR_hat_cross - 1) / np.sqrt(v_N_cross / min(len(series1), len(series2)))
                comparison_data.append({
                    "Filename": file_name,
                    "Frequency": label,
                    "Series 1": col1,
                    "Series 2": col2,
                    "N": N,
                    "z_N": z_N_cross
                })
            pbar.update(1)
        # Self series
        for col in numerical_columns:
            series = return_data[col].dropna().values
            for N in Z_values:
                VR_hat_self = calculate_vr_hat(series, N)
                v_N_self = calculate_v_N(series, N)
                z_N_self = (VR_hat_self - 1) / np.sqrt(v_N_self / len(series))
                comparison_data.append({
                    "Filename": file_name,
                    "Frequency": label,
                    "Series 1": col,
                    "Series 2": col,
                    "N": N,
                    "z_N": z_N_self
                })
            pbar.update(1)

def calc_vr_statistic_stats(source_dir):
    """
    Computes and saves the VR-statistic for each file in source_dir.

    Args:
        source_dir (str): Directory with cleaned input files.

    Returns:
        None
    """
    start_time = time.time()
    comparison_data = []
    file_list = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    for file_name in tqdm(file_list, desc="Processing files", ncols=100):
        file_path = os.path.join(source_dir, file_name)
        try:
            df = load_timeseries_data(file_path)
            if df.empty:
                logger.warning(f"Skipping empty file: {file_name}")
                continue
            returns = np.log10(df / df.shift(1)).dropna()
            abs_returns = returns.abs()
            squared_returns = returns ** 2
            run_vr_test(returns, 'Return', file_name, comparison_data)
            run_vr_test(abs_returns, 'Abs Return', file_name, comparison_data)
            run_vr_test(squared_returns, 'Squared Return', file_name, comparison_data)
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

