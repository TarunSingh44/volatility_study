import os
import pandas as pd
import numpy as np
import time
import logging
import logging.config
import yaml
from tqdm import tqdm
from src.config.constants import (
    LOGGING_FILE,
    Q_STAT_DIR,
    FREQUENCIES,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(Q_STAT_DIR, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads a CSV, finds the datetime column, sets it as index, and keeps only numeric columns.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame with datetime index and only numeric columns.
    """
    try:
        t1 = time.time()
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
        t2 = time.time()
        logger.info(f"Loaded {file_path} in {t2-t1:.2f} seconds")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

def cross_sample_autocorrelation(series1, series2, tau):
    """
    Computes the cross-sample autocorrelation between two series at a given lag.

    Args:
        series1 (np.ndarray): First numeric series.
        series2 (np.ndarray): Second numeric series.
        tau (int): Lag value.

    Returns:
        float: Cross-sample autocorrelation, or np.nan if denominator is zero.
    """
    r_bar1 = series1.mean()
    r_bar2 = series2.mean()
    numerator = np.sum((series1[:-tau] - r_bar1) * (series2[tau:] - r_bar2))
    denominator = np.sqrt(np.sum((series1 - r_bar1) ** 2) * np.sum((series2 - r_bar2) ** 2))
    return numerator / denominator if denominator != 0 else np.nan

def cross_sample_Q_statistic(returns, max_tau):
    """
    Calculates the cross-sample Q-statistic for all pairs of columns in returns.

    Args:
        returns (pd.DataFrame): DataFrame of returns.
        max_tau (int): Maximum lag to compute.

    Returns:
        dict: Dictionary with keys (col1, col2) and values as lists of Q-statistics for each lag.
    """
    n = len(returns)
    Q_values = {}
    columns = returns.columns
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i <= j:
                Q_k_values = []
                for k in range(1, max_tau + 1):
                    Q_k = 0
                    for tau in range(1, k + 1):
                        series1 = returns[col1].dropna().values
                        series2 = returns[col2].dropna().values
                        if len(series1) > tau and len(series2) > tau:
                            autocorr = cross_sample_autocorrelation(series1, series2, tau)
                            if not np.isnan(autocorr):
                                Q_k += autocorr ** 2
                    Q_k_values.append(Q_k * n)
                Q_values[(col1, col2)] = Q_k_values
    return Q_values

def calc_q_statistic_stats(
    source_dir,
    max_tau=30
):
    """
    Computes and saves the cross-sample Q-statistic for each file and frequency.

    Args:
        source_dir (str): Directory containing input files.
        max_tau (int): Maximum lag for Q-statistic.

    Returns:
        None
    """
    file_list = os.listdir(source_dir)
    file_list = [file for file in file_list if os.path.isfile(os.path.join(source_dir, file))]
    comparison_data = []
    start_time = time.time()

    for file_name in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(source_dir, file_name)
        try:
            df = load_timeseries_data(file_path)
            if df.empty:
                logger.warning(f"Skipping empty file: {file_name}")
                continue
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}", exc_info=True)
            continue

        for freq_name, freq in FREQUENCIES.items():
            try:
                if freq and isinstance(df.index, pd.DatetimeIndex):
                    df_resampled = df.resample(freq).mean()
                else:
                    df_resampled = df
                returns = np.log10(df_resampled / df_resampled.shift(1)).dropna()
                cross_Q_statistic_results = {}
                q_stat_dict = cross_sample_Q_statistic(returns, max_tau)
                for (col1, col2), Q_values in tqdm(q_stat_dict.items(), desc=f"Q-stat {file_name}-{freq_name}", leave=False):
                    cross_Q_statistic_results[(col1, col2)] = Q_values
                cross_Q_statistic_df = pd.DataFrame(cross_Q_statistic_results, index=range(1, max_tau + 1))
                q_stat_path = os.path.join(Q_STAT_DIR, f'{os.path.splitext(file_name)[0]}_cross_Q_statistic_{freq_name}.csv')
                cross_Q_statistic_df.to_csv(q_stat_path)
                logger.info(f"Q-statistic for {file_name} at {freq_name} frequency saved to {q_stat_path}")

                for (col1, col2), q_values in cross_Q_statistic_results.items():
                    for lag, q_value in enumerate(q_values, start=1):
                        comparison_data.append({
                            "Filename": file_name,
                            "Frequency": freq_name,
                            "Series 1": col1,
                            "Series 2": col2,
                            "Lag": lag,
                            "Q-value": q_value
                        })
            except Exception as e:
                logger.error(f"Error computing Q-stat for {file_name} at {freq_name}: {e}", exc_info=True)
                continue

    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(Q_STAT_DIR, 'cross_Q_statistic_comparison.csv')
    comparison_df = comparison_df.sort_values(by=['Filename', 'Series 1'], ascending=True)
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Comparison table saved to {comparison_csv_path}")

    total_time = time.time() - start_time
    logger.info(f"Time taken to run the Q-statistic code: {total_time:.2f} seconds")


