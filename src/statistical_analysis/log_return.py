import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import logging.config
import yaml
from datetime import datetime
from src.config.constants import (
    LOGGING_FILE,
    LOG_RETURN_DIR,
    LOG_RETURN_PLOTS_DIR,
    FREQUENCIES,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dirs(plot: bool = True):
    """
    Ensures required output directories exist.

    Args:
        plot (bool): If True, also creates plot directories.
    """
    dirs = [LOG_RETURN_DIR]
    if plot:
        dirs.append(LOG_RETURN_PLOTS_DIR)
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads a CSV, finds the datetime column, sets it as index, and keeps only numeric columns.

    Args:
        file_path (str): Path to CSV.

    Returns:
        pd.DataFrame: DataFrame with datetime index and numeric columns.
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

def save_log_return_and_plot(df, csv_path, plot_path_dir, base_file_name, freq_name, plot=True):
    """
    Saves log return DataFrame as CSV and (optionally) saves a plot for each column.

    Args:
        df (pd.DataFrame): DataFrame of log returns.
        csv_path (str): Path to save CSV.
        plot_path_dir (str): Directory for plots.
        base_file_name (str): Base for filenames.
        freq_name (str): Frequency label.
        plot (bool): Whether to generate plots.
    """
    try:
        df.to_csv(csv_path, index=True)
        logger.info(f"Saved log returns ({freq_name}) to {csv_path}")
        if plot:
            for column in df.columns:
                plt.figure(figsize=(12, 7))
                plt.plot(df.index, df[column], label='Log Returns')
                plt.title(f'{column} - Log Returns ({freq_name})')
                plt.xlabel('Date/Time')
                plt.xticks(rotation=45)
                plt.ylabel('Log Returns Value')
                plt.legend()
                plot_path = os.path.join(plot_path_dir, f'{base_file_name}_{column}_log_return_{freq_name}.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved plot for log returns of {column} at {freq_name} to {plot_path}")
    except Exception as e:
        logger.error(f"Failed saving log returns or plots for {base_file_name} ({freq_name}): {e}", exc_info=True)

def calc_log_return_stats(
    source_dir, 
    plot=True
):
    """
    Computes, saves, and (optionally) plots log returns for each file and frequency.

    Args:
        source_dir (str): Directory with input files.
        plot (bool): Whether to generate plots.

    Returns:
        None
    """
    ensure_dirs(plot=plot)

    start_time = time.time()
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith('.csv'):
            continue
        logger.info(f"Processing file: {file_name}")
        try:
            df = load_timeseries_data(file_path)
            if df.empty:
                logger.warning(f"Skipping empty file: {file_name}")
                continue
            base_file_name = os.path.splitext(file_name)[0]
            for freq_name, freq in FREQUENCIES.items():
                df_resampled = df.resample(freq).mean() if freq and isinstance(df.index, pd.DatetimeIndex) else df
                returns = np.log10(df_resampled / df_resampled.shift(1)).dropna()
                mean_return = returns.mean()
                log_return = np.log10(np.abs(returns - mean_return)).dropna()

                log_return_path = os.path.join(LOG_RETURN_DIR, f'{base_file_name}_log_return_{freq_name}.csv')
                save_log_return_and_plot(
                    log_return,
                    log_return_path,
                    LOG_RETURN_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    plot=plot
                )
        except Exception as e:
            logger.error(f"Failed processing {file_name}: {e}", exc_info=True)
    total_time = time.time() - start_time
    logger.info(f"Time taken to run the log return code: {total_time:.2f} seconds")