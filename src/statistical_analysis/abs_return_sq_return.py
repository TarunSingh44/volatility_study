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
    ABS_RETURN_DIR,
    SQUARED_RETURN_DIR,
    ABS_RETURN_PLOTS_DIR,
    SQUARED_RETURN_PLOTS_DIR,
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
    Ensures that all required output directories exist.

    Args:
        plot (bool): Whether to create plot directories as well.
    """
    dirs = [
        ABS_RETURN_DIR, SQUARED_RETURN_DIR
    ]
    if plot:
        dirs += [ABS_RETURN_PLOTS_DIR, SQUARED_RETURN_PLOTS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads a CSV file, infers and sets the datetime index, and keeps only numeric columns.

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

def save_stat_and_plot(df, csv_path, plot_path_dir, base_file_name, freq_name, stat_type, ylabel, plot=True):
    """
    Saves a DataFrame as a CSV and (optionally) generates/saves a plot for each column.

    Args:
        df (pd.DataFrame): DataFrame to save and plot.
        csv_path (str): Path to save CSV file.
        plot_path_dir (str): Directory to save plots.
        base_file_name (str): Base for plot filenames.
        freq_name (str): Frequency name.
        stat_type (str): Type of statistic (used for filename/legend).
        ylabel (str): Y-axis label for plots.
        plot (bool): Whether to generate plots.
    """
    try:
        df.to_csv(csv_path, index=True)
        logger.info(f"Saved {stat_type} ({freq_name}) to {csv_path}")
        if plot:
            for column in df.columns:
                plt.figure(figsize=(12, 7))
                plt.plot(df.index, df[column], label=stat_type.replace("_", " ").title())
                plt.title(f'{column} - {stat_type.replace("_", " ").title()} ({freq_name})')
                plt.xlabel('Date/Time')
                plt.xticks(rotation=45)
                plt.ylabel(ylabel)
                plt.legend()
                plot_path = os.path.join(plot_path_dir, f'{base_file_name}_{column}_{stat_type}_{freq_name}.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved plot for {stat_type} of {column} at {freq_name} to {plot_path}")
    except Exception as e:
        logger.error(f"Failed saving {stat_type} or plot for {base_file_name} ({freq_name}): {e}", exc_info=True)

def calc_abs_squared_return_stats(
    source_dir, 
    plot=True
):
    """
    Computes and saves absolute and squared returns for each file and frequency. Optionally saves plots.

    Args:
        source_dir (str): Directory containing input files.
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
                abs_returns = returns.abs().dropna()
                squared_returns = returns.pow(2).dropna()

                abs_return_path = os.path.join(ABS_RETURN_DIR, f'{base_file_name}_abs_return_{freq_name}.csv')
                squared_return_path = os.path.join(SQUARED_RETURN_DIR, f'{base_file_name}_squared_return_{freq_name}.csv')

                save_stat_and_plot(
                    abs_returns,
                    abs_return_path,
                    ABS_RETURN_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    "abs_return",
                    "Absolute Returns Value",
                    plot=plot
                )
                save_stat_and_plot(
                    squared_returns,
                    squared_return_path,
                    SQUARED_RETURN_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    "squared_return",
                    "Squared Returns Value",
                    plot=plot
                )
        except Exception as e:
            logger.error(f"Failed processing {file_name}: {e}", exc_info=True)
    total_time = time.time() - start_time
    logger.info(f"Time taken to run the abs/squared returns code: {total_time:.2f} seconds")



