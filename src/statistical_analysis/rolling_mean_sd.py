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
    ROLLING_MEAN_DIR, 
    ROLLING_STD_DIR, 
    ROLLING_MEAN_PLOTS_DIR, 
    ROLLING_STD_PLOTS_DIR,
    ROLLING_MEAN_RETURN_DIR, 
    ROLLING_STD_RETURN_DIR, 
    ROLLING_MEAN_RETURN_PLOTS_DIR, 
    ROLLING_STD_RETURN_PLOTS_DIR,
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
    Ensures all output directories exist. If plot is False, skips plot directories.

    Args:
        plot (bool): Whether to create plot directories.
    """
    dirs = [
        ROLLING_MEAN_DIR, ROLLING_STD_DIR,
        ROLLING_MEAN_RETURN_DIR, ROLLING_STD_RETURN_DIR
    ]
    if plot:
        dirs += [
            ROLLING_MEAN_PLOTS_DIR, ROLLING_STD_PLOTS_DIR,
            ROLLING_MEAN_RETURN_PLOTS_DIR, ROLLING_STD_RETURN_PLOTS_DIR
        ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads and processes timeseries data. Sets the first datetime-like column as index, keeps only numeric data.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with datetime index.
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

def calculate_returns(df):
    """
    Calculates log10 returns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame of numeric time series.

    Returns:
        pd.DataFrame: DataFrame of log returns.
    """
    try:
        returns = np.log10(df / df.shift(1)).dropna()
        return returns
    except Exception as e:
        logger.error(f"Error calculating returns: {e}", exc_info=True)
        return pd.DataFrame()

def save_rolling_stats_and_plot(rolling_df, out_csv, out_plot_dir, base_file_name, freq_name, stat_type, plot=True):
    """
    Saves rolling statistics as CSV and optionally generates/saves individual plots for each column.

    Args:
        rolling_df (pd.DataFrame): DataFrame with rolling stat values.
        out_csv (str): Path to save the rolling stat CSV.
        out_plot_dir (str): Directory to save the plots.
        base_file_name (str): Base name for output files.
        freq_name (str): Frequency label.
        stat_type (str): Statistic type for labeling.
        plot (bool): Whether to generate plots.

    Returns:
        None
    """
    try:
        rolling_df.to_csv(out_csv, index=True)
        logger.info(f"Saved rolling {stat_type} ({freq_name}) to {out_csv}")
        if plot:
            for column in rolling_df.columns:
                plt.figure(figsize=(12, 7))
                plt.plot(rolling_df.index, rolling_df[column], label=f'Rolling {stat_type.replace("_", " ").title()}')
                plt.title(f'{column} - Rolling {stat_type.replace("_", " ").title()} ({freq_name})')
                plt.xlabel('Date/Time')
                plt.xticks(rotation=45)
                plt.ylabel(f'Rolling {stat_type.replace("_", " ").title()}')
                plt.legend()
                plot_path = os.path.join(out_plot_dir, f'{base_file_name}_{column}_rolling_{stat_type}_{freq_name}.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved plot for rolling {stat_type} of {column} at {freq_name} to {plot_path}")
    except Exception as e:
        logger.error(f"Failed saving rolling {stat_type} or plots for {base_file_name} ({freq_name}): {e}", exc_info=True)

def calc_rolling_mean_sd(source_dir, window_size=5, plot=True):
    """
    Main routine to compute and save rolling mean/std and (optionally) their plots for both actual values and returns.

    Args:
        source_dir (str): Directory with cleaned data files.
        window_size (int): Rolling window size.
        plot (bool): Whether to generate and save rolling plots.

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
                # Resample if frequency is set
                df_resampled = df.resample(freq).mean() if freq and isinstance(df.index, pd.DatetimeIndex) else df
                # Rolling mean/std (actual)
                rolling_mean = df_resampled.rolling(window=window_size).mean().dropna()
                rolling_std = df_resampled.rolling(window=window_size).std().dropna()
                save_rolling_stats_and_plot(
                    rolling_mean,
                    os.path.join(ROLLING_MEAN_DIR, f'{base_file_name}_rolling_mean_{freq_name}.csv'),
                    ROLLING_MEAN_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    stat_type="mean",
                    plot=plot
                )
                save_rolling_stats_and_plot(
                    rolling_std,
                    os.path.join(ROLLING_STD_DIR, f'{base_file_name}_rolling_std_{freq_name}.csv'),
                    ROLLING_STD_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    stat_type="std",
                    plot=plot
                )

                # Rolling mean/std (returns)
                returns = calculate_returns(df_resampled)
                rolling_mean_return = returns.rolling(window=window_size).mean().dropna()
                rolling_std_return = returns.rolling(window=window_size).std().dropna()
                save_rolling_stats_and_plot(
                    rolling_mean_return,
                    os.path.join(ROLLING_MEAN_RETURN_DIR, f'{base_file_name}_rolling_mean_return_{freq_name}.csv'),
                    ROLLING_MEAN_RETURN_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    stat_type="mean_return",
                    plot=plot
                )
                save_rolling_stats_and_plot(
                    rolling_std_return,
                    os.path.join(ROLLING_STD_RETURN_DIR, f'{base_file_name}_rolling_std_return_{freq_name}.csv'),
                    ROLLING_STD_RETURN_PLOTS_DIR,
                    base_file_name,
                    freq_name,
                    stat_type="std_return",
                    plot=plot
                )
        except Exception as e:
            logger.error(f"Failed processing {file_name}: {e}", exc_info=True)
    total_time = time.time() - start_time
    logger.info(f"Time taken to run the rolling stats code: {total_time:.2f} seconds")

