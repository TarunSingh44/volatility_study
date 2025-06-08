import os
import pandas as pd
import time
import numpy as np
import logging
import logging.config
import yaml
from datetime import datetime
from pathlib import Path
from src.config.constants import (
    LOGGING_FILE,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_timeseries_data(file_path):
    """
    Loads and preprocesses timeseries data from a CSV file.
    Drops date/time columns and keeps only numeric columns.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: (pd.DataFrame, float) - Cleaned dataframe and time taken to load/process in seconds.
    """
    start_time = time.time()
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data_start_line = 0
        for i, line in enumerate(lines):
            if "date" in line.lower() and "temp" in line.lower():
                data_start_line = i
                break

        df = pd.read_csv(file_path, skiprows=data_start_line, comment='#', parse_dates=True)
        df_cleaned = df.drop(
            columns=[col for col in df.columns if "date" in col.lower() or "time" in col.lower()],
            errors='ignore'
        )
        df_cleaned = df_cleaned.select_dtypes(include=[float, int])
        time_taken = time.time() - start_time

        logger.info(f"Loaded and cleaned file {file_path} in {time_taken:.2f} seconds.")
        return df_cleaned, time_taken

    except Exception as e:
        logger.error(f"Failed to load or clean {file_path}: {e}", exc_info=True)
        return None, 0.0

def calculate_statistical_moments(df):
    """
    Calculates basic statistical moments for a DataFrame: mean, variance,
    standard deviation, kurtosis, and skewness.

    Args:
        df (pd.DataFrame): Numeric dataframe.

    Returns:
        pd.DataFrame: Transposed dataframe of moments (moments as rows).
    """
    try:
        if df is not None and not df.empty:
            moments = pd.DataFrame({
                "Mean": df.mean(),
                "Variance": df.var(),
                "Standard Deviation": df.std(),
                "Kurtosis": df.kurt(),
                "Skewness": df.skew()
            })
            return moments.T
        else:
            logger.warning("Empty dataframe provided to calculate_statistical_moments.")
            return None
    except Exception as e:
        logger.error(f"Error calculating statistical moments: {e}", exc_info=True)
        return None

def calculate_return(df):
    """
    Calculates the log10 returns for a DataFrame.

    Args:
        df (pd.DataFrame): Numeric dataframe.

    Returns:
        pd.DataFrame: DataFrame of log10 returns.
    """
    try:
        returns = np.log10(df / df.shift(1)).dropna()
        return returns
    except Exception as e:
        logger.error(f"Error calculating returns: {e}", exc_info=True)
        return None

def save_with_metadata(df, output_file, metadata_df):
    """
    Saves a DataFrame to CSV and appends metadata as a row at the end.

    Args:
        df (pd.DataFrame): Dataframe to save.
        output_file (str): Output file path.
        metadata_df (dict): Metadata dictionary.

    Returns:
        None
    """
    try:
        metadata_row = pd.DataFrame(metadata_df, index=[df.shape[1]])
        combined_df = pd.concat([df, metadata_row], axis=0)
        combined_df.to_csv(output_file)
        logger.info(f"Saved with metadata to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {output_file}: {e}", exc_info=True)

def calc_stats_moments(input_dir, output_actual_dir, output_return_dir):
    """
    Processes all CSV files in a directory:
      - Loads timeseries data.
      - Calculates statistical moments for actual and return values.
      - Saves results with metadata to output directories.

    Args:
        input_dir (str): Directory with input cleaned CSVs.
        output_actual_dir (str): Output directory for actual value stats.
        output_return_dir (str): Output directory for return value stats.

    Returns:
        None
    """
    os.makedirs(output_actual_dir, exist_ok=True)
    os.makedirs(output_return_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".csv"):
            try:
                df, time_taken = load_timeseries_data(file_path)
                if df is None or df.empty:
                    logger.warning(f"Skipping empty or invalid file: {filename}")
                    continue

                moments_actual = calculate_statistical_moments(df)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata_df = {
                    'File Name': filename,
                    'File Path': file_path,
                    'Date of Execution': current_time,
                    'Row Count': df.shape[0],
                    'Column Count': df.shape[1],
                    'Time Taken (seconds)': f"{time_taken:.2f}",
                    'Data': 'Cleaned',
                    'Code': 'Python',
                    'Study': 'Volatility Study Code - Tarun Singh'
                }

                output_actual_file = os.path.join(output_actual_dir, f"stats_moments_{filename}")
                if moments_actual is not None:
                    save_with_metadata(moments_actual, output_actual_file, metadata_df)

                df_return = calculate_return(df)
                moments_return = calculate_statistical_moments(df_return)
                output_return_file = os.path.join(output_return_dir, f"stats_moments_return_{filename}")
                if moments_return is not None:
                    save_with_metadata(moments_return, output_return_file, metadata_df)

            except Exception as e:
                logger.error(f"Processing failed for file {filename}: {e}", exc_info=True)


