import os
import pandas as pd
import time
import logging
import logging.config
import yaml
from datetime import datetime
from pathlib import Path
from src.config.constants import (
    LOGGING_FILE,
    RAW_DATA,
    RAW_STATS_OUTPUT_DIR,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


os.makedirs(RAW_STATS_OUTPUT_DIR, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads a timeseries CSV file, identifying the correct header row by searching for columns such as 'date' and 'temp'.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame containing the timeseries data.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start_line = 0
    for i, line in enumerate(lines):
        if "date" in line.lower() and "temp" in line.lower():
            data_start_line = i
            break

    df = pd.read_csv(file_path, skiprows=data_start_line, comment='#', parse_dates=True)
    return df

def summarize_raw_file(file_path, output_dir):
    """
    Summarizes a raw timeseries data CSV, computing missing/zero/duplicate statistics and saving the results with metadata.

    Args:
        file_path (str): Path to the raw CSV file.
        output_dir (str): Directory where the summary statistics CSV will be saved.

    Returns:
        None
    """
    start_time = time.time()
    file_name = os.path.basename(file_path)
    try:
        df = load_timeseries_data(file_path)
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return

    missing_values = df.isnull().sum()
    zero_values = (df == 0).sum()
    duplicates = df.duplicated().sum()
    null_percentage = (df.isna().sum() / len(df)) * 100
    zero_percentage = (df.isin([0]).sum() / len(df)) * 100
    num_rows, num_cols = df.shape

    summary_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Zero Values': zero_values,
        'Duplicates': [duplicates] + [None] * (len(missing_values) - 1),
        'Percentage_Missing_Values': null_percentage.round(1),
        'Percentage_Zeros': zero_percentage.round(2)
    })

    summary_df.loc['Number of Rows', :] = [num_rows, None, None, None, None]
    summary_df.loc['Number of Columns', :] = [num_cols, None, None, None, None]

    try:
        station_id = df['Station_ID'].iloc[3]
    except Exception:
        station_id = file_name.split('.')[0]

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_taken = time.time() - start_time

    metadata_df = pd.DataFrame({
        'File Name': [file_name],
        'File Path': [file_path],
        'Date of Execution': [current_time],
        'Time Taken (seconds)': [f"{time_taken:.2f}"],
        'Data': ['Raw'],
        'Code': ['Python'],
        'Study': ['Volatility Study Code - Tarun Singh']
    })

    output_file_name = f"{station_id}_Uncleaned_Stats.csv"
    output_file_path = os.path.join(output_dir, output_file_name)

    full_output_df = pd.concat([summary_df, metadata_df.T])
    try:
        full_output_df.to_csv(output_file_path, index=True)
        logger.info(f"Saved summary for {file_name} as {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save summary for {file_name}: {e}")

def generate_raw_data_stats():
    """
    Iterates over all CSV files in the raw data directory and generates a detailed summary statistics
    CSV for each file, including per-column missing/zero value statistics, duplicates, shape,
    and a metadata footer. Logs progress and errors.

    Returns:
        None
    """
    start_time = time.time()
    for file_name in os.listdir(RAW_DATA):
        if file_name.endswith('.csv'):
            file_path = os.path.join(RAW_DATA, file_name)
            summarize_raw_file(file_path, RAW_STATS_OUTPUT_DIR)
    total_time_taken = time.time() - start_time
    logger.info(f"Total time taken to process all files: {total_time_taken:.2f} seconds")


