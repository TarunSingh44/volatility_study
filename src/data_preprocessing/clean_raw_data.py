import os
import pandas as pd
import time
import logging
import logging.config
import yaml
from pathlib import Path
from src.config.constants import (
    LOGGING_FILE,
    RAW_DATA,
    CLEANED_DATA,
    CLEANED_STATS,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

os.makedirs(CLEANED_DATA, exist_ok=True)
os.makedirs(CLEANED_STATS, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads a timeseries CSV file starting from the line containing column headers.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe containing the timeseries data.
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

def clean_dataframe(df, breakdown):
    """
    Cleans a DataFrame by removing columns and rows with excessive NaNs or zeros.

    Steps:
      - Drops columns with more than 50% NaN values.
      - Drops rows containing any NaN values.
      - Drops columns where more than 50% of values are 0.
      - Drops rows containing any zero values.

    Args:
        df (pd.DataFrame): The input dataframe to clean.
        breakdown (dict): Dictionary for recording the row counts after each cleaning step.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    initial_rows = len(df)
    breakdown['initial_rows'] = initial_rows

    # Drop columns with too many NaNs
    threshold = len(df) * 0.5
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    breakdown['after_drop_columns_NaN'] = len(df_cleaned)

    # Drop rows with any NaNs
    df_cleaned = df_cleaned.dropna()
    breakdown['after_drop_rows_NaN'] = len(df_cleaned)

    # Drop columns where 50%+ values are 0
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).mean() > 0.5]
    breakdown['after_drop_columns_zero'] = len(df_cleaned)

    # Drop rows with any 0s
    df_cleaned = df_cleaned[(df_cleaned != 0).all(axis=1)]
    breakdown['after_drop_rows_zero'] = len(df_cleaned)

    return df_cleaned


def raw_data_cleaning():
    """
    Orchestrates the process of loading, cleaning, and saving timeseries data,
    and saving the cleaning statistics for each file.

    Workflow:
      - Iterates over CSV files in the source directory.
      - Loads each file as a DataFrame.
      - Cleans the DataFrame and tracks stats at each step.
      - Saves cleaned data and stats CSV for each input file.
      - Logs all steps and time taken.
    """
    start_time = time.time()

    for filename in os.listdir(RAW_DATA):
        file_path = os.path.join(RAW_DATA, filename)
        if filename.endswith('.csv'):
            logger.info(f"Processing file: {filename}")
            breakdown = {}

            # Load
            df = load_timeseries_data(file_path)
            breakdown['loaded'] = len(df)

            # Clean
            df_cleaned = clean_dataframe(df, breakdown)
            cleaned_filename = f"Cleaned_{filename}"
            cleaned_file_path = os.path.join(CLEANED_DATA, cleaned_filename)

            # Save and reload for a final zero-row check
            df_cleaned.to_csv(cleaned_file_path, index=False)
            df_final = pd.read_csv(cleaned_file_path)
            before_final_zero = len(df_final)
            df_final = df_final[(df_final != 0).all(axis=1)]
            after_final_zero = len(df_final)
            df_final.to_csv(cleaned_file_path, index=False)

            # Prepare stats
            stats_dict = {
                "Original_rows": breakdown.get('initial_rows', 0),
                "After_drop_NaN_columns": breakdown.get('after_drop_columns_NaN', 0),
                "After_drop_NaN_rows": breakdown.get('after_drop_rows_NaN', 0),
                "After_drop_zero_columns": breakdown.get('after_drop_columns_zero', 0),
                "After_drop_zero_rows": breakdown.get('after_drop_rows_zero', 0),
                "After_final_zero_rows": after_final_zero,
                "Rows_dropped_total": breakdown.get('initial_rows', 0) - after_final_zero
            }
            stats_filename = f"Stats_{os.path.splitext(filename)[0]}.csv"
            stats_file_path = os.path.join(CLEANED_STATS, stats_filename)
            pd.DataFrame([stats_dict]).to_csv(stats_file_path, index=False)
            logger.info(f"Saved cleaned file: {cleaned_file_path}")
            logger.info(f"Saved stats file: {stats_file_path}")

    end_time = time.time()
    time_taken = end_time - start_time
    logger.info(f"Time taken to run the code: {time_taken:.2f} seconds")

