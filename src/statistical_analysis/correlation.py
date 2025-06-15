import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

    Drops columns containing date/time information and keeps only numeric columns.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe containing only numeric data.
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
        elapsed = time.time() - start_time
        logger.info(f"Loaded and cleaned {file_path} in {elapsed:.2f} seconds")
        return df_cleaned
    except Exception as e:
        logger.error(f"Failed to load or clean {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

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
        return pd.DataFrame()

def save_correlation_matrix(df, output_csv_path):
    """
    Saves the correlation matrix of a DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): DataFrame whose correlation matrix will be saved.
        output_csv_path (str): Output path for the correlation CSV.

    Returns:
        None
    """
    try:
        corr_matrix = df.corr()
        corr_matrix.to_csv(output_csv_path)
        logger.info(f"Saved correlation matrix to {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save correlation matrix: {e}", exc_info=True)

def generate_correlation_heatmap(df, output_img_path, title):
    """
    Generates and saves a correlation heatmap image for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame for correlation.
        output_img_path (str): Path to save the heatmap image.
        title (str): Title of the heatmap.

    Returns:
        None
    """
    try:
        corr_matrix = df.corr()
        mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))
        mask_lower = np.tril(np.ones_like(corr_matrix, dtype=bool)) & ~np.eye(corr_matrix.shape[0], dtype=bool)
        plt.figure(figsize=(14, 10))
        cmap_upper = sns.diverging_palette(220, 10, as_cmap=True)
        cmap_lower = sns.diverging_palette(10, 220, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask_upper, cmap='gray', center=0,
                    square=True, linewidths=.5, cbar=False, annot=True)
        sns.heatmap(corr_matrix, mask=mask_lower, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar=True, annot=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_img_path)
        plt.close()
        logger.info(f"Saved correlation heatmap to {output_img_path}")
    except Exception as e:
        logger.error(f"Failed to generate correlation heatmap: {e}", exc_info=True)

def calc_correlation(input_dir, output_actual_dir, output_return_dir):
    """
    Processes each CSV in the input directory:
      - Loads and cleans timeseries data.
      - Computes and saves actual and return correlation matrices (CSV and PNG).

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
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(input_dir, filename)
        try:
            df = load_timeseries_data(file_path)
            if not df.empty:
                output_actual_csv = os.path.join(output_actual_dir, f"corr_matrix_actual_{filename}")
                output_actual_img = os.path.join(output_actual_dir, f"corr_heatmap_actual_{filename}.png")
                save_correlation_matrix(df, output_actual_csv)
                generate_correlation_heatmap(df, output_actual_img, title=f"Correlation Heatmap (Actual) - {filename}")

            df_return = calculate_return(df)
            if not df_return.empty:
                output_return_csv = os.path.join(output_return_dir, f"corr_matrix_return_{filename}")
                output_return_img = os.path.join(output_return_dir, f"corr_heatmap_return_{filename}.png")
                save_correlation_matrix(df_return, output_return_csv)
                generate_correlation_heatmap(df_return, output_return_img, title=f"Correlation Heatmap (Returns) - {filename}")
        except Exception as e:
            logger.error(f"Processing failed for file {filename}: {e}", exc_info=True)

 
