import os
import pandas as pd
import numpy as np
import time
import logging
import logging.config
import yaml
from datetime import datetime
from src.config.constants import (
    LOGGING_FILE,
    CROSS_CORR_DIR,
    FREQUENCIES, 
    CATEGORIES, 
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(CROSS_CORR_DIR, exist_ok=True)

def load_timeseries_data(file_path):
    """
    Loads a CSV file, infers and sets the datetime index, and keeps only numeric columns.

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
    n = len(series1)
    r_bar1 = series1.mean()
    r_bar2 = series2.mean()
    numerator = np.sum((series1[:-tau] - r_bar1) * (series2[tau:] - r_bar2))
    denominator = np.sqrt(np.sum((series1 - r_bar1) ** 2) * np.sum((series2 - r_bar2) ** 2))
    return numerator / denominator if denominator != 0 else np.nan

def categorize_autocorrelation(autocorr):
    """
    Categorizes a correlation value into bins.

    Args:
        autocorr (float): Correlation value.

    Returns:
        str: Category label.
    """
    if autocorr < -0.1:
        return '(i) ρ < -0.1'
    elif -0.1 <= autocorr < -0.05:
        return '(ii) -0.1 ≤ ρ < -0.05'
    elif -0.05 <= autocorr < 0:
        return '(iii) -0.05 ≤ ρ < 0'
    elif 0 <= autocorr < 0.05:
        return '(iv) 0 ≤ ρ < 0.05'
    elif 0.05 <= autocorr < 0.1:
        return '(v) 0.05 ≤ ρ < 0.1'
    else:
        return '(vi) 0.1 ≤ ρ'

def calc_cross_auto_corr_stats(
    source_dir,
    lags=range(1, 31)
):
    """
    Computes cross-sample autocorrelation tables (and category table) for each input file and frequency.

    Args:
        source_dir (str): Directory containing input files.
        lags (range): Lags to compute for.
        plot (bool): Placeholder (plots not implemented in this function).

    Returns:
        None
    """

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
                if freq and isinstance(df.index, pd.DatetimeIndex):
                    df_resampled = df.resample(freq).mean()
                else:
                    df_resampled = df

                returns = np.log10(df_resampled / df_resampled.shift(1)).dropna()
                columns = returns.columns
                results = []

                for i, col1 in enumerate(columns):
                    for j, col2 in enumerate(columns):
                        if i <= j:
                            for tau in lags:
                                series1 = returns[col1].dropna().values
                                series2 = returns[col2].dropna().values
                                if len(series1) > tau and len(series2) > tau:
                                    cross_corr_value = cross_sample_autocorrelation(series1, series2, tau)
                                    category = categorize_autocorrelation(cross_corr_value)
                                    results.append([col1, col2, tau, cross_corr_value, category])

                cross_auto_corr_df = pd.DataFrame(
                    results,
                    columns=['Column 1', 'Column 2', 'Lag', 'Cross-Auto-Correlation', 'Category']
                )
                table1 = cross_auto_corr_df.pivot_table(
                    index='Lag', columns=['Column 1', 'Column 2'], values='Cross-Auto-Correlation'
                ).T
                table2 = cross_auto_corr_df.groupby(['Column 1', 'Column 2', 'Category']).size().unstack(fill_value=0)
                table2 = table2.reindex(columns=CATEGORIES, fill_value=0)

                table1_path = os.path.join(CROSS_CORR_DIR, f'{base_file_name}_cross_auto_corr_table1_{freq_name}.csv')
                table2_path = os.path.join(CROSS_CORR_DIR, f'{base_file_name}_cross_auto_corr_table2_{freq_name}.csv')
                table1.to_csv(table1_path)
                table2.to_csv(table2_path)
                logger.info(f"Table1 for {file_name} at {freq_name} frequency saved to {table1_path}")
                logger.info(f"Table2 for {file_name} at {freq_name} frequency saved to {table2_path}")
        except Exception as e:
            logger.error(f"Failed processing {file_name}: {e}", exc_info=True)
    total_time = time.time() - start_time
    logger.info(f"Time taken to run the cross-auto-correlation code: {total_time:.2f} seconds")

