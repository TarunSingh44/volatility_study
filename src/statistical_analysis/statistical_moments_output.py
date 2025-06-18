import os
import time
import logging
import logging.config
import pandas as pd
import numpy as np
import yaml

from src.config.constants import (
    LOGGING_FILE,
)

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_timeseries_data(file_path, freq=None):
    """Load time series data from a CSV file and optionally resample.

    Args:
        file_path (str): Path to the CSV file.
        freq (str, optional): Frequency string for resampling (pandas style).

    Returns:
        tuple: DataFrame of time series data, time taken to load (float).
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    try:
        df = pd.read_csv(file_path, comment='#', parse_dates=['Date_Time'], index_col='Date_Time')
        df = df.select_dtypes(include=[float, int])
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]

        if freq:
            df = df.resample(freq).mean().dropna()

        end_time = time.time()
        time_taken = end_time - start_time
        logger.info(f"Loaded {file_path} with freq '{freq}' in {time_taken:.2f}s")
        return df, time_taken
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}", exc_info=True)
        return pd.DataFrame(), 0.0


def calculate_statistical_moments(df):
    """Calculate basic statistical moments for a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame of calculated moments, or None if empty.
    """
    logger = logging.getLogger(__name__)
    try:
        if df is not None and not df.empty:
            moments = pd.DataFrame({
                "Min": df.min(),
                "Max": df.max(),
                "Mean": df.mean(),
                "Variance": df.var(),
                "Standard Deviation": df.std(),
                "Kurtosis": df.kurt(),
                "Skewness": df.skew()
            })
            return moments.T
        logger.warning("DataFrame is empty. No moments calculated.")
        return None
    except Exception as e:
        logger.error("Error calculating statistical moments.", exc_info=True)
        return None


def calculate_return(df):
    """Calculate log returns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame of log returns.
    """
    logger = logging.getLogger(__name__)
    try:
        returns = np.log10(df / df.shift(1)).dropna()
        return returns
    except Exception as e:
        logger.error("Error calculating return.", exc_info=True)
        return pd.DataFrame()


def calculate_sq_return(df):
    """Calculate squared log returns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame of squared log returns.
    """
    logger = logging.getLogger(__name__)
    try:
        returns = np.log10(df / df.shift(1)).dropna()
        returns = returns.pow(2).dropna()
        return returns
    except Exception as e:
        logger.error("Error calculating squared return.", exc_info=True)
        return pd.DataFrame()


def calculate_abs_return(df):
    """Calculate absolute log returns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame of absolute log returns.
    """
    logger = logging.getLogger(__name__)
    try:
        returns = np.log10(df / df.shift(1)).dropna()
        returns = returns.abs().dropna()
        return returns
    except Exception as e:
        logger.error("Error calculating absolute return.", exc_info=True)
        return pd.DataFrame()


def calculate_log_abs_return(df):
    """Calculate log of absolute deviation of returns from mean.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame of log(abs(r - avg[r])).
    """
    logger = logging.getLogger(__name__)
    try:
        returns = np.log10(df / df.shift(1)).dropna()
        mean_return = returns.mean()
        returns = np.log10(np.abs(returns - mean_return)).dropna()
        return returns
    except Exception as e:
        logger.error("Error calculating log absolute return.", exc_info=True)
        return pd.DataFrame()


def save_with_metadata(df, output_file, metadata_df):
    """Save DataFrame to CSV with metadata row at the bottom.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_file (str): Output file path.
        metadata_df (dict): Metadata to append as last row.
    """
    logger = logging.getLogger(__name__)
    try:
        metadata_row = pd.DataFrame(metadata_df, index=[df.shape[1]])
        combined_df = pd.concat([df, metadata_row], axis=0)
        combined_df.to_csv(output_file)
        logger.info(f"Saved with metadata to {output_file}")
    except Exception as e:
        logger.error(f"Error saving {output_file}: {e}", exc_info=True)


def output_stats_moments(input_dir, output_actual_dir, output_return_dir, output_sq_return_dir, output_abs_return_dir, output_log_abs_return_dir, frequencies):
    """Process all CSV files in the input directory for various statistical moments.

    Args:
        input_dir (str): Input directory with CSV files.
        output_actual_dir (str): Directory for actual moments.
        output_return_dir (str): Directory for return moments.
        output_sq_return_dir (str): Directory for squared return moments.
        output_abs_return_dir (str): Directory for absolute return moments.
        output_log_abs_return_dir (str): Directory for log(abs(r - avg[r])) moments.
        frequencies (dict): Dictionary of label to frequency strings.
    """
    logger = logging.getLogger(__name__)

    os.makedirs(output_actual_dir, exist_ok=True)
    os.makedirs(output_return_dir, exist_ok=True)
    os.makedirs(output_sq_return_dir, exist_ok=True)
    os.makedirs(output_abs_return_dir, exist_ok=True)
    os.makedirs(output_log_abs_return_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not filename.endswith(".csv"):
            logger.info(f"Skipping non-csv file: {filename}")
            continue

        for freq_label, freq in frequencies.items():
            try:
                df, time_taken = load_timeseries_data(file_path, freq)

                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                metadata_df = {
                    'File Name': filename,
                    'File Path': file_path,
                    'Frequency': freq_label,
                    'Date of Execution': current_time,
                    'Row Count': df.shape[0],
                    'Column Count': df.shape[1],
                    'Time Taken (seconds)': f"{time_taken:.2f}",
                    'Data': 'Cleaned',
                    'Code': 'Python',
                    'Study': 'Volatility Study Code - Tarun Singh'
                }

                moments_actual = calculate_statistical_moments(df)
                output_actual_file = os.path.join(output_actual_dir, f"stats_moments_{freq_label}_{filename}")
                if moments_actual is not None:
                    save_with_metadata(moments_actual, output_actual_file, metadata_df)

                df_return = calculate_return(df)
                moments_return = calculate_statistical_moments(df_return)
                output_return_file = os.path.join(output_return_dir, f"stats_moments_return_{freq_label}_{filename}")
                if moments_return is not None:
                    save_with_metadata(moments_return, output_return_file, metadata_df)

                df_sq_return = calculate_sq_return(df)
                moments_sq_return = calculate_statistical_moments(df_sq_return)
                output_sq_return_file = os.path.join(output_sq_return_dir, f"stats_moments_sq_return_{freq_label}_{filename}")
                if moments_sq_return is not None:
                    save_with_metadata(moments_sq_return, output_sq_return_file, metadata_df)

                df_abs_return = calculate_abs_return(df)
                moments_abs_return = calculate_statistical_moments(df_abs_return)
                output_abs_return_file = os.path.join(output_abs_return_dir, f"stats_moments_abs_return_{freq_label}_{filename}")
                if moments_abs_return is not None:
                    save_with_metadata(moments_abs_return, output_abs_return_file, metadata_df)

                df_log_abs_return = calculate_log_abs_return(df)
                moments_log_abs_return = calculate_statistical_moments(df_log_abs_return)
                output_log_abs_return_file = os.path.join(output_log_abs_return_dir, f"stats_moments_log_abs_return_{freq_label}_{filename}")
                if moments_log_abs_return is not None:
                    save_with_metadata(moments_log_abs_return, output_log_abs_return_file, metadata_df)

            except Exception as e:
                logger.error(f"Exception during processing {filename} ({freq_label}): {e}", exc_info=True)