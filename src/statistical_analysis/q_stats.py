import os
import pandas as pd
import numpy as np
import time
import logging
import logging.config
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    """Read a CSV file and return a cleaned numeric DataFrame indexed by datetime.

    The function
    1. Detects a column whose name contains "date" or "time".
    2. Parses that column to ``datetime64[ns]`` and sets it as the index.
    3. Drops non-numeric columns and rows with NaT in the index.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        A ``pd.DataFrame`` with a ``DatetimeIndex`` and only numeric columns.
        If an error occurs, an **empty** DataFrame is returned.

    Raises:
        ValueError: If no obvious date/time column is present in the CSV.
        All other exceptions are caught, logged, and converted to an empty
        DataFrame so batch processing continues.
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

def _calc_q_pair(args):

    """Compute cumulative Q-statistic sequence for one column pair.

    This helper is designed for ``ProcessPoolExecutor`` mapping, so the
    signature is a single *tuple* of arguments.

    Args:
        args: Tuple ``(col1, col2, s1, s2, n, max_tau)`` where

            * ``col1, col2`` – column names (str)
            * ``s1, s2``     – centred series as 1-D ``np.ndarray``
            * ``n``          – sample length (int)
            * ``max_tau``    – maximum lag to include (int)

    Returns:
        2-tuple:
            * key ``(col1, col2)``
            * list of Q-statistics for lags ``1 … max_tau`` (length ``max_tau``)
    """

    col1, col2, s1, s2, n, max_tau = args
    ac_sqs = []
    length = len(s1)
    for k in range(1, max_tau + 1):
        Q_k = 0
        for tau in range(1, k + 1):
            if length > tau:
                num = np.dot(s1[:-tau], s2[tau:])
                denom = np.sqrt(np.dot(s1, s1) * np.dot(s2, s2))
                autocorr = num / denom if denom != 0 else np.nan
                if not np.isnan(autocorr):
                    Q_k += autocorr ** 2
        ac_sqs.append(Q_k * n)
    return (col1, col2), ac_sqs

def parallel_cross_sample_Q_statistic(returns, max_tau, n_jobs=None):

    """Compute cross-sample Q-statistics for every column pair in parallel.

    Args:
        returns: DataFrame of *aligned* returns, each column a series.
        max_tau: Largest lag to include in the cumulative statistic.
        n_jobs:  Number of worker processes (``None`` → default by Executor).

    Returns:
        Dictionary mapping ``(col1, col2)`` → list of Q values (index 0 is
        lag 1, … index ``max_tau-1`` is lag ``max_tau``).
    """

    n = len(returns)
    columns = returns.columns
    centered = {col: (returns[col] - returns[col].mean()).values for col in columns}
    tasks = [
        (col1, col2, centered[col1], centered[col2], n, max_tau)
        for i, col1 in enumerate(columns)
        for j, col2 in enumerate(columns) if i <= j
    ]
    Q_values = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(
            executor.map(_calc_q_pair, tasks),
            total=len(tasks),
            desc="Cross Q-stat Pairs",
            leave=False
        ))
    for key, vals in results:
        Q_values[key] = vals
    return Q_values


def calc_q_statistic_stats(
    source_dir,
    max_tau=30,
    n_jobs=None
):
    
    """Batch-process all CSV files in a directory and compute Q-statistics.

    For each file the function:

    1. Loads and cleans the data with :pyfunc:`load_timeseries_data`.
    2. Resamples to every frequency defined in ``FREQUENCIES``.
    3. Calculates log-returns, then cross-sample Q-statistics up to
       ``max_tau`` with :pyfunc:`parallel_cross_sample_Q_statistic`.
    4. Writes a per-file CSV of Q values and aggregates a comparison table.

    Args:
        source_dir: Directory containing the input CSV files.
        max_tau: Maximum lag (positive integer) for the Q statistic.
        n_jobs:  Number of parallel worker processes

    Returns:
        None. (All outputs are files and log messages.)
    """

    file_list = os.listdir(source_dir)
    file_list = [file for file in file_list if os.path.isfile(os.path.join(source_dir, file))]
    comparison_data = []
    start_time = time.time()
    logger.info(f"Number of workers:" + str(n_jobs))

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
                if returns.empty or returns.shape[1] == 0:
                    logger.warning(f"No valid returns for {file_name} at {freq_name}")
                    continue

                q_stat_dict = parallel_cross_sample_Q_statistic(returns, max_tau, n_jobs=n_jobs)
                cross_Q_statistic_df = pd.DataFrame(q_stat_dict, index=range(1, max_tau + 1))
                q_stat_path = os.path.join(Q_STAT_DIR, f'{os.path.splitext(file_name)[0]}_cross_Q_statistic_{freq_name}.csv')
                cross_Q_statistic_df.to_csv(q_stat_path)
                logger.info(f"Q-statistic for {file_name} at {freq_name} frequency saved to {q_stat_path}")

                for (col1, col2), q_values in q_stat_dict.items():
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