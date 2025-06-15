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