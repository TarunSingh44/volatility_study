"""Copyright (c) 2025, Trinity College Dublin"""

import os 
from pathlib import Path

BASE_PATH = os.getcwd()
LOGGING_FILE = str(Path(BASE_PATH) / "src" / "config" / "logging.yml")

###################################
# Data Acquire & Cleaning Constants
###################################

# Data Directory
DATA_DIR = os.path.join(BASE_PATH, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
RAW_DATA = os.path.join(RAW_DATA_DIR, "raw_data")
RAW_STATS_OUTPUT_DIR = os.path.join(RAW_DATA_DIR, "raw_stats")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
CLEANED_DATA = os.path.join(CLEANED_DATA_DIR, "cleaned_data")
CLEANED_STATS = os.path.join(CLEANED_DATA_DIR, "cleaned_stats")


# Google Drive ID
GDRIVE_ID = "17qFL4wZ_GbjuPgBQ2psA5RXSKnETIRX3"

###################################
# Volatility Stats Constants
###################################
STATS_DIR = os.path.join(DATA_DIR, "volatility_stats")

FREQUENCIES = {
    'base': None,
    '30mins': '30T',
    '1hr': '1H',
    '12hrs': '12H',
    '1day': '1D',
    '1week': '1W',
    '1month': '1M',
    '1year': '1Y'
}

# Statistical Moments
STATISTICAL_MOMENTS = os.path.join(STATS_DIR, "statistical_moments")
STATISTICAL_MOMENTS_ACTUAL = os.path.join(STATISTICAL_MOMENTS, "actual")
STATISTICAL_MOMENTS_RETURN = os.path.join(STATISTICAL_MOMENTS, "return")

# Correlation 
CORRELATION =  os.path.join(STATS_DIR, "correlation")
CORRELATION_ACTUAL = os.path.join(CORRELATION, "actual")
CORRELATION_RETURN = os.path.join(CORRELATION, "return")

# Rolling stats (data and plots for both actual and returns)
ROLLING_STATS = os.path.join(STATS_DIR, "rolling_stats")
ROLLING_ACTUAL = os.path.join(ROLLING_STATS, "actual")
ROLLING_RETURN = os.path.join(ROLLING_STATS, "return")

ROLLING_MEAN_DIR = os.path.join(ROLLING_ACTUAL, "data", "rolling_mean")
ROLLING_STD_DIR = os.path.join(ROLLING_ACTUAL, "data", "rolling_std")
ROLLING_MEAN_PLOTS_DIR = os.path.join(ROLLING_ACTUAL, "plots", "rolling_mean")
ROLLING_STD_PLOTS_DIR = os.path.join(ROLLING_ACTUAL, "plots", "rolling_std")

ROLLING_MEAN_RETURN_DIR = os.path.join(ROLLING_RETURN, "data", "rolling_mean_return")
ROLLING_STD_RETURN_DIR = os.path.join(ROLLING_RETURN, "data", "rolling_std_return")
ROLLING_MEAN_RETURN_PLOTS_DIR = os.path.join(ROLLING_RETURN, "plots", "rolling_mean_return")
ROLLING_STD_RETURN_PLOTS_DIR = os.path.join(ROLLING_RETURN, "plots", "rolling_std_return")

# Abs/Squared Return - Data and Plots
ABS_SQUARED_RETURN_DIR = os.path.join(STATS_DIR, "abs_squared_return")

ABS_RETURN_DIR = os.path.join(ABS_SQUARED_RETURN_DIR, "data", "abs_return")
SQUARED_RETURN_DIR = os.path.join(ABS_SQUARED_RETURN_DIR, "data", "squared_return")
ABS_RETURN_PLOTS_DIR = os.path.join(ABS_SQUARED_RETURN_DIR, "plots", "abs_return")
SQUARED_RETURN_PLOTS_DIR = os.path.join(ABS_SQUARED_RETURN_DIR, "plots", "squared_return")
