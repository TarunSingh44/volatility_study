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
STATS_OUTPUT_DIR = os.path.join(RAW_DATA_DIR, "raw_stats")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
CLEANED_DATA = os.path.join(CLEANED_DATA_DIR, "cleaned_data")
CLEANED_STATS = os.path.join(CLEANED_DATA_DIR, "cleaned_stats")
STATS_DIR = os.path.join(DATA_DIR, "volatility_stats")

# Google Drive ID
GDRIVE_ID = "17qFL4wZ_GbjuPgBQ2psA5RXSKnETIRX3"



