"""Copyright (c) 2025, Trinity College Dublin"""

import os 
from pathlib import Path

BASE_PATH = os.getcwd()
LOGGING_FILE = str(Path(BASE_PATH) / "src" / "config" / "logging.yml")

########################
# Data Acquire Constants
########################

# Data Directory
DATA_DIR = os.path.join(BASE_PATH, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
STATS_DIR = os.path.join(DATA_DIR, "volatility_stats")

# Google Drive ID
GDRIVE_ID = "17qFL4wZ_GbjuPgBQ2psA5RXSKnETIRX3"



