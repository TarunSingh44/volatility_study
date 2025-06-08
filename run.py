import yaml
import logging
from src.config.constants import(
    LOGGING_FILE,
    RAW_DATA,
    GDRIVE_ID,
)
from src.acquire_data.download_dataset import download_and_unzip_from_gdrive
from src.data_preprocessing.raw_data_stats import generate_raw_data_stats
from src.data_preprocessing.clean_raw_data import raw_data_cleaning


with open(LOGGING_FILE, 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    ############################################################
    ##### STEP 1: Acquire Raw Data Files from Google Drive #####
    ############################################################

    logger.info("Step 1: Starting to download data from Google Drive")
    download_and_unzip_from_gdrive(GDRIVE_ID, RAW_DATA)
    logger.info("Step 1: All raw data files acquired successfully.")

    ############################################################
    ##### STEP 2: Get Raw Data Stats                       #####
    ############################################################

    logger.info("Step 2: Generating Raw Data Stats")
    generate_raw_data_stats()
    logger.info("Step 2: Raw Data Stats Generated.")

    ############################################################
    ##### STEP 3: Clean Data & Save Statistics             #####
    ############################################################

    logger.info("Step 3: Starting Cleaning Process")
    raw_data_cleaning()
    logger.info("Step 3: Data Cleaning Completed.")