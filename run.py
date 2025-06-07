import yaml
import logging
from src.config.constants import(
    LOGGING_FILE,
    RAW_DATA_DIR,
    GDRIVE_ID,
)
from src.acquire_data.download_dataset import download_and_unzip_from_gdrive

# Set up logging from YAML config
with open(LOGGING_FILE, 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    ############################################################
    ##### STEP 1: Acquire Raw Data Files from Google Drive #####
    ############################################################

    logger.info("Step 1: Starting to download data from Google Drive")
    download_and_unzip_from_gdrive(GDRIVE_ID, RAW_DATA_DIR)
    logger.info("Step 1: All raw data files acquired successfully.")

