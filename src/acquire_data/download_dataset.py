import os
import gdown
import zipfile
import logging
import logging.config
import yaml
from pathlib import Path
from src.config.constants import(
    LOGGING_FILE,
)

# Logging Setup 
if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# Data Download & Unzip Function
def download_and_unzip_from_gdrive(gdrive_id, dest_dir):
    """
    Downloads a file from Google Drive and unzips it to the destination directory.
    Args:
        gdrive_id (str): Google Drive file ID.
        dest_dir (str): Destination directory for unzipped files.
    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        output_path = os.path.join(dest_dir, "downloaded_file.zip")
        url = f"https://drive.google.com/uc?id={gdrive_id}"

        logger.info(f"Downloading file from Google Drive ID: {gdrive_id}")
        gdown.download(url, output_path, quiet=False)
        logger.info(f"Downloaded to {output_path}")

        # Unzip
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        logger.info(f"Extracted contents to {dest_dir}")

        # Remove zip file if desired
        os.remove(output_path)
        logger.info(f"Removed downloaded zip file: {output_path}")

    except Exception as e:
        logger.error(f"Error in download_and_unzip_from_gdrive: {e}", exc_info=True)
        raise

