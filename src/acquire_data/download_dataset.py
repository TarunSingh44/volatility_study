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

if os.path.exists(LOGGING_FILE):
    with open(LOGGING_FILE, 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def download_and_unzip_from_gdrive(gdrive_id, dest_dir):
    """
    Downloads a zip file from Google Drive using its file ID, unzips it to the specified destination directory, 
    and removes the downloaded zip file.

    Args:
        gdrive_id (str): The unique file ID of the file on Google Drive.
        dest_dir (str): The local directory path where the contents will be extracted.

    Raises:
        Exception: If any error occurs during download or extraction, the exception is logged and re-raised.
        
    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        output_path = os.path.join(dest_dir, "downloaded_file.zip")
        url = f"https://drive.google.com/uc?id={gdrive_id}"

        logger.info(f"Downloading file from Google Drive ID: {gdrive_id}")
        gdown.download(url, output_path, quiet=False)
        logger.info(f"Downloaded to {output_path}")

        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        logger.info(f"Extracted contents to {dest_dir}")

        os.remove(output_path)
        logger.info(f"Removed downloaded zip file: {output_path}")

    except Exception as e:
        logger.error(f"Error in download_and_unzip_from_gdrive: {e}", exc_info=True)
        raise

