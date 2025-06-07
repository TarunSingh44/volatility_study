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

# Clean Raw Data