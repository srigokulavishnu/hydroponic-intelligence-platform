import os
from pathlib import Path

# Base directory is the directory containing this script
BASE_DIR = Path(__file__).parent.absolute()

API_URL = "http://192.168.100.131:5000/api/data"
COLLECTION_INTERVAL = 60
REQUEST_TIMEOUT = 10

# Directory paths
DATA_DIRECTORY = BASE_DIR / "data"
LOG_DIRECTORY = BASE_DIR / "logs"

CSV_FILENAME = "sensor_environment.csv"
LOG_FILENAME = "sensor_logger.log"
