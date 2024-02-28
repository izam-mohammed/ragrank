import logging
import os
import sys

LOGGING_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "ragrank_logs.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("ragrank_logger")
