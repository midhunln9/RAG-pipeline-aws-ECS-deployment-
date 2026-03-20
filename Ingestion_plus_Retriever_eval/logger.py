import logging
import os
from datetime import datetime

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("DocumentIngestion") # without mentioning the name, it will default to root logger, which is same as logging.basicConfig()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if logger.handlers:
        return

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)