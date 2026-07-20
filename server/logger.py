import logging
import os
from logging.handlers import RotatingFileHandler


LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


logger = logging.getLogger("RAG_SYSTEM")

logger.setLevel(logging.INFO)


formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


file_handler = RotatingFileHandler(
    "logs/rag_system.log",
    maxBytes=5 * 1024 * 1024,     ## Once the log file reaches 5 MB, Python creates a new log file.
)

file_handler.setFormatter(formatter)


console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
#level : INFO, DEBUG, WARNING, ERROR, CRITICAL
logger.addHandler(file_handler)
logger.addHandler(console_handler)