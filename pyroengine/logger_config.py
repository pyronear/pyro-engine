import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Define the logging format
log_format = "%(asctime)s | %(levelname)s: %(message)s"
os.makedirs(os.path.dirname("/var/log/engine.log"), exist_ok=True)

# Create a StreamHandler (for stdout)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))

# Create a TimedRotatingFileHandler (for writing logs to a new file every day)
file_handler = TimedRotatingFileHandler("/var/log/engine.log", when="midnight", interval=1, backupCount=5)
file_handler.setFormatter(logging.Formatter(log_format))
file_handler.suffix = "%Y-%m-%d"  # Adds a date suffix to the log file name

# Export the logger instance
logger = logging.getLogger()  # Get the root logger

# Ensure we only add handlers once
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.INFO)
logger.info("Logger is set up correctly.")
