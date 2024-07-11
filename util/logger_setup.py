'''
This script sets up the logger for the training script. The logger logs messages to a file and to the console.
The logger has the following configuration:
    - The logger logs messages at the INFO level and above.
    - The logger logs messages to a file named 'log/training.log'.
    - The logger logs messages to the console.
'''

import logging
from datetime import datetime

def setup_logger():
    # Create a logger
    logger_file_path = f'log/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(logger_file_path)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Logging format: timestamp - log level - message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()