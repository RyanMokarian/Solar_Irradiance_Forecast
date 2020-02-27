import logging
import absl.logging

LOG_FILENAME = 'logs.txt'

# Remove useless warning. This is updated in latest tensorflow version.
logging.root.removeHandler(absl.logging._absl_handler) # https://github.com/abseil/abseil-py/issues/99
absl.logging._warn_preinit_stderr = False # https://github.com/abseil/abseil-py/issues/102
 
def get_logger():
    # Get logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # Define handlers
    file_handler = logging.FileHandler(LOG_FILENAME, 'w+')
    stream_handler = logging.StreamHandler()

    # Set levels
    stream_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handler
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger