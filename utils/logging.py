import logging
import absl.logging

# Remove useless warning. This is updated in latest tensorflow version.
logging.root.removeHandler(absl.logging._absl_handler) # https://github.com/abseil/abseil-py/issues/99
absl.logging._warn_preinit_stderr = False # https://github.com/abseil/abseil-py/issues/102
 
def get_logger():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger