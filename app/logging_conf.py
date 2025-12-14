import logging
import sys

def get_logger():
    logger = logging.getLogger("heart-api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger
