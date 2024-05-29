# ----- Python Imports
import logging

def getLogger(name=__name__, level=logging.INFO):
    """ Setup Logging
    """
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    return logger
