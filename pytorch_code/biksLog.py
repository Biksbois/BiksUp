import logging
from rich.logging import RichHandler
import sys

APP_LOGGER_NAME = 'BiksUP'


def get_logger():
    return setup_applevel_logger(file_name='logfile.log')
    # logging.basicConfig(
    #     level="NOTSET",
    #     format="%(message)s",
    #     datefmt="[%X]",
    #     handlers=[RichHandler(rich_tracebacks=True)]
    # )

    # log = logging.getLogger("rich")
    # return log

def setup_applevel_logger(logger_name = APP_LOGGER_NAME, file_name=None): 
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    
    logger.handlers.clear()
    logger.addHandler(RichHandler(rich_tracebacks=True))
    
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger