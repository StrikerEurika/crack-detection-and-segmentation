import logging
import sys

def setup_logger(name: str = "crack_detection") -> logging.Logger:
    """Sets up a standardized console logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
