# src/srtad/utils/logger.py

import logging
from pathlib import Path

def setup_logger(name: str = "srtad", level: str = "INFO", file: str | None = None):
    """
    Configure the central project logger.

    - Adds a console handler
    - Adds a file handler (optional)
    - Prevents duplicate handlers if called multiple times
    """

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Avoid double handlers if the function is called again
    if logger.handlers:
        return logger

    # Formatter
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level.upper())
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if file:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level.upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
