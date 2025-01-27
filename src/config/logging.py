import logging
from logging.handlers import RotatingFileHandler


def setup_logger(
    name,
    debug_level=logging.INFO,
    log_file="app.log",
    max_bytes=1048576,
    backup_count=5,
):
    """
    Set up a logger with file rotation.

    :param name: Name of the logger (usually __name__).
    :param debug_level: Logging level (e.g., logging.DEBUG, logging.INFO). Default is logging.INFO.
    :param log_file: Path to the log file. Default is 'app.log'.
    :param max_bytes: Maximum size in bytes before the log file rotates. Default is 1MB.
    :param backup_count: Number of backup log files to keep. Default is 5.
    :return: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(debug_level)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a rotating file handler
    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
