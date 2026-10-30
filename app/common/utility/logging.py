"""Standard-stream logging configuration for the API process."""

import logging
import sys
from typing import TextIO


class MaximumLevelFilter(logging.Filter):
    """Allow records below a maximum severity."""

    def __init__(self, maximum_level: int) -> None:
        super().__init__()
        self.maximum_level = maximum_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether the record belongs on the lower-severity stream."""

        return record.levelno < self.maximum_level


def configure_logging(
    level: int = logging.INFO,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> None:
    """Send informational logs to stdout and errors to stderr."""

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    standard_handler = logging.StreamHandler(stdout or sys.stdout)
    standard_handler.setLevel(level)
    standard_handler.addFilter(MaximumLevelFilter(logging.ERROR))
    standard_handler.setFormatter(formatter)

    error_handler = logging.StreamHandler(stderr or sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(standard_handler)
    root_logger.addHandler(error_handler)
