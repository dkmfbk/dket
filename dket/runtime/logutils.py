"""Logging utilities."""

import logging

HDEBUG = 9
logging.addLevelName(HDEBUG, 'HDEBUG')

FMT = '%(asctime)s\t%(levelname)s\t%(funcName)s\t%(filename)s:%(lineno)s\t%(message)s'
FORMATTER = logging.Formatter(FMT)

# Default stream handler logging.
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.WARNING)
STREAM_HANDLER.setFormatter(FORMATTER)
logging.getLogger().addHandler(STREAM_HANDLER)


def parse_level(level):
    """Parse a string into a log level."""
    if not level:
        raise ValueError('`level` must be a valid string.')
    if level == 'NOTSET':
        level = HDEBUG
    elif level == 'DEBUG':
        level = logging.DEBUG
    elif level == 'INFO':
        level = logging.INFO
    elif level == 'WARNING':
        level = logging.WARNING
    elif level == 'HDEBUG':
        level = HDEBUG
    else:
        raise ValueError(
            """Invalid log level: """ + level)
    return level

def _validate(level):
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return parse_level(level)
    return parse_level(str(level))


def config(level=logging.DEBUG, fpath='.log', stderr=False):
    """Configure the default logging infrastructure."""

    level = _validate(level)

    logging.getLogger().setLevel(0)
    fhand = logging.FileHandler(fpath, mode='a')
    fhand.setLevel(level)
    fhand.setFormatter(FORMATTER)
    logging.getLogger().addHandler(fhand)

    # If required, fix the stream handler log level.
    if stderr and level is not logging.WARNING:
        STREAM_HANDLER.setLevel(level)
