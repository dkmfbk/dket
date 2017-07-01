"""Logging utilities."""

import logging

HDEBUG = 9
logging.addLevelName(HDEBUG, 'HDEBUG')

FMT = '%(asctime)s %(levelname).1s %(funcName)s\t%(filename)s:%(lineno)s\t%(message)s'
FORMATTER = logging.Formatter(FMT)


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

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level)

    fhand = logging.FileHandler(fpath, mode='a')
    fhand.setLevel(level)
    fhand.setFormatter(FORMATTER)
    logger.addHandler(fhand)

    # Default stream handler logging.
    shand = logging.StreamHandler()
    shand.setLevel(logging.WARNING)
    if stderr and level is not logging.WARNING:
        shand.setLevel(level)
    shand.setFormatter(FORMATTER)
    logger.addHandler(shand)
