import logging
import sys
import os
import os.path as osp
from typing import Optional

from termcolor import colored

class _ColorfulFormatter(logging.Formatter):

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "magenta")
        elif record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(
    name: Optional[str] = None, 
    log_level: int = logging.INFO, 
    rank: int = 0, 
    color: bool = True, 
    output_dir: Optional[str] = None
) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(levelname)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
    color_formatter = _ColorfulFormatter(
        fmt=colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s", 
        datefmt="%m/%d %H:%M:%S"
    )

    # hack, may not be safe
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
        
    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(color_formatter if color else formatter)
        logger.addHandler(console_handler)

    # save to file 
    if rank == 0 and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(osp.join(output_dir, f"rank{rank}.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f'Logger({name}) initialized.')
    return logger