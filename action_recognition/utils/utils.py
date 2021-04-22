import os
import sys
import logging
import logging.config
import warnings
import random
from typing import Optional

import numpy as np
import torch

# Initiate Logger
logger = logging.getLogger(__name__)


def setup_logging(log_path: Optional[str] = None, level: str = "DEBUG"):
    handlers_dict = {
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    }

    if log_path is not None:
        safe_dir(log_path, with_filename=True)
        handlers_dict["file_handler"] = {
            "class": "logging.FileHandler",
            "formatter": "full",
            "level": "DEBUG",
            "filename": log_path,
            "encoding": "utf8"
        }

    # Configure logging
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[ %(asctime)s ] %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            },
            "full": {
                "format": "[ %(asctime)s ] %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": handlers_dict,
        "loggers": {
            "action_recognition": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
            "__main__": {
                "level": level,
                "handlers": list(handlers_dict.keys())
            },
        }
    }

    # Deal with dual log issue
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger().handlers[0].setLevel(logging.WARNING)

    # Exception Logging
    sys.excepthook = handle_exception

    warnings.filterwarnings("ignore")

    logging.config.dictConfig(config_dict)
    logger.info("Setup Logging!")


def safe_dir(path: str, with_filename: bool = False) -> str:
    dir_path = os.path.dirname(path) if with_filename else path
    if not os.path.exists(dir_path):
        logger.info("Dir %s not exist, creating directory!", dir_path)
        os.makedirs(dir_path)
    return os.path.abspath(path)


def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def setup_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # TODO
