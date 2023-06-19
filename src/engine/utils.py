from collections import deque
import random
import os
import logging
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger()

class AverageMeter:
    def __init__(self):
        self._buffer = []

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, val: float) -> None:
        self._buffer.append(val)

    @property
    def val(self) -> float:
        return self._buffer[-1]

    def avg(self, window_size:Optional[int] = None) -> float:
        if window_size is None:
            return np.mean(self._buffer)
        else:
            return np.mean(self._buffer[-window_size:])


def unwrap_model(model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel) -> torch.nn.Module:
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def set_random_seed(seed: int = 2022, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}.")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("The CUDNN is set to deterministic. This will increase reproducibility, "
                    "but may slow down the training considerably.")


class PlaceholderModule:
    def __init__(self):
        self.dummy_func = lambda *args, **kwargs: None

    def __getattr__(self, name: str) -> Callable:
        return self.dummy_func