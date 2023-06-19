import functools
import time
from collections.abc import Callable
import os
import os.path as osp
import tempfile

from rich import print as rprint


def timeit(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func}: {end - start}s')
        return result
    return wrapper


def with_timer(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        wrapper.n_calls += 1
        wrapper.total_time += end - start
        return result
    wrapper.n_calls = 0
    wrapper.total_time = 0
    return wrapper


def is_path_writable(path: str) -> bool:
    abs_path = osp.abspath(osp.expandvars(osp.expanduser(path)))
    try:
        with tempfile.TemporaryFile(dir=osp.dirname(abs_path)):
            pass
        return True
    except OSError:
        return False