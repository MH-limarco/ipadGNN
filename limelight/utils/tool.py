

__all__ = ["fix_seed", "parse_args", "set_attrs", "get_caller_path",
           "import_rich", "import_plotext", "get_func_args"]

import os
import random
import argparse
import inspect
import torch
import numpy as np
from collections.abc import Iterable
from typing import Any


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args(args):
    if isinstance(args, argparse.Namespace):
        return vars(args)
    elif isinstance(args, dict):
        return args
    raise ValueError(f"Unknown type of args: {type(args)}")

def set_attrs(obj: object, names: [str, list], values: [Any, list]):
    _iterable = [isinstance(names, Iterable), isinstance(values, Iterable)]
    if all(_iterable):
        for name, default in zip(names, values):
            setattr(obj, name, default)

    elif not any(_iterable):
        setattr(obj, names, values)

    elif not _iterable[1]:
        for name in names:
            setattr(obj, name, values)

    else:
        raise ValueError("Length of names and values must be the same.")

def get_caller_path(frame, return_right=True):
    caller_path = frame.filename
    relative_path = os.path.relpath(caller_path, os.path.join(os.path.dirname(__file__), "../"))
    if return_right:
        return relative_path.replace(os.sep, ".").rsplit(".", 1)[0]
    return relative_path.replace(os.sep, ".").split(".", 1)[0]

def import_rich():
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.logging import RichHandler
        from rich.layout import Layout
        from rich.text import Text
        import atexit
        import time
        RICH = True

    except ImportError:
        Live, Table, RichHandler, Layout, Text = None, None, None
        RICH = False

    return Live, Table, RichHandler, Layout, Text, RICH

def import_plotext():
    try:
        import plotext as plt
        PLOTEXT = True
    except ImportError:
        plt = None
        PLOTEXT = False

    return plt, PLOTEXT


def get_func_args(func):
    sig = inspect.signature(func)
    parts = []
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{name}")
        else:
            parts.append(name)
    return tuple(parts)