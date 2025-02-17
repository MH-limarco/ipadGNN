

import random
import argparse
import torch
import numpy as np
from collections.abc import Iterable
from typing import Any

from limelight.api import parse_config

config = parse_config()
DEFAULT_SEED = config.seed

def fix_seed(seed=DEFAULT_SEED):
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