

import random
import argparse
import torch
import numpy as np

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
