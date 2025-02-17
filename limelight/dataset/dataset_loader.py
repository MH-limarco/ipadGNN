

__all__ = ["data_loader"]

import scipy
import torch
import numpy as np
from os import path
from torch_geometric.data import Data

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS
from torch_geometric.datasets import HeterophilousGraphDataset
from ogb.nodeproppred import NodePropPredDataset
import gdown

from limelight.dataset.utils import parse_setting, parse_data_dir

import torch_geometric.transforms as T
DEFAULT_TRANSFORM = T.NormalizeFeatures()


default_api_map = {
    "planetoid": {"func": Planetoid, "transform": None},
    "amazon": {"func": Amazon, "transform": DEFAULT_TRANSFORM},
    "coauthor": {"func": Coauthor, "transform": DEFAULT_TRANSFORM},
    "wikics": {"func": WikiCS, "transform": None},
    "hetero": {"func": HeterophilousGraphDataset, "transform": None},
}

custom_api_map = {
    "ogbn": {"func": NodePropPredDataset, "transform": None},
}


def data_loader(func_name, name):
    if func_name in default_api_map:
        return load_build_in_data(func_name, name)
    elif func_name in custom_api_map:
        return load_custom_data(func_name, name)
    else:
        raise ValueError(f"Invalid dataset name: {name}")

def load_build_in_data(func_name, name):
    func, transform = parse_setting(default_api_map, func_name)
    data_dir = parse_data_dir(func_name, name)
    data = func(root=data_dir, name=name, transform=transform)
    return data[0]

def load_custom_data(func_name, name):
    func, transform = parse_setting(custom_api_map, func_name)
    data_dir = parse_data_dir(func_name, name)
    data = func(root=data_dir, name=name, transform=transform)
    return data[0]


if __name__ == "__main__":
    pass
    print(data_loader("hetero", name='roman-empire'))
