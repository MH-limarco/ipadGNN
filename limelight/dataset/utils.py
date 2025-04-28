
from os import path
from limelight.api import parse_config

CONFIG = parse_config()
DEFAULT_DIR = CONFIG.dir
PROJECT_ROOT = CONFIG.PROJECT_ROOT


def parse_setting(map, name):
    setting = map[name]
    return setting["func"], setting["transform"]

def parse_data_dir(func_name, name):
    raw_data_root = path.join(PROJECT_ROOT, DEFAULT_DIR)
    raw_data_dir = path.join(raw_data_root, func_name, name)
    return raw_data_root, raw_data_dir
