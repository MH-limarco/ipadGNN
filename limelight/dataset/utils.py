
from os import path
from limelight.api import parse_config

CONFIG = parse_config()
DEFAULT_DIR = CONFIG.dir
PROJECT_ROOT = CONFIG.PROJECT_ROOT


def parse_setting(map, name):
    setting = map[name]
    return setting["func"], setting["transform"]

def parse_data_dir(func_name, name):
    return path.join(PROJECT_ROOT, DEFAULT_DIR, func_name, name)
