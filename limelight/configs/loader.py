
__all__ = ["CONFIG", "SRC_DIR", "CONFIGS_DIR", "PROJECT_ROOT"]

import yaml
import pathlib
from os import path, getenv


from limelight.configs.wrapper import ConfigWrapper

SRC_DIR = "limelight"
CONFIGS_DIR = "configs"
YAML_NAME = "magic_number.yaml"

#PROJECT_ROOT = getenv("LIME_DIR", path.abspath( path.join(r"../../", path.dirname(__file__), )))
PROJECT_ROOT = getenv("LIME_DIR", pathlib.Path(__file__).resolve().parents[2])
CONFIG_PATH = path.join(PROJECT_ROOT, SRC_DIR, CONFIGS_DIR, YAML_NAME)

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

# 加載配置
CONFIG = load_config()