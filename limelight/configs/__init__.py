__all__ = ["CONFIG", "SRC_DIR", "CONFIGS_DIR", "PROJECT_ROOT"]

import yaml
from os import path, getenv

# 讀取 YAML 配置檔
SRC_DIR = "limelight"
CONFIGS_DIR = "configs"
YAML_NAME = "magic_number.yaml"

PROJECT_ROOT = getenv("LIME_DIR", path.abspath(path.join(path.dirname(__file__), "../")))
CONFIG_PATH = path.join(PROJECT_ROOT, CONFIGS_DIR, YAML_NAME)

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

# 加載配置
CONFIG = load_config()
