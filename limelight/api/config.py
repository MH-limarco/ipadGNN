
__all__ = ["parse_config"]

import os
import inspect
from functools import lru_cache
from limelight.configs import cached_parse_config

def parse_config():
    """解析設定檔，並回傳 ConfigWrapper 實例"""
    caller_frame = inspect.stack()[1]
    caller_path = caller_frame.filename

    relative_path = os.path.relpath(caller_path, os.path.join(os.path.dirname(__file__), "../"))
    module_name = relative_path.replace(os.sep, ".").rsplit(".", 1)[0]
    return cached_parse_config(module_name)


if __name__ == "__main__":
    print(dir(parse_config()))
    print(parse_config()["CONFIGS_DIR"])
