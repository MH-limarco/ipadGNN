
__all__ = ["parse_config"]

import os
import inspect
from functools import lru_cache
from limelight.configs import *

def parse_config():
    """解析設定檔，並回傳 ConfigWrapper 實例"""
    caller_frame = inspect.stack()[1]
    caller_path = caller_frame.filename

    relative_path = os.path.relpath(caller_path, os.path.join(os.path.dirname(__file__), "../"))
    module_name = relative_path.replace(os.sep, ".").rsplit(".", 1)[0]
    return _cached_parse_config(module_name)


@lru_cache(maxsize=None)
def _cached_parse_config(module_name):
    """
    快取解析結果，避免重複解析相同模組的設定
    """
    cfg = {"SRC_DIR": SRC_DIR, "CONFIGS_DIR": CONFIGS_DIR, "PROJECT_ROOT": PROJECT_ROOT}

    # 解析 YAML 對應的鍵
    keys = module_name.split(".")
    path_cfg = _parse_config(CONFIG, keys)

    if path_cfg:
        cfg.update(path_cfg)

    return ConfigWrapper(cfg)  # 回傳快取的 ConfigWrapper


def _parse_config(cfg, keys):
    result = cfg
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, {})
        else:
            return {}
    return result


if __name__ == "__main__":
    print(dir(parse_config()))
