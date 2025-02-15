
__all__ = ["parse_config"]

import os
import inspect
from functools import lru_cache
from limelight.configs import *


class ConfigWrapper:
    """允許點 (.) 存取的設定類別，並提供方法查詢所有屬性"""
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, name):
        """允許使用 `config.key` 存取變數"""
        if name in self._config_dict:
            value = self._config_dict[name]
            return ConfigWrapper(value) if isinstance(value, dict) else value
        raise AttributeError(f"ConfigWrapper 沒有屬性 '{name}'")

    def __getitem__(self, key):
        """允許 dict 風格存取 (`config['key']`)"""
        return self._config_dict.get(key, None)

    def get(self, key, default=None):
        """提供 dict 的 get() 方法"""
        return self._config_dict.get(key, default)

    def keys(self, recursive=False):
        """
        取得所有可用的屬性：
        - `recursive=False` (預設)：只回傳頂層鍵
        - `recursive=True`：遞迴列出所有巢狀結構中的鍵
        """
        if not recursive:
            return list(self._config_dict.keys())

        def recursive_keys(d, prefix=""):
            keys = []
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                keys.append(full_key)
                if isinstance(v, dict):
                    keys.extend(recursive_keys(v, full_key))
            return keys

        return recursive_keys(self._config_dict)

    def __dir__(self):
        """讓 `dir(config)` 也能列出所有設定屬性"""
        return list(self._config_dict.keys())

    def to_dict(self):
        """轉換回原始 dict"""
        return self._config_dict


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
