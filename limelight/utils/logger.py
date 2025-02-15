

import os
import sys
import logging
from datetime import datetime

from limelight.api import parse_config

class LoggerManager:
    """
    高級 Logging 管理器
    - 支援多個 Logger
    - 支援動態變更 Log Level
    - 支援 `config.yaml` 設定 Logging 參數
    - 支援多種輸出格式與 Handler
    """
    _instance = None

    @staticmethod
    def get_instance():
        if LoggerManager._instance is None:
            LoggerManager._instance = LoggerManager()
        return LoggerManager._instance

    def __init__(self):
        if LoggerManager._instance is not None:
            raise Exception("Please use `LoggerManager.get_instance()` to get the instance.")

        config = parse_config()


