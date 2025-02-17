

import os
import sys
import logging
from datetime import datetime

from limelight.api import parse_config

config = parse_config()


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


        self.log_dir = config.get("log_dir", "logs")
        self.log_level = config.get("log_level", "INFO").upper()
        self.log_format = config.get("log_format", None)
        self.date_format = config.get("date_format", None)

        #os.makedirs(self.log_dir, exist_ok=True)
        #log_filename = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

        # 建立 `handlers`，預設輸出到終端與 LOG 檔案
        #self.handlers = [logging.StreamHandler(sys.stdout),  # 輸出到終端
        #           logging.FileHandler(log_filename)  # 輸出到檔案
        #]

        # 存放所有 Logger
        self.loggers = {}

    def get_logger(self, name="LIME"):
        """取得 Logger（如果不存在則自動建立）"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, self.log_level, logging.INFO))

            # 設定 Formatter
            formatter = logging.Formatter(self.log_format, self.date_format)
            for handler in self.handlers:
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            self.loggers[name] = logger

        return self.loggers[name]

    def set_level(self, level):
        """動態修改全局 Logger 等級"""
        self.log_level = level.upper()
        for logger in self.loggers.values():
            logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        print(f"[Logger] 日誌等級已變更為 `{self.log_level}`")

    def set_format(self, log_format, date_format=None):
        """動態修改全局 Logger 格式"""
        self.log_format = log_format
        self.date_format = date_format
        formatter = logging.Formatter(self.log_format, self.date_format)
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        print(f"[Logger] 日誌格式已變更")

    def add_handler(self, handler_type, **kwargs):
        """動態增加新的 Handler（支援 `stream`, `file`, `http`, `syslog`）"""
        handler = None

        if handler_type == "stream":
            handler = logging.StreamHandler(sys.stdout)
        elif handler_type == "file":
            filename = kwargs.get("filename", "logs/custom_log.log")
            handler = logging.FileHandler(filename)
        elif handler_type == "http":
            from logging.handlers import HTTPHandler
            handler = HTTPHandler(kwargs.get("host"), kwargs.get("url"), method="POST")
        elif handler_type == "syslog":
            from logging.handlers import SysLogHandler
            handler = SysLogHandler(address=(kwargs.get("host", "localhost"), kwargs.get("port", 514)))

        if handler:
            handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
            for logger in self.loggers.values():
                logger.addHandler(handler)
            print(f"[Logger] 新增 `{handler_type}` 處理器")


# ✅ 初始化 LoggerManager
logger_manager = LoggerManager.get_instance()
logger = logger_manager.get_logger()
logger.info("Logging system initialized!")


