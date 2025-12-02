
__all__ = ["LoggerManager"]


import os
import sys
import logging
import inspect

from limelight.api import parse_config
from limelight.utils import get_caller_path, import_rich

CONFIG = parse_config()
PROJECT_ROOT = CONFIG.PROJECT_ROOT
_, _, RichHandler, _, _, _ = import_rich()


class LoggerManagerInstance:
    """
    高級 Logging 管理器
    - 支援多個 Logger 與多級日誌輸出
    - 支援動態變更 Log Level 與日誌格式
    - 支援按模型功能建立獨立的 logger
    - 支援手動初始化時設定 save dir，並同步更新所有 logger
    """
    _instance = None

    @staticmethod
    def get_instance():
        if LoggerManagerInstance._instance is None:
            # 預設先不傳 log_dir，待手動初始化後更新
            LoggerManagerInstance._instance = LoggerManagerInstance()
        return LoggerManagerInstance._instance

    def __init__(self, log_dir=None):
        if LoggerManagerInstance._instance is not None:
            raise Exception("請使用 `LoggerManagerInstance.get_instance()` 獲取實例。")

        # 初始設定，可從 CONFIG 取得預設值
        self.log_dir = log_dir if log_dir is not None else CONFIG.get("log_dir", "logs")
        self.log_level = CONFIG.get("log_level", "INFO").upper()
        self.log_format = CONFIG.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.rich_log_format = "[%(name)s] - %(message)s"
        self.date_format = CONFIG.get("date_format", "%Y-%m-%d %H:%M:%S")
        self.log_filename = os.path.join(self.log_dir, "log.txt")

        # 建立 handlers 列表（先建立 stream handler；file handler 會根據 log_dir 建立）
        self.handlers = []
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(getattr(logging, self.log_level, logging.INFO))

        rich_handler = RichHandler() if RichHandler is not None else None
        rich_handler.setLevel(getattr(logging, self.log_level, logging.INFO))

        self.handlers.extend([stream_handler, rich_handler])
        #self._create_file_handler()

        # 儲存所有 logger 的字典
        self.loggers = {}

    def _create_file_handler(self):
        """依據當前 log_dir 建立 FileHandler，並加入 handlers 列表"""
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_filename = os.path.join(self.log_dir, "log.txt")
        file_handler = logging.FileHandler(self.log_filename, encoding="utf-8")
        file_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
        self.handlers.append(file_handler)

    def set_save_dir(self, log_dir):
        """
        動態更新 save dir，並同步更新所有 logger 的 FileHandler。
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        new_log_filename = os.path.join(self.log_dir, "log.txt")
        self.log_filename = new_log_filename

        # 對於內部的 handlers，如果有 FileHandler 則替換掉
        new_handlers = []
        update = False
        for handler in self.handlers:
            if isinstance(handler, logging.FileHandler):
                update = True
                # 將原 logger 上的舊 file handler 移除
                for logger in self.loggers.values():
                    logger.removeHandler(handler)
                # 新建 FileHandler
                new_file_handler = logging.FileHandler(new_log_filename, encoding="utf-8")
                new_file_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
                new_file_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
                new_handlers.append(new_file_handler)
                # 同步更新已存在的 logger
                for logger in self.loggers.values():
                    logger.addHandler(new_file_handler)
            else:
                new_handlers.append(handler)

        self.handlers = new_handlers
        if not update:
            self._create_file_handler()
        print(f"[Logger] Save_dir 已更新為：{self.log_dir}")

    def get_logger(self, name=None, rich_stout=False):
        """
        根據呼叫者路徑自動產生 logger，如不存在則建立新 logger。
        """
        if name is None:
            name = get_caller_path(frame=inspect.stack()[1], return_right=False)

        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, self.log_level, logging.INFO))
            formatter = logging.Formatter(self.log_format, self.date_format)
            rich_formatter = logging.Formatter(self.rich_log_format, self.date_format)

            for handler in self.handlers:
                if handler in logger.handlers:
                    continue

                _formatter = None
                if isinstance(handler, RichHandler) and rich_stout:
                    _formatter = rich_formatter
                elif type(handler) is logging.StreamHandler and not rich_stout:
                    _formatter = formatter
                elif not (type(handler) in [logging.StreamHandler, RichHandler]):
                    _formatter = formatter

                if _formatter is not None:
                    handler.setFormatter(_formatter)
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
        if date_format is not None:
            self.date_format = date_format
        formatter = logging.Formatter(self.log_format, self.date_format)
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        print(f"[Logger] 日誌格式已變更")

    def add_handler(self, handler_type, **kwargs):
        """動態增加新的 Handler（支援 stream, file, http, syslog）"""
        handler = None
        if handler_type == "stream":
            handler = logging.StreamHandler(sys.stdout)
        elif handler_type == "file":
            filename = kwargs.get("filename", os.path.join(self.log_dir, "custom_log.log"))
            handler = logging.FileHandler(filename, encoding="utf-8")
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
            self.handlers.append(handler)
            #print(f"[Logger] 新增 `{handler_type}` 處理器")

LoggerManager = LoggerManagerInstance.get_instance()
