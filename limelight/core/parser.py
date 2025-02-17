import argparse
import json
from types import SimpleNamespace
from limelight.api import parse_config

def str2bool(v):
    """將字串轉換為布林值 (適用於 argparse)"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class ParserWrapper:
    """封裝 `argparse`，統一管理參數解析"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="LIME 實驗參數解析器")
        self._define_arguments()

    def _define_arguments(self):
        """定義 argparse 參數"""
        # ✅ 基本訓練參數
        self.parser.add_argument("--learning_rate", type=float, default=0.001, help="學習率")
        self.parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
        self.parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
        self.parser.add_argument("--seed", type=int, default=42, help="隨機種子")
        self.parser.add_argument("--use_gpu", type=str2bool, default=True, help="是否使用 GPU")

        # ✅ 模型選擇
        self.parser.add_argument("--model_type", type=str, default="GNN", choices=["GNN", "MLP"], help="選擇模型類型")
        self.parser.add_argument("--filling_method", type=str, default="zero_filling", choices=["zero_filling", "random_filling"], help="缺失值填充方法")
        self.parser.add_argument("--dataset", type=str, default="Cora", help="數據集名稱")

        # ✅ 其他超參數 (未來可擴展)

    def parse(self, cmd_args=None, api_args=None, return_dict=False):
        """
        解析命令列參數 & API 參數
        - cmd_args: 模擬命令列輸入 (`list`)，通常在 CMD 執行時為 `None`
        - api_args: API 內部調用時提供的 `dict`
        - return_dict: 是否回傳 `dict` (預設 `Namespace`)
        """
        # 解析 CMD 輸入
        if cmd_args:
            args = self.parser.parse_args(cmd_args)
        else:
            args = self.parser.parse_args()  # 正常解析 CMD 輸入

        # 轉換成 dict
        args_dict = vars(args)

        # ✅ 合併 API 輸入參數 (如果 API 傳入，覆蓋 CMD)
        if api_args:
            args_dict.update(api_args)

        # ✅ 是否以 dict 形式回傳
        if return_dict:
            return args_dict
        return SimpleNamespace(**args_dict)

# ✅ 創建 `ParserWrapper` 實例
parser_wrapper = ParserWrapper()

# ✅ 對外提供 `get_args()`
def get_args(cmd_args=None, api_args=None, return_dict=False):
    return parser_wrapper.parse(cmd_args, api_args, return_dict)
