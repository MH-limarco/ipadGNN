import argparse
import json
import torch
from types import SimpleNamespace

__all__ = ["parser_wrapper"]

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
        self.parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
        self.parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
        self.parser.add_argument("--seed", type=int, default=42, help="隨機種子")
        self.parser.add_argument("--use_gpu", type=str2bool, default=True, help="是否使用 GPU")
        self.parser.add_argument("--amp", type=str2bool, default=False, help="是否使用混合精度")


        # ✅ 資料集參數
        self.parser.add_argument("--dataset", type=str, default="Cora", help="數據集名稱")
        self.parser.add_argument("--split_method", type=str, default="random",
                                 choices=["random", "class", "fixed"], help="數據集分割方法")

        # ✅ 分割方法參數
        self.parser.add_argument("--train_prop", type=float, default=0.5, help="random_split-訓練集比例")
        self.parser.add_argument("--valid_prop", type=float, default=0.25, help="random_split驗證集比例")
        self.parser.add_argument("--label_num_per_class", type=int, default=20, help="class_split-每個類別的標籤數")
        self.parser.add_argument("--valid_num", type=int, default=500, help="class_split-驗證集大小")
        self.parser.add_argument("--mask_idx", type=int, default=0, help="fixed_split-遮罩索引")

        # ✅ 缺失值參數
        self.parser.add_argument("--missing_rate", type=float, default=0.0, help="缺失值比例")
        self.parser.add_argument("--missing_type", type=str, default="uniform",
                                 choices=["uniform", "structural"],
                                 help="缺失值類型")
        self.parser.add_argument("--filling_method", type=str, default="FixValueFilling",
                                 choices=["FixValueFilling", "RandomFilling",
                                          "MeanFilling", "NeighborhoodMeanFilling",
                                          "FeaturePropagationFilling"], help="缺失值填充方法")

        # ✅ 模型選擇
        self.parser.add_argument("--model_type", type=str, default="norm", choices=["norm", "boost"], help="選擇模型類型")
        self.parser.add_argument("--conv_type", type=str, default="GCNConv", help="layer類型")
        self.parser.add_argument("--hidden_channels", type=int, default=64, help="隱藏層維度")
        self.parser.add_argument("--num_layers", type=int, default=3, help="層數")
        self.parser.add_argument("--heads", type=int, default=4, help="注意力頭數") # Attention-based GNN
        self.parser.add_argument("--K", type=int, default=1, help="K value") # ChebConv-based GNN
        self.parser.add_argument("--dropout", type=float, default=0.5, help="dropout比例")
        self.parser.add_argument("--res", type=str2bool, default=False, help="是否使用殘差連接")
        self.parser.add_argument("--jk", type=str2bool, default=False, help="是否使用 Jumping Knowledge")
        self.parser.add_argument("--pre_linear", type=str2bool, default=False, help="是否使用預線性層")
        self.parser.add_argument("--norm_type", type=str, default=None, choices=["layer", "batch", None], help="正則化類型")
        self.parser.add_argument("--act_type", type=str, default="relu", help="激活函數類型")

        # ✅ 訓練設定
        self.parser.add_argument("--resumable", type=str2bool, default=True, help="是否可恢復訓練")
        self.parser.add_argument("--max_patience", type=float, default=float("inf"), help="Early-Stop 最大耐心值")
        self.parser.add_argument("--display_each_step", type=int, default=10, help="顯示訓練步驟")
        self.parser.add_argument("--eval_each_step", type=int, default=1, help="評估步驟")

        # ✅ 訓練參數
        self.parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                                 help='evaluation metric')
        self.parser.add_argument("--optimizer", type=str, default="Adam", help="優化器")
        self.parser.add_argument("--lr", type=float, default=0.001, help="學習率")
        self.parser.add_argument("--weight_decay", type=float, default=5e-4, help="權重衰減")

        # ✅ 顯示與工具
        self.parser.add_argument("--display_step", type=int, default=10, help="顯示步驟")

        # ✅ 其他超參數 (未來可擴展)

    def parse(self, api_args=None, return_dict=True):
        """
        解析命令列參數 & API 參數
        - cmd_args: 模擬命令列輸入 (`list`)，通常在 CMD 執行時為 `None`
        - api_args: API 內部調用時提供的 `dict`
        - return_dict: 是否回傳 `Namespace` (預設 `dict`)
        """
        # 解析 CMD 輸入
        #if cmd_args:
        #    args = self.parser.parse_args(cmd_args)
        #else:
        args = self.parser.parse_args()  # 正常解析 CMD 輸入

        # 轉換成 dict
        args_dict = vars(args)

        # ✅ 合併 API 輸入參數 (如果 API 傳入，覆蓋 CMD)
        if api_args:
            args_dict.update(api_args)

        args_dict["device"] = "cuda" if (args_dict["use_gpu"] and torch.cuda.is_available()) else "cpu"

        # ✅ 是否以 dict 形式回傳
        if return_dict:
            return args_dict
        return SimpleNamespace(**args_dict)

# ✅ 創建 `ParserWrapper` 實例
parser_wrapper = ParserWrapper()

if __name__ == "__main__":
    pass
    #print(get_args())