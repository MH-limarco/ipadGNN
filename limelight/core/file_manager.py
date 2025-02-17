import os
import json
import torch
from datetime import datetime
from limelight.api import parse_config

class ExperimentManager:
    """
    實驗管理器
    - 統一管理實驗存儲格式
    - 根據 `model_type`, `filling_method`, `dataset_name`, `seed` 產生存儲目錄
    - 提供 API 來存儲模型、超參數、結果、日誌
    """

    def __init__(self, model_type, filling_method, dataset_name, seed):
        """
        初始化實驗存儲目錄
        :param model_type: str - 模型類型 (如 GNN, MLP)
        :param filling_method: str - 缺失值填充方法 (如 zero_filling, random_filling)
        :param dataset_name: str - 數據集名稱 (如 Cora, Citeseer)
        :param seed: int - 隨機種子
        """

        # 讀取 `config.yaml`
        config = parse_config()
        base_save_dir = os.path.join(config.get("base_dir", "save"))

        # 建立 `save` 路徑
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_save_dir, model_type, filling_method, dataset_name, f"seed{seed}_{timestamp}")

        # 目錄結構
        self.weight_dir = os.path.join(self.exp_dir, "weight")  # 儲存模型
        self.log_dir = os.path.join(self.exp_dir, "logs")  # 儲存日誌
        self.args_file = os.path.join(self.exp_dir, "args.json")  # 儲存超參數
        self.result_file = os.path.join(self.exp_dir, "result.json")  # 儲存實驗結果

        # 確保目錄存在
        self._ensure_dirs()

    def _ensure_dirs(self):
        """確保所有實驗目錄存在"""
        for directory in [self.exp_dir, self.weight_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)

    ## 🔹 SAVE & LOAD FUNCTIONS ##
    def save_json(self, data, file_path):
        """儲存 JSON 檔案"""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_json(self, file_path):
        """加載 JSON 檔案"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON 檔案 `{file_path}` 不存在")
        with open(file_path, "r") as f:
            return json.load(f)

    def save_args(self, args):
        """儲存超參數 JSON"""
        self.save_json(args, self.args_file)

    def save_result(self, result):
        """儲存實驗結果 JSON"""
        self.save_json(result, self.result_file)

    def save_model(self, model, filename):
        """儲存 PyTorch 模型"""
        filepath = os.path.join(self.weight_dir, filename)
        torch.save(model.state_dict(), filepath)
        return filepath

    def load_model(self, model, filename):
        """加載 PyTorch 模型"""
        filepath = os.path.join(self.weight_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型 `{filepath}` 不存在")
        model.load_state_dict(torch.load(filepath))

    def save_log(self, log_content, filename="experiment.log"):
        """儲存 Log"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            f.write(log_content)
        return filepath

    def get_exp_dir(self):
        """返回實驗目錄"""
        return self.exp_dir