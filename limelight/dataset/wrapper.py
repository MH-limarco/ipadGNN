
__all__ = ["DatasetWrapper"]

import torch
from torch_geometric.data import Data
from limelight.dataset.split_manager import SplitManager
from limelight.dataset.missing_data import MissingDataHandler

class DatasetWrapper:
    """
    封裝 PyG 數據集，提供統一的數據操作方法
    """

    def __init__(self, dataset: Data, missing_mask: torch.Tensor = None):
        """
        參數：
        - dataset: PyG 數據集
        - missing_mask: 缺失數據的 mask (可選)
        """
        self.dataset = dataset
        self.missing_mask = missing_mask  # 默認無缺失數據

    def apply_missing(self, missing_rate, missing_type="uniform"):
        """
        應用缺失值，並返回新的 `DatasetWrapper`
        """
        handler = MissingDataHandler(missing_rate, missing_type)
        dataset_with_missing = handler.apply_missing(self.dataset)

        return DatasetWrapper(dataset_with_missing, handler.missing_mask)

    def split(self, method="random", **kwargs):
        """
        對數據集進行 `train/val/test` 拆分
        """
        split_manager = SplitManager(self.dataset)

        if method == "random":
            dataset = split_manager.random_split(**kwargs)
        elif method == "class":
            dataset = split_manager.class_split(**kwargs)
        elif method == "fixed":
            dataset = split_manager.fixed_split(**kwargs)
        else:
            raise ValueError(f"未知的 split method: {method}")

        return DatasetWrapper(dataset, self.missing_mask)