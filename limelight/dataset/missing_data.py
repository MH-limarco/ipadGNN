
__all__ = ["MissingDataHandler"]

import torch

class MissingDataHandler:
    def __init__(self, missing_rate=0.9, missing_type="uniform"):
        self.missing_rate = missing_rate
        self.missing_type = missing_type
        self.missing_mask = None

    def generate_mask(self, n_nodes, n_features):
        """
        產生缺失值掩碼，決定哪些特徵應該被隱藏。
        - `uniform`: 每個特徵以 `missing_rate` 概率缺失
        - `structural`: 每個節點要麼所有特徵都缺失，要麼所有特徵都存在
        """
        if self.missing_type == "structural":
            mask = torch.bernoulli(torch.full((n_nodes, 1), 1 - self.missing_rate)).bool()
            return mask.repeat(1, n_features)  # 讓整個節點的特徵一起缺失
        elif self.missing_type == "uniform":
            return torch.bernoulli(torch.full((n_nodes, n_features), 1 - self.missing_rate)).bool()
        else:
            raise ValueError(f"未知的缺失類型: {self.missing_type}")


    def apply_missing(self, data):
        """
        在 `data.x` 上應用缺失值，返回新數據和缺失掩碼。
        """
        if data.x is None:
            raise ValueError("data.x 不能為 None，請確保節點特徵已經載入。")

        # 生成或讀取 missing mask
        if self.missing_mask is None:
            self.missing_mask = self.generate_mask(data.x.shape[0], data.x.shape[1])

        data.x[~self.missing_mask] = float("nan")  # 將缺失位置設為 NaN
        return data