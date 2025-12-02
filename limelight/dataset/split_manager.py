
__all__ = ['SplitManager']

import torch
import numpy as np
from limelight.utils import set_attrs

LABEL = 'y'
TRAIN_MASK = 'train_mask'
VAL_MASK = 'val_mask'
TEST_MASK = 'test_mask'

TRAIN_MASK_npz = 'train'
VAL_MASK_npz = 'valid'
TEST_MASK_npz = 'test'


class SplitManager:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = dataset.data_dir
        self.data_root = dataset.data_root

        self.name = dataset.name
        self.num_nodes = dataset.num_nodes
        self.label = dataset.y.squeeze()  # 確保標籤形狀正確
        self.reset_masks()

    def set_masks(self):
        if hasattr(self.dataset, TRAIN_MASK) and hasattr(self.dataset, VAL_MASK) and hasattr(self.dataset, TEST_MASK):
            if getattr(self, TRAIN_MASK) is None:
                _masks = [getattr(self.dataset, TRAIN_MASK),
                          getattr(self.dataset, VAL_MASK),
                          getattr(self.dataset, TEST_MASK)]
                set_attrs(obj=self, names=[TRAIN_MASK, VAL_MASK, TEST_MASK], values=_masks)
        else:
            raise AttributeError("No masks found in dataset.")

    def reset_masks(self):
        set_attrs(obj=self, names=[TRAIN_MASK, VAL_MASK, TEST_MASK], values=None)

    def _create_bool_mask(self, train_idx, valid_idx, test_idx):
        """
        根據索引生成 `bool` mask，確保所有 `mask` 都是 `(num_nodes,)` 形狀
        """
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        valid_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

        set_attrs(obj=self.dataset,
                  names=[TRAIN_MASK, VAL_MASK, TEST_MASK],
                  values=[train_mask, valid_mask, test_mask])

    def random_split(self, train_prop=0.5, valid_prop=0.25, ignore_negative=True, **kwargs):
        labeled_nodes = torch.nonzero(self.label != -1, as_tuple=True)[0] if ignore_negative else self.label
        num_labeled = labeled_nodes.shape[0]
        perm = torch.randperm(num_labeled)

        train_num = int(num_labeled * train_prop)
        valid_num = int(num_labeled * valid_prop)

        train_idx = perm[:train_num]
        valid_idx = perm[train_num: train_num + valid_num]
        test_idx = perm[train_num + valid_num:]

        self._create_bool_mask(train_idx, valid_idx, test_idx)
        return self.dataset

    def class_split(self, label_num_per_class, valid_num=500, test_num=1000, **kwargs):
        class_list = self.label.unique()

        train_idx, non_train_idx = [], []
        all_idx = torch.arange(self.num_nodes)

        # 針對每個類別進行拆分
        for c in class_list:
            idx_c = all_idx[self.label == c]
            rand_idx = idx_c[torch.randperm(idx_c.shape[0])]  # 隨機排列
            train_idx += rand_idx[:label_num_per_class].tolist()
            non_train_idx += rand_idx[label_num_per_class:].tolist()

        # 轉換為 `torch.Tensor`
        train_idx = torch.as_tensor(train_idx)
        non_train_idx = torch.as_tensor(non_train_idx)

        # 隨機打亂非訓練集
        perm = torch.randperm(non_train_idx.shape[0])
        non_train_idx = non_train_idx[perm]

        valid_idx = non_train_idx[:valid_num]
        test_idx = non_train_idx[valid_num : valid_num + test_num] # 剩下的所有數據作為測試集

        # 創建 `bool` mask
        self._create_bool_mask(train_idx, valid_idx, test_idx)
        return self.dataset

    def fixed_split(self, mask_idx=0, **kwargs):
        if self.name in ["cs", "physics", "computers", "photo"]:
            _masks_np = np.load(f'{self.data_root}/{self.name}_split.npz')
            _masks = [torch.from_numpy(_masks_np[n]) for n in [TRAIN_MASK_npz, VAL_MASK_npz, TEST_MASK_npz]]
            _max_idx = max([i.max() for i in _masks])
            _masks = [torch.zeros(_max_idx + 1, dtype=torch.bool).scatter_(0, mask, True) for mask in _masks]

        elif self.name in ["wikics"]:
            if not hasattr(self.dataset, "_"+TRAIN_MASK):
                for name in [TRAIN_MASK, VAL_MASK, TEST_MASK]:
                    setattr(self.dataset, f"_{name}", getattr(self.dataset, name))

            _masks = [getattr(self.dataset, f"_{name}")[:, mask_idx] if getattr(self.dataset, f"_{name}").dim() > 1
                      else getattr(self.dataset, f"_{name}") for name in [TRAIN_MASK, VAL_MASK, TEST_MASK]]

        else:
            raise ValueError(f"Unsupported dataset: {self.name} for fixed split.")

        set_attrs(obj=self.dataset, names=[TRAIN_MASK, VAL_MASK, TEST_MASK], values=_masks)

        return self.dataset


if __name__ == "__main__":
    from limelight.dataset.dataset_loader import data_loader

    mag = SplitManager(data_loader("wikci", name='photo')) #roman-empire
    print(mag.fixed_split(2))
    print(mag.random_split())
    print(mag.class_split(label_num_per_class=10))
