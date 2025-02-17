
__all__ = ['SplitManager']

import torch
from limelight.utils import set_attrs

TRAIN_MASK = 'train_mask'
VAL_MASK = 'val_mask'
TEST_MASK = 'test_mask'

class SplitManager:
    def __init__(self, dataset):
        self.dataset = dataset
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

    def random_split(self, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
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

    def class_split(self, label_num_per_class, valid_num=500):
        class_list = self.label.unique()

        train_idx, non_train_idx = [], []
        all_idx = torch.arange(self.num_nodes)

        # 針對每個類別進行拆分
        for c in class_list:
            idx_c = all_idx[self.label == c]
            idx_c = idx_c[torch.randperm(idx_c.shape[0])]  # 隨機排列
            train_idx += idx_c[:label_num_per_class].tolist()
            non_train_idx += idx_c[label_num_per_class:].tolist()

        # 轉換為 `torch.Tensor`
        train_idx = torch.as_tensor(train_idx)
        non_train_idx = torch.as_tensor(non_train_idx)

        # 隨機打亂非訓練集
        perm = torch.randperm(non_train_idx.shape[0])
        valid_idx = non_train_idx[perm[:valid_num]]
        test_idx = non_train_idx[perm[valid_num:]]  # 剩下的所有數據作為測試集

        # 創建 `bool` mask
        self._create_bool_mask(train_idx, valid_idx, test_idx)

        return self.dataset

    def fixed_split(self, idx=0):
        self.set_masks()
        if getattr(self, TRAIN_MASK).dim() == 1:
            _masks = [getattr(self, n) for n in [TRAIN_MASK, VAL_MASK, TEST_MASK]]
        else:
            _masks = [getattr(self, n)[:, idx] for n in [TRAIN_MASK, VAL_MASK, TEST_MASK]]
        set_attrs(obj=self.dataset, names=[TRAIN_MASK, VAL_MASK, TEST_MASK], values=_masks)
        return self.dataset


if __name__ == "__main__":
    from limelight.dataset.dataset_loader import load_data

    mag = SplitManager(load_data("hetero", name='roman-empire'))
    print(mag.fixed_split(2).train_mask)
    print(mag.random_split())
    print(mag.class_split(label_num_per_class=10))
