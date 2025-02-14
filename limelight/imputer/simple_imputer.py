
__all__ = ["FixValueFilling", "RandomFilling", "MeanFilling", "NeighborhoodMeanFilling"]

import torch
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import degree
from limelight.imputer.base import BaseFilling

FIX_VALUE = 0

class FixValueFilling(BaseFilling):
    fix_value = FIX_VALUE
    def fill(self, dataset):
        return torch.ones_like(dataset.x) * self.fix_value


class RandomFilling(BaseFilling):
    def fill(self, dataset):
        return torch.randn_like(dataset.x, device=dataset.x.device)


class MeanFilling(BaseFilling):
    def fill(self, dataset):
        return self.compute_mean(dataset.x)

    @staticmethod
    def compute_mean(x):
        return torch.nanmean(x, dim=0).repeat(x.shape[0], 1)


class NeighborhoodMeanFilling(BaseFilling):
    conv = SimpleConv(aggr='add')
    def fill(self, dataset):
        x, edge_index, feature_mask = dataset.x, dataset.edge_index, dataset.feature_mask
        x_masked = x.clone().masked_fill(dataset.x.isnan(), 0.0)

        neighbor_sum = self.conv(x_masked, edge_index)
        deg = degree(edge_index[1],
                     num_nodes=x.shape[0],
                     dtype=torch.float32)

        return torch.where(deg.view(-1, 1) > 0.0,
                           neighbor_sum / deg.view(-1, 1),
                           torch.zeros_like(neighbor_sum))
