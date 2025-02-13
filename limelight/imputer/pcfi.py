
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import get_laplacian

from limelight.imputer.base import BaseFilling


class FeaturePropagationFilling(BaseFilling):
    num_iter = 40
    conv = SimpleConv(aggr='add')
    def fill(self, dataset):
        raise NotImplementedError