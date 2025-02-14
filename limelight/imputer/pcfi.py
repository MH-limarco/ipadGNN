
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import get_laplacian

from limelight.imputer.base import BaseFilling

NUMBER_ITERATIONS = 40

class PCFIFilling(BaseFilling):
    conv = SimpleConv(aggr='add')
    def fill(self, dataset):
        raise NotImplementedError