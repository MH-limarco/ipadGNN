
__all__ = ['FeaturePropagationFilling']

from torch_geometric.nn import SimpleConv
from torch_geometric.utils import get_laplacian

from limelight.imputer.base import BaseFilling

NUMBER_ITERATIONS = 40

class FeaturePropagationFilling(BaseFilling):
    conv = SimpleConv(aggr='add')
    def fill(self, dataset):
        x, edge_index = dataset.x, dataset.edge_index
        mask = x.isnan()
        out = x.clone()
        out[mask] = 0

        _, edge_weight = get_laplacian(edge_index,
                                       num_nodes=out.shape[0],
                                       normalization='sym')
        for _ in range(NUMBER_ITERATIONS):
            out = self.conv(out, edge_index, edge_weight=edge_weight)
            out[~mask] = x[~mask]

        return out
