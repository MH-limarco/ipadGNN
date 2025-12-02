
__all__ = ['FeaturePropagationFilling']

import warnings
import torch
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import get_laplacian, remove_self_loops

from limelight.imputer.base import BaseFilling
from limelight.api import parse_config

warnings.filterwarnings('ignore',
                        '.*Sparse CSR tensor support is in beta state.*')
CONFIG = parse_config()
NUMBER_ITERATIONS = CONFIG.num_iterations
NUMBER_ITERATIONS = 40
CUDA = "cuda" if torch.cuda.is_available() else "cpu"


def get_normalized_adj(edge_index, n_nodes=None):
    from torch_geometric.utils.num_nodes import maybe_num_nodes
    n_nodes = maybe_num_nodes(edge_index, n_nodes)

    #edge_index, edge_weight = get_laplacian(edge_index, normalization="sym")
    #edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes=n_nodes)
    return edge_index, edge_weight, n_nodes


def get_symmetrically_normalized_adjacency(edge_index, num_nodes):
        """
        Given an edge_index, return the same edge_index and edge weights computed as
        \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
        """
        from torch_scatter import scatter_add
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, DAD


class FeaturePropagationFilling(BaseFilling):
    conv = SimpleConv(aggr='add').to(CUDA)

    def fill(self, dataset):
        _dataset = dataset.to(CUDA)
        x, edge_index = _dataset.x, _dataset.edge_index
        missing_mask = x.isnan()
        known_mask = ~missing_mask

        out = x.clone()
        out[missing_mask] = 0.0

        adj = self.get_normalized_adj(edge_index)
        for _ in range(NUMBER_ITERATIONS):
            out = self.conv(out, adj)
            out[known_mask] = x[known_mask]
        return out

    def __fill(self, dataset):
        from torch_geometric.transforms.feature_propagation import FeaturePropagation
        missing_mask = dataset.x.isnan()
        func = FeaturePropagation(missing_mask, num_iterations=NUMBER_ITERATIONS)
        out = func(dataset).x
        return out


    @staticmethod
    def get_normalized_adj(edge_index, num_nodes=None):
        edge_index, edge_weight, num_nodes = get_normalized_adj(edge_index, num_nodes)

        adj = torch.sparse_coo_tensor(edge_index, values=edge_weight,
                                      size=(num_nodes, num_nodes)).to_sparse_csr().to(edge_index.device)
        return adj