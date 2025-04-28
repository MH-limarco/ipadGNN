
from torch_geometric.nn import SimpleConv
from torch_scatter import scatter_add
import torch

from limelight.imputer.base import BaseFilling

NUMBER_ITERATIONS = 40
MAX_K_HOP = 100
CUDA = "cuda" if torch.cuda.is_available() else "cpu"

class APCFIFilling(BaseFilling):
    conv = SimpleConv(aggr='add')
    def __init__(self, alpha, beta, **kwargs):
        super(APCFIFilling, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mask = None

    def fill(self, dataset):
        x, edge_index = dataset.x, dataset.edge_index
        mask = ~x.isnan()

        diffusion_x, spd_matrix = self._diffusion(x, edge_index, mask)
        ## node propagation
        output = self.node_propagation(rebuild_x=diffusion_x,
                                       spd_matrix=spd_matrix)
        return output

    def _diffusion(self, x: torch.Tensor, edge_index, mask: torch.Tensor):
        spd_matrix = self.approximate_k_hop(edge_index, mask)

        ## pseudo confidence matrix
        confidence_matrix = self.pseudo_confidence(edge_index=edge_index,
                                                   spd_matrix=spd_matrix)

        ## row stochastic transition
        normalize_confidence_matrix = self.row_stochastic_transition(confidence_matrix=confidence_matrix,
                                                                     row_index=edge_index[0],
                                                                     dim_size=spd_matrix.shape[1])

        ## channel diffusion
        diffusion_x = self.approximate_channel_diffusion(org_x=x,
                                                         confidence_matrix=normalize_confidence_matrix,
                                                         edge_index=edge_index, mask=mask)
        return diffusion_x, spd_matrix

    def approximate_k_hop(self, edge_index, feature_mask):
        _org_device = edge_index.device

        edge_index = edge_index.to(CUDA)
        feature_mask = feature_mask.to(CUDA)
        _spd_matrix = torch.zeros_like(feature_mask, dtype=torch.float, device=torch.device(CUDA))

        k = 1
        _spd_matrix[feature_mask] = 1
        clone_spd = _spd_matrix.clone()
        for k in range(k, MAX_K_HOP):
            prev_clone_spd = clone_spd.clone()  # 保存上一步的 spd矩陣
            clone_spd = self.conv(clone_spd, edge_index)  # 進行傳播
            # **找到剛剛變成非零的節點 (new_index)**
            new_nonzero = (prev_clone_spd == 0) & (clone_spd > 0)  # 上一次傳播為 0，本次傳播首次變為非零值

            _spd_matrix[new_nonzero] = k  # 更新 hop 數
            if not torch.any(_spd_matrix == 0):  # 如果所有節點都已經更新，則停止
                break

        spd_matrix = _spd_matrix.T
        spd_matrix = spd_matrix.to(_org_device)
        return spd_matrix

    def pseudo_confidence(self, edge_index, spd_matrix):
        row, col = edge_index[0], edge_index[1]
        feat_dim = spd_matrix.shape[0]
        confidence_list = []
        for i in range(feat_dim):
            d_row = spd_matrix[i][row]
            d_col = spd_matrix[i][col]
            confidence = self.alpha ** (d_col - d_row + 1)
            confidence_list.append(confidence)

        confidence_matrix = torch.stack(confidence_list, dim=1)
        return confidence_matrix

    @staticmethod
    def row_stochastic_transition(confidence_matrix, row_index, dim_size):
        deg_w = scatter_add(confidence_matrix, row_index, dim=0, dim_size=dim_size)
        deg_w_inv = deg_w.pow(-1.0)
        deg_w_inv.masked_fill_(deg_w_inv == float("inf"), 0)
        norm_confidence_matrix = confidence_matrix * deg_w_inv[row_index]
        return norm_confidence_matrix

    def approximate_channel_diffusion(self, org_x, confidence_matrix, edge_index, mask):
        edge_index = edge_index.to(CUDA)
        org_x = org_x.to(CUDA)
        confidence_matrix = confidence_matrix.to(CUDA)

        dest = edge_index[0]  # 目标節點索引
        src = edge_index[1]  # 源節點索引

        out_x = torch.zeros_like(org_x, device=org_x.device)
        if mask is not None:
            out_x[mask] = org_x[mask]

        for _ in range(NUMBER_ITERATIONS):
            out_x = self._approximate_channel_diffusion(org_x, out_x,
                                                        confidence_matrix,
                                                        src, dest,
                                                        mask)
        return out_x

    @staticmethod
    def _approximate_channel_diffusion(org_x, out_x, confidence_matrix, src_index, dest_index, mask):
        src_features = out_x[src_index, :]
        contrib = confidence_matrix * src_features
        new_out = torch.zeros_like(out_x)
        new_out.index_add_(0, dest_index, contrib)
        out_x = new_out
        out_x[mask] = org_x[mask]
        return out_x

    def node_propagation(self, rebuild_x, spd_matrix):
        conf = torch.corrcoef(rebuild_x.T).nan_to_num().fill_diagonal_(0)
        spd_matrix = spd_matrix.to(rebuild_x.device)
        alpha_mean = (self.alpha ** spd_matrix.T) * (rebuild_x - torch.mean(rebuild_x, dim=0))
        alpha_mean_conf = torch.matmul(alpha_mean, conf)
        update_x = self.beta * (1 - (self.alpha ** spd_matrix.T)) * alpha_mean_conf
        out_x = rebuild_x + update_x
        return out_x
