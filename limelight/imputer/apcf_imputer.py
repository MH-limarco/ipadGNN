
from torch_geometric.nn import SimpleConv
from torch_scatter import scatter_add
import torch
from tqdm import tqdm
from limelight.imputer.base import BaseFilling

NUMBER_ITERATIONS = 40
MAX_K_HOP = 100
CUDA = "cuda" if torch.cuda.is_available() else "cpu"

class APCFIFilling(BaseFilling):
    conv = SimpleConv(aggr='add')
    def __init__(self, alpha, beta, star=False, batch=None, **kwargs):
        super(APCFIFilling, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.star = star
        self.mask = None
        self.batch = batch

    def fill(self, dataset):
        x, edge_index = dataset.x, dataset.edge_index
        mask = ~x.isnan()
        batch = x.size(-1) if self.batch is None else self.batch

        diffusion_x = torch.zeros_like(x, device=x.device)
        spd_matrix = torch.zeros_like(x, device=x.device)
        for batch_s, batch_e in tqdm(zip(range(0, x.size(-1) + 1, batch),
                                    range(batch, x.size(-1) + batch, batch))):
            batch_e = min(batch_e, x.size(-1))
            x_batch = x[:, batch_s:batch_e]
            mask_batch = mask[:, batch_s:batch_e]

            _diffusion_x, _spd_matrix = self._diffusion(x_batch, edge_index, mask_batch)
            diffusion_x[:, batch_s:batch_e] = _diffusion_x
            spd_matrix[:, batch_s:batch_e] = _spd_matrix.T

        ## node propagation
        output = self.node_propagation(rebuild_x=diffusion_x,
                                       spd_matrix=spd_matrix.T)

        torch.cuda.empty_cache()
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

        k = 1
        _spd_matrix = torch.zeros_like(feature_mask, dtype=torch.float, device=torch.device(CUDA))
        _spd_matrix[feature_mask] = k

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
            if self.star:
                confidence = torch.nn.functional.softmax(confidence, dim=0)
            confidence_list.append(confidence)
        confidence_matrix = torch.stack(confidence_list, dim=1)
        return confidence_matrix

    def _batch_pseudo_confidence(self, edge_index, spd_matrix, batch_size=10000):
        row, col = edge_index[0], edge_index[1]  # (num_edges,)
        _spd_matrix = spd_matrix.to(CUDA)
        confidence_list = []


        for i in range(0, row.size(0), batch_size):
            batch_row = row[i:i + batch_size]
            batch_col = col[i:i + batch_size]

            d_row = _spd_matrix[:, batch_row]  # (feat_dim, batch_size)
            d_col = _spd_matrix[:, batch_col]  # (feat_dim, batch_size)
            confidence = self.alpha ** (d_col - d_row + 1)  # (feat_dim, batch_size)
            confidence = confidence.t()  # (batch_size, feat_dim)

            if self.star:
                confidence = torch.nn.functional.softmax(confidence, dim=1)
            confidence_list.append(confidence)

        confidence_matrix = torch.cat(confidence_list, dim=0)  # (num_edges, feat_dim)


        return confidence_matrix.cpu()


    @staticmethod
    def row_stochastic_transition(confidence_matrix, row_index, dim_size, batch_size=10000):
        #deg_w = scatter_add(confidence_matrix, row_index, dim=0, dim_size=dim_size)
        N, D = confidence_matrix.shape
        row_index = row_index
        deg_w = torch.zeros(dim_size, D, device=CUDA)

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            deg_w += scatter_add(confidence_matrix[i:end].to(CUDA), row_index[i:end].to(CUDA), dim=0, dim_size=dim_size)

        deg_w = deg_w.pow(-1.0)
        deg_w.masked_fill_(deg_w == float("inf"), 0)
        norm_confidence_matrix = confidence_matrix.to(CUDA) * deg_w[row_index]
        return norm_confidence_matrix

    def approximate_channel_diffusion(self, org_x, confidence_matrix, edge_index, mask):
        dest = edge_index[0]  # 源節點索引
        src = edge_index[1]  # 目标節點索引

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
    def _approximate_channel_diffusion(org_x, out_x, confidence_matrix, src_index, dest_index, mask, print_debug=False):
        src_features = out_x[src_index, :].to(CUDA)
        contrib = confidence_matrix * src_features
        new_out = torch.zeros_like(out_x).to(CUDA)
        new_out.index_add_(0, dest_index.to(CUDA), contrib)
        #if print_debug:
        #    print("new_out:\n", new_out, "\n")
        out_x = new_out
        out_x[mask] = org_x[mask].to(out_x.device)
        #if print_debug:
        #    print("contrib:\n", out_x, "\n")

        return out_x

    def node_propagation(self, rebuild_x, spd_matrix):
        conf = torch.corrcoef(rebuild_x.T).nan_to_num().fill_diagonal_(0)
        #print("conf\n", conf, '\n')
        spd_matrix = spd_matrix.to(rebuild_x.device)
        alpha_mean = (self.alpha ** spd_matrix.T) * (rebuild_x - torch.mean(rebuild_x, dim=0))
        #print("alpha_mean\n", alpha_mean, '\n')
        alpha_mean_conf = torch.matmul(alpha_mean, conf)
        #print("alpha_mean_conf\n", alpha_mean_conf, '\n')
        update_x = self.beta * (1 - (self.alpha ** spd_matrix.T)) * alpha_mean_conf
        #print("update_x\n", update_x, '\n')

        out_x = rebuild_x + update_x
        return out_x


if __name__ == "__main__":
    f = APCFIFilling(alpha=0.2, beta=0.5, star=True, batch=None, debug=True)

    _ = [i for i in range(1, 7)]
    _x = [[i * 10 + j for i in range(5)] for j in _]
    _x = torch.tensor(_x, dtype=torch.float)
    mask = [[0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1,],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 1, 1, 0, 1]
            ]
    mask = torch.tensor(mask, dtype=torch.bool)
    x = _x.clone()
    x[~mask] = float('nan')
    edges_index = torch.tensor([
                                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    ], dtype=torch.long)


    from torch_geometric.data import Data
    dataset = Data(x=x, edge_index=edges_index)
    print()
    out = f(dataset)
    print("final output:")
    print(out.x)

    print("original x:")
    print(_x)
