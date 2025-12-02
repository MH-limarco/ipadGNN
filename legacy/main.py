


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SimpleConv, GINConv, ARMAConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch import nn
from torch_geometric.nn.inits import glorot
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gc


pruning_ratio = 0.5

class ChebMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 K: int = 3,
                 normalization= 'sym',
                 bias=True, **kwargs):

        super(ChebMLP_Conv, self).__init__()
        self.K = K
        self.lins = torch.nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)])

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @property
    def in_channels(self):
        return self.lins[0].in_features

    @property
    def out_channels(self):
        return self.lins[0].out_features

    def reset_parameters(self):
        from torch_geometric.nn.inits import zeros
        for lin in self.lins:
            glorot(lin.weight)
        zeros(self.bias)

    def forward(
            self,
            x,
            edge_index=None,
            edge_weight = None,
            batch = None,
            lambda_max = None,):

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = x
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = Tx_1
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


class ARAMMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 num_stacks: int = 1, num_layers: int = 1,
                 shared_weights: bool = False,
                 act = nn.ReLU(), dropout: float = 0.,
                 bias: bool = True, **kwargs):

            super(ARAMMLP_Conv, self).__init__()
            self.num_stacks = num_stacks
            self.num_layers = num_layers
            self.act = act
            self.shared_weights = shared_weights
            self.dropout = dropout

            layers = 1 if shared_weights else num_layers

            # 1) init_weight: 第一層從 in_channels -> out_channels
            self.init_linears = nn.ModuleList([
                nn.Linear(in_channels, out_channels, bias=bias)
                for _ in range(num_stacks)
            ])

            self.stack_linears = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(out_channels, out_channels, bias=bias)
                    for _ in range(num_stacks)
                ]) for _ in range(layers)
            ])

            # 3) root_weight: skip-connection 映射 in_channels -> out_channels
            self.root_linears = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(in_channels, out_channels, bias=bias)
                    for _ in range(num_stacks)
                ]) for _ in range(num_layers)
            ])

            # dropout
            self.dropout_layer = nn.Dropout(p=dropout)

            # 初始化所有線性層
            self.reset_parameters()

    @property
    def in_channels(self):
        return self.init_linears[0].in_features

    @property
    def out_channels(self):
        return self.init_linears[0].out_features

    def reset_parameters(self):
        # 針對所有線性層呼叫內建初始化
        for lin in self.init_linears:
            lin.reset_parameters()
        for layer in self.stack_linears:
            for lin in layer:
                lin.reset_parameters()
        for layer in self.root_linears:
            for lin in layer:
                lin.reset_parameters()



    def forward(self, x, edge_index: None, edge_weight = None):
        # 2) 增加 stack 維度：batch_size x num_stacks x N x F
        x = x.unsqueeze(-3)  # -> [..., 1, N, F_in]
        out = x

        # 3) 逐層計算
        for t in range(self.num_layers):
            for k in range(self.num_stacks):
                # 3.1) 第 0 層用 init_linears
                if t == 0:
                    h = self.init_linears[k](x[..., k, :, :])
                else:
                    # 若共享則都用 layer 0
                    layer_idx = 0 if self.shared_weights else t - 1
                    h = self.stack_linears[layer_idx][k](out[..., k, :, :])

                # 3.3) skip-connection
                root = self.dropout_layer(x[..., k, :, :])
                root = self.root_linears[t if not self.shared_weights else 0][k](root)
                h = h + root

                # 3.5) activation
                if self.act is not None:
                    h = self.act(h)

                if k == 0:
                    tmp = h.unsqueeze(-3)
                else:
                    tmp = torch.cat([tmp, h.unsqueeze(-3)], dim=-3)
            out = tmp

        return out.mean(dim=-3)


class SAGEMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True,
                 pmlp=False):

        super(SAGEMLP_Conv, self).__init__()
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = torch.nn.Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def update(self):
        self.out_channels = self.lin_l.out_features

    def forward(self, x, edge_index=None):
        """"""
        if isinstance(x, torch.Tensor):
            x = (x, x)

        out = x[0]
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

class GCNMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalize: bool = True,
                 bias: bool = True,
                 pmlp=False):

        super().__init__()
        self.normalize = normalize
        self.pmlp = pmlp
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')


        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.nn.inits import zeros
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index=None):
        """"""

        out = x
        out = self.lin(out)

        #if not self.training:
        #    out = self.conv(out, edge_index)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out


class _GCNMLP_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels,
                 normalize: bool = False,
                 bias: bool = True,
                 pmlp=False, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.normalize = normalize

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.nn.inits import zeros
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin(out)
        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j):
        return x_j


class GINMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 eps = 0.0,
                 bias: bool = True,
                 train_eps: bool = True,
                 pmlp=False):

        super(GINMLP_Conv, self).__init__()
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')
        self.pmlp = False
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))


        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels, bias=bias),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(out_channels),
            torch.nn.Linear(out_channels, out_channels, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.nn.inits import reset
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def update(self):
        self.out_channels = self.mlp[-1].out_features

    def forward(self, x, edge_index=None):
        """"""
        if not self.training and self.pmlp:
            out = self.conv(x, edge_index)
        else:
            out = x

        out = out + (1 + self.eps) * x
        out = self.nn(out)
        out = F.dropout(out, p=0.5, training=self.training)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_type="SAGE", cls_mlp=True, pmlp=False, norm=False, res=False, num_layers=2, dropout=0.0):
        super(MLP, self).__init__()
        self.pured = False
        self.norm = norm
        self.res = res

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        self.dropout = dropout
        self.gnn_type = gnn_type


        gnn_type = gnn_type.lower()
        if gnn_type == "sage":
            gnnmlp_conv = SAGEMLP_Conv

        elif gnn_type == "gcn":
            gnnmlp_conv =  GCNMLP_Conv

        elif gnn_type == "gin":
            gnnmlp_conv = GINMLP_Conv

        elif gnn_type == "aram":
            gnnmlp_conv = ARAMMLP_Conv

        elif gnn_type == "cheb":
            gnnmlp_conv = ChebMLP_Conv


        # Input layer
        self.layers.append(gnnmlp_conv(input_dim, hidden_dim, pmlp=pmlp))
        self.res_layers.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - (1 if cls_mlp else 2)):
            self.layers.append(gnnmlp_conv(hidden_dim, hidden_dim, pmlp=pmlp))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.res_layers.append(nn.Linear(hidden_dim, hidden_dim))

        if not cls_mlp:
            self.cls_layers = gnnmlp_conv(hidden_dim, hidden_dim)
        else:
            self.cls_layers = torch.nn.Linear(hidden_dim, output_dim)

    def reset_norm_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

        for layer in self.res_layers:
            layer.reset_parameters()

        self.cls_layers.reset_parameters()
        self.reset_norm_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for idx, layer in enumerate(self.layers):
            #x = layer(x) if isinstance(layer, (Linear, (SAGEMLP_Conv, GCNMLP_Conv))) else layer(x, edge_index)
            if self.res:
                x = layer(x, edge_index) + self.res_layers[idx](x)
            else:
                x = layer(x, edge_index)

            if self.norm: #and not isinstance(layer, (Linear, SAGEMLP_Conv, GCNMLP_Conv, GINMLP_Conv)):
                x = self.norms[idx](x) #not self.pured and

            x = nn.functional.relu(x)
            #if isinstance(layer, (SAGEMLP_Conv, GCNMLP_Conv)):
            if not isinstance(layer, (Linear, SAGEMLP_Conv, GCNMLP_Conv, GINMLP_Conv)):
                x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.cls_layers(x)
        return out

def mlp2gnn(mlp, gat=False):
    model = deepcopy(mlp)
    norm = model.norm
    model.use_edge = True
    norm_layer = nn.LayerNorm #if not mlp.pured else nn.BatchNorm1d

    for i in range(len(model.layers)):
        if isinstance(model.layers[i], GCNMLP_Conv):
            if norm:
                model.norms[i] = norm_layer(model.layers[i].lin.out_features)
            if gat:
                model.layers[i] = GATConv(model.layers[i].lin.in_features, model.layers[i].lin.out_features, heads=1)
            else:
                model.layers[i] = GCNConv(model.layers[i].lin.in_features, model.layers[i].lin.out_features)
        elif isinstance(model.layers[i], SAGEMLP_Conv):
            if norm:
                model.norms[i] = norm_layer(model.layers[i].lin_l.out_features)
            model.layers[i] = SAGEConv(model.layers[i].lin_l.in_features, model.layers[i].lin_l.out_features)
        elif isinstance(model.layers[i], GINMLP_Conv):
            if norm:
                model.norms[i] = norm_layer(model.layers[i].nn[-1].out_features)
            _nn = model.layers[i].nn
            for layer in _nn:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            model.layers[i] = GINConv(nn=_nn, eps=model.layers[i].initial_eps)

        elif isinstance(model.layers[i], ARAMMLP_Conv):
            if norm:
                model.norms[i] = norm_layer(model.layers[i].out_channels)
            model.layers[i] = ARMAConv(model.layers[i].in_channels, model.layers[i].out_channels,
                                      num_stacks=model.layers[i].num_stacks, num_layers=model.layers[i].num_layers,
                                      act=model.layers[i].act, dropout=model.layers[i].dropout,)

        elif isinstance(model.layers[i], ChebMLP_Conv):
            if norm:
                model.norms[i] = norm_layer(model.layers[i].out_channels)
            model.layers[i] = ARMAConv(model.layers[i].in_channels, model.layers[i].out_channels)


    return model

def eval(model, data, device):
    with torch.no_grad():
        model.eval()
        with torch.autocast(device_type=device):
            if isinstance(model, MLP):
                out = model(data)
            else:
                out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def train(model, data, epochs, device, lr=0.001, reture_best=True, early=None):
    from time import perf_counter

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) #weight_decay=5e-4
    scaler = torch.amp.GradScaler()
    best_acc = 0.0
    best_epoch = -1
    _early = 0.0
    with torch.no_grad():
        model.eval()
        if isinstance(model, MLP):
            out = model(data)
        else:
            out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        init_acc = acc

    qbar = tqdm(range(epochs))
    torch.cuda.reset_peak_memory_stats(device)
    start_time = perf_counter()
    for epoch in qbar:
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=device):
            if isinstance(model, MLP):
                out = model(data)
            else:
                out = model(data.x, data.edge_index)
            loss = nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        qbar.set_description(f"Epoch {epoch} Loss {loss.item():.3f}")

        acc = eval(model, data, device)
        qbar.set_postfix(loss=loss.detach().cpu().item(), test_acc=acc)

        if best_acc < acc:
                best_epoch = epoch
                best_acc = acc
                best_state_dict = model.state_dict()
                _early = 0
        else:
                _early += 1

        if early is not None and _early >= early:
                break

    #torch.cuda.synchronize()
    end_time = perf_counter()
    max_vram = torch.cuda.max_memory_reserved(device) / 1024 ** 2
    print(f"Max VRAM: {max_vram:.2f} MB")

    print(f"Best Test Accuracy: {best_acc:.4f} on epoch - {best_epoch}")
    print(f"init acc:{init_acc:.2f}")
    if reture_best:
        _state_dict = best_state_dict
    else:
        _state_dict = model.state_dict()
    return _state_dict, best_acc, end_time - start_time, best_epoch

def prune(model, data, imp="group", isomorphic=True):
    num_class = data.num_classes
    import torch_pruning as tp
    model.pured = True
    model = model.cuda()
    data = data.cuda()

    for m in model.modules():
        m.requires_grad_(True)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if m.out_features == num_class:
                ignored_layers.append(m)


    unwrapped_parameters = []
    #for name, param in model.named_parameters():
    #    if "eps" in name or("bias" in name and "norms" not in name and "res_layers" not in name and "lin" not in name):
    #        unwrapped_parameters.append([param, 0])
    for name, param in model.named_parameters():
        _name = name.split('.')
        if len(_name) > 2 and ((_name[0] not in ("norms", "res_layers") and ("bias" in _name[2] or "weight" in _name[2])) or
                               (_name[0] in ("layers") and _name[2] == "eps")):
            unwrapped_parameters.append([param, 0])

    model.eval()
    if imp.lower() == "group":
        imp = tp.importance.GroupMagnitudeImportance(p=2) #GroupMagnitudeImportance
    else:
        imp = tp.importance.LAMPImportance()

    pruner_func = tp.pruner.GroupNormPruner #BasePruner BNScalePruner GroupNormPruner
    pruner = pruner_func(model,
                                  example_inputs=data,
                                  importance=imp,
                                  isomorphic=isomorphic,
                                  global_pruning=True,
                                  pruning_ratio=pruning_ratio,
                                  ignored_layers=ignored_layers,
                                  unwrapped_parameters=unwrapped_parameters,
                                  )

    #base_macs, base_nparams = tp.utils.count_ops_and_params(model, data)
    #print(f"Full:\nmacs:{base_macs:.2f} | params:{base_nparams}")  # print the number of MACs and parameters before pruning
    pruner.step()
    #macs, nparams = tp.utils.count_ops_and_params(model, data)
    #print(f"Prune:\nmacs:{macs:.2f} | params:{nparams}")  # print the number of MACs and parameters after pruning
    return model

def trim_arr(arr, k=1):
    arr = np.array(arr)
    if k <= 0:
        return arr
    else:
        arr = np.sort(arr, axis=None)
        return arr[k:-k]


def save_log(run_test_log, exp_args, data_name, fill_methood, missing_rate, classic, log_name="log", trim=1):
    model_name = exp_args.methood
    _model = exp_args.model
    _mr = exp_args.missing_rate

    run_test_acc, run_train_time = run_test_log
    run_test_acc = trim_arr(run_test_acc, k=0)
    avg_test_acc = np.mean(run_test_acc)
    std_test_acc = np.std(run_test_acc)

    run_test_acc_trim = trim_arr(run_test_acc, k=trim)
    avg_test_acc_trim = np.mean(run_test_acc_trim)
    std_test_acc_trim = np.std(run_test_acc_trim)

    run_train_time = trim_arr(run_train_time, k=0)
    avg_test_time = np.mean(run_train_time)
    std_test_time = np.std(run_train_time)

    output_test_acc = f"{avg_test_acc:.4f} ± {std_test_acc:.3f}"
    output_test_acc_trim = f"{avg_test_acc_trim:.4f} ± {std_test_acc_trim:.3f}"
    output_test_time = f"{avg_test_time:.4f} ± {std_test_time:.3f}"

    print(f"{_mr}_{f'_{fill_methood}' if fill_methood is not None else ''} {_model} {model_name.lower()} - {data_name.lower()}: Test Acc: {output_test_acc}")

    log_file = open(f"{data_name.lower()}_{log_name}.txt", encoding="utf-8", mode="a+")
    with log_file as txt:
        print('model', "test_acc", "test_acc_trim", "train_time", file=txt)

        print(f"{'classic_' if classic else ''}{model_name}{f'_{fill_methood}' if fill_methood is not None else ''}_{missing_rate}",
              output_test_acc, output_test_acc_trim, output_test_time, file=txt, sep=',')


def exp(methood, model, imp="Group", isomorphic=True, mlp_epoch=1000, lr=0.0005, mlp_lr=0.001, init=False):
    methood = methood.lower()
    from time import perf_counter
    model.reset_parameters()
    start = perf_counter()

    if methood == "gnn":
        if isinstance(model, MLP):
            model = mlp2gnn(model).to(device)

        gnn = model.to(device)
        gnn.reset_parameters()
        _, acc, _, _ = train(gnn, dataset, 1500, device, lr=lr)

    elif methood == "prune_gnn":
        #best_epoch = 0
        lr = 0.001

        state_dict_mlp, acc, _, best_epoch = train(model, dataset, mlp_epoch, device,
                                                   lr=mlp_lr, early=round(mlp_epoch * 0.5))

        i = 0
        while (best_epoch + 1) / mlp_epoch <= 0.05 and init and i < 2:
            model.reset_parameters()
            state_dict_mlp, acc, _, best_epoch = train(model, dataset, mlp_epoch, device,
                                                 lr=mlp_lr, early=round(mlp_epoch*0.5))


            i += 1
        model.load_state_dict(state_dict_mlp)
        prune_model = prune(model, dataset, imp=imp, isomorphic=isomorphic)
        #prune_model.reset_norm_parameters()
        #state_dict_prune_mlp = prune_model.state_dict()
        del model
        print(prune_model)
        A

        prune_gnn = mlp2gnn(prune_model).to(device)
        #prune_gnn.load_state_dict(state_dict_prune_mlp)
        #print(prune_gnn)
        torch.cuda.empty_cache()
        gc.collect()

        _, acc, _, _ = train(prune_gnn, dataset, 1500, device, lr=lr)
        del prune_gnn

    elif methood == "prune_gnn_mlpinit":
        state_dict_mlp, _, _, _ = train(model, dataset, mlp_epoch, device, lr=0.01)
        model.load_state_dict(state_dict_mlp)
        prune_model = prune(model, dataset, imp=imp, isomorphic=isomorphic)
        del model
        state_dict_prune_mlp, acc, _, _ = train(prune_model, dataset, 100, device, lr=0.001)
        prune_model.load_state_dict(state_dict_prune_mlp)
        prune_gnn_mlpinit = mlp2gnn(prune_model).to(device)
        del prune_model
        prune_gnn_mlpinit.load_state_dict(state_dict_prune_mlp)
        _, acc, _, _ = train(prune_gnn_mlpinit, dataset, 2000, device, lr=lr)

    else:
        raise ValueError(f"Unknown method: {methood}")

    return acc, perf_counter() - start


if __name__ == "__main__":
    from limelight.api.dataset import load_dataset
    from limelight.api.imputer import filling_data
    from limelight.utils import fix_seed
    import argparse

    parser = argparse.ArgumentParser(description="MLP to GNN")
    parser.add_argument('--methood', type=str,
                        choices=["gnn", "prune_gnn", "prune_gnn_mlpinit"],
                        default='prune_gnn')

    parser.add_argument('--runs', type=int,
                        default=10)

    parser.add_argument('--lr_rate', type=float,
                        default=1)

    parser.add_argument('--missing_rate', type=float,
                        default=0.0)

    parser.add_argument('--model', type=str,
                        choices=["GCN", "SAGE", "GIN", "ARAM", "Cheb"],
                        default="ARAM")

    parser.add_argument("--fill", type=str, default="fp",
                        choices=["fp", "apcfi", "zero"],
                        help="filling method")


    parser.add_argument('--pmlp', type=bool,
                        default=False)
    parser.add_argument('--norm', type=bool,
                        default=False)

    parser.add_argument('--classic', action='store_true')


    exp_args = parser.parse_args()

    classic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model: {exp_args.model}, Method: {exp_args.methood}, Missing Rate: {exp_args.missing_rate}, Filling: {exp_args.fill}, Classic: {exp_args.classic}")

    apcfi_args = {"cora": [[0.9, 10**(-1)], [0.7, 10**(-2)]],
                  "citeseer": [[0.5, 10**(0)], [0.8, 10**(-5)]],
                  "pubmed": [[0.8, 10**(0)], [0.7, 10**(-2.5)]],
                  "photo": [[0.2, 10**(-6)], [0.2, 10**(-1.5)]],
                  "computers": [[0.2, 10**(-4)], [0.1, 10**(-2.5)]],
                  "cs": [[0.8, 10**(-6)], [0.2, 10**(-4)]],
    }

    gcn_args = {
                 "cora" : [512, 3, 0.7, "class", "lamp", True, 300, 0.0005, 0.0001, False, False, False, False], #group
                 #"citeseer": [512, 2, 0.5, "class", "group", True, 200, 0.001, 0.0005, False, False, False, False],
                 #"pubmed": [256, 2, 0.7, "class", "lamp", True, 1, 0.001, 0.001, False, False, False, False],
                 #"photo": [256, 6, 0.5, "fixed", "group", True, 500, 0.00001, 0.00025, False, True, True, False], #0.0001
                 #"computers": [256, 3, 0.5, "fixed", "lamp", True, 100, 0.001, 0.001, False, True, False, True], #sage: group, #gcn: 0.01
                 #"cs": [512, 2, 0.3, "fixed", "group", True, 500, 0.0001, 0.001, False, True, True, True],
    }

    sage_args = {
        #"cora": [256, 3, 0.7, "class", "lamp", True, 300, 0.0005, 0.0005, False, False, False, False],  # group
        "cora": [256, 3, 0.7, "class", "lamp", True, 300, 0.001, 0.001, False, False, False, False],  # group
        "citeseer": [512, 3, 0.2, "class", "group", True, 200, 0.001, 0.001, False, False, False, False],
        "pubmed": [512, 4, 0.7, "class", "group", True, 200, 0.005, 0.001, False, False, False, False],
        "photo": [64, 6, 0.2, "fixed", "group", True, 500, 0.001, 0.00025, False, True, True, False],  # 0.0001
        "computers": [64, 4, 0.3, "fixed", "group", True, 500, 0.0001, 0.00075, False, True, False, True],
        "cs": [512, 2, 0.5, "fixed", "group", True, 500, 0.0001, 0.001, False, True, True, False],
    }

    data_args = sage_args if exp_args.model == "SAGE" else gcn_args

    runs = exp_args.runs
    missing_rate = exp_args.missing_rate
    exp_args.fill = exp_args.fill if missing_rate > 0 else None

    if classic:
        exp_args.classic = classic

    for data_name, args in data_args.items():
        fix_seed(123)
        if exp_args.classic:
            args[1] = min(args[1], 2)
            args[2] = 0.0
            args[10] = False
            args[11] = False

        _apcfi = apcfi_args[data_name]
        if missing_rate < 0.995:
            apcfi = _apcfi[0]
        else:
            apcfi = _apcfi[1]


        dataset, mask = load_dataset(data_name, args[3], label_num_per_class=20, missing_rate=missing_rate)
        print(dataset.train_mask.sum().item(), dataset.val_mask.sum().item(), dataset.test_mask.sum().item())
        if missing_rate > 0:
            dataset = filling_data(dataset, exp_args.fill, alpha=apcfi[0], beta=apcfi[1])

        acc_dict = {exp_args.methood: [[], []],}
        if not(exp_args.model == "GCN" and exp_args.methood == "prune_gnn_mlpinit"):
            for run in range(runs):
                _seed = 42 * (run + 1)
                fix_seed(_seed)

                mlp = MLP(input_dim=dataset.x.size(1),
                          hidden_dim=args[0],
                          output_dim=dataset.num_classes,
                          gnn_type=exp_args.model,
                          cls_mlp=True, pmlp=args[9], norm=args[10], res=args[11],
                          num_layers=args[1], dropout=args[2]).to(device)
                #if run == 0:
                #    print(mlp)

                print(f"\n\n{data_name} run {run + 1} / {runs}")

                test_acc, train_time = exp(exp_args.methood, mlp,
                                           imp=args[4], isomorphic=args[5],
                                           mlp_epoch=args[6], lr=args[7] * exp_args.lr_rate, mlp_lr=args[8], init=args[12])
                acc_dict[exp_args.methood][0].append(test_acc)
                acc_dict[exp_args.methood][1].append(train_time)

            torch.cuda.empty_cache()
            gc.collect()

            save_log(acc_dict[exp_args.methood], exp_args, data_name,
                     exp_args.fill, exp_args.missing_rate, exp_args.classic, exp_args.model.lower())


