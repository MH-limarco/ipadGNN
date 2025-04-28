


import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SimpleConv, GINConv
from torch_geometric.nn.dense.linear import Linear
from torch import nn
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gc


pruning_ratio = 0.5


class SAGEMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True,
                 pmlp=False):

        super(SAGEMLP_Conv, self).__init__()
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')
        self.normalize = normalize
        self.root_weight = root_weight
        self.pmlp = pmlp

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = torch.nn.Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels[1], out_channels, bias=False)

        self.in_channels = self.lin_l.in_features
        self.out_channels = self.lin_l.out_features

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def update(self):
        self.out_channels = self.lin_l.out_features

    def forward(self, x, edge_index=None):
        """"""
        out = x
        out = self.lin_l(out)

        x_r = x
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if not self.training and self.pmlp:
            out = self.conv(out, edge_index)

        return out


class GCNMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalize: bool = True,
                 bias: bool = True,
                 pmlp=False):

        super(GCNMLP_Conv, self).__init__()
        self.normalize = normalize
        self.pmlp = pmlp
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = torch.nn.Linear(in_channels[0], out_channels, bias=bias)

        self.in_channels = self.lin_l.in_features
        self.out_channels = self.lin_l.out_features

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()

    def update(self):
        self.out_channels = self.lin_l.out_features

    def forward(self, x, edge_index=None):
        """"""
        out = x
        out = self.lin_l(out)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if not self.training and self.pmlp:
            out = self.conv(out, edge_index)

        return out

class GINMLP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 eps = 0.0,
                 bias: bool = True,
                 pmlp=True):

        super(GINMLP_Conv, self).__init__()
        self.pmlp = True
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')
        self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_channels[0], out_channels, bias=bias),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(out_channels),
            torch.nn.Linear(out_channels, out_channels, bias=bias),
        )

        self.in_channels = self.nn[0].in_features
        self.out_channels = self.nn[-1].out_features

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.nn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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
    model.use_edge = True
    norm_layer = nn.LayerNorm #if not mlp.pured else nn.BatchNorm1d

    for i in range(len(model.layers)):
        if isinstance(model.layers[i], GCNMLP_Conv):
            model.norms[i] = norm_layer(model.layers[i].lin_l.out_features)
            if not model.pured:
                model.norms[i] = norm_layer(model.layers[i].lin_l.out_features)
            if gat:
                model.layers[i] = GATConv(model.layers[i].lin_l.in_features, model.layers[i].lin_l.out_features, heads=1)
            else:
                model.layers[i] = GCNConv(model.layers[i].lin_l.in_features, model.layers[i].lin_l.out_features)
        elif isinstance(model.layers[i], SAGEMLP_Conv):
            model.norms[i] = norm_layer(model.layers[i].lin_l.out_features)
            model.layers[i] = SAGEConv(model.layers[i].lin_l.in_features, model.layers[i].lin_l.out_features)
        elif isinstance(model.layers[i], GINMLP_Conv):
            model.norms[i] = norm_layer(model.layers[i].nn[-1].out_features)
            _nn = model.layers[i].nn
            for layer in _nn:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            model.layers[i] = GINConv(nn=_nn, eps=model.layers[i].eps)
    return model

def train(model, data, epochs, device, lr=0.001, reture_best=True, early=None):
    from time import perf_counter

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) #weight_decay=5e-4
    best_acc = 0.0
    best_epoch = -1
    _early = 0.0
    with torch.no_grad():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        init_acc = acc


    qbar = tqdm(range(epochs))
    start_time = perf_counter()
    for epoch in qbar:
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=device):
            out = model(data)
            loss = nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        qbar.set_description(f"Epoch {epoch} Loss {loss.item():.3f}")

        with torch.no_grad():
            model.eval()
            with torch.autocast(device_type=device):
                out = model(data)
            pred = out.argmax(dim=1)
            acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            qbar.set_postfix(test_acc=acc)

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
    if best_epoch > 0:
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

    model.eval()
    if imp.lower() == "group":
        imp = tp.importance.GroupMagnitudeImportance(p=2) #GroupMagnitudeImportance
    else:
        imp = tp.importance.LAMPImportance()

    pruner = tp.pruner.BasePruner(model,
                                  example_inputs=data,
                                  importance=imp,
                                  isomorphic=isomorphic,
                                  global_pruning=True,
                                  pruning_ratio=pruning_ratio,
                                  ignored_layers=ignored_layers,
                                  )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, data)
    #print(f"Full:\nmacs:{base_macs:.2f} | params:{base_nparams}")  # print the number of MACs and parameters before pruning
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, data)
    #print(f"Prune:\nmacs:{macs:.2f} | params:{nparams}")  # print the number of MACs and parameters after pruning
    return model

def trim_arr(arr, k=1):
    arr = np.array(arr)
    if k <= 0:
        return arr
    else:
        arr = np.sort(arr, axis=None)
        return arr[k:-k]


def save_log(run_test_log, model_name, data_name, fill_methood, missing_rate, classic, log_name="log", trim=0):

    run_test_acc, run_train_time = run_test_log
    run_test_acc = trim_arr(run_test_acc, k=trim)
    avg_test_acc = np.mean(run_test_acc)
    std_test_acc = np.std(run_test_acc)

    run_train_time = trim_arr(run_train_time, k=trim)
    avg_test_time = np.mean(run_train_time)
    std_test_time = np.std(run_train_time)

    output_test_acc = f"{avg_test_acc:.4f} ± {std_test_acc:.3f}"
    output_test_time = f"{avg_test_time:.4f} ± {std_test_time:.3f}"

    print(f"{model_name.lower()} - {data_name.lower()}: Test Acc: {output_test_acc}")

    log_file = open(f"{data_name.lower()}_{log_name}.txt", encoding="utf-8", mode="a+")
    with log_file as txt:
        print('model', "train_time",
              "test_acc", file=txt)

        print(f"{'classic_' if classic else ''}{model_name}{f'_{fill_methood}' if fill_methood is not None else ''}_{missing_rate}",
              output_test_time, output_test_acc, file=txt, sep=',')


def exp(methood, model, imp="Group", isomorphic=True, mlp_epoch=1000, lr=0.0005, mlp_lr=0.001, init=False):
    methood = methood.lower()
    from time import perf_counter
    model.reset_parameters()
    start = perf_counter()

    if methood == "gnn":
        gnn = mlp2gnn(model).to(device)
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
        #state_dict_prune_mlp = prune_model.state_dict()
        del model
        prune_gnn = mlp2gnn(prune_model).to(device)
        #prune_gnn.load_state_dict(state_dict_prune_mlp)
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
                        default='gnn')

    parser.add_argument('--runs', type=int,
                        default=10)

    parser.add_argument('--lr_rate', type=float,
                        default=1)

    parser.add_argument('--missing_rate', type=float,
                        default=0.0)

    parser.add_argument('--model', type=str,
                        choices=["GCN", "SAGE", "GIN"],
                        default="GCN")

    parser.add_argument("--fill", type=str, default="fp",
                        choices=["fp", "apcfi", "zero"],
                        help="filling method")


    parser.add_argument('--pmlp', type=bool,
                        default=False)
    parser.add_argument('--norm', type=bool,
                        default=False)

    parser.add_argument('--classic', action='store_true')


    exp_args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Norm: {exp_args.norm}")

    apcfi_args = {"cora": [[0.9, 10**(-1)], [0.7, 10**(-2)]],
                  "citeseer": [[0.5, 10**(0)], [0.8, 10**(-5)]],
                  "pubmed": [[0.8, 10**(0)], [0.7, 10**(-2.5)]],
                  "photo": [[0.2, 10**(-6)], [0.2, 10**(-1.5)]],
                  "computers": [[0.2, 10**(-4)], [0.1, 10**(-2.5)]],
                  "cs": [[0.8, 10**(-6)], [0.2, 10**(-4)]],
    }

    data_args = {
                 "cora" : [512, 3, 0.7, "class", "group", True, 200, 0.001, 0.001, False, False, False, False], #group
                 "citeseer": [512, 2, 0.5, "class", "group", True, 200, 0.001, 0.001, False, False, False, False],
                 "pubmed": [256, 2, 0.7, "class", "group", True, 200, 0.005, 0.001, False, False, False, False],
                 "photo": [256, 6, 0.5, "fixed", "group", True, 500, 0.001, 0.00025, False, False, True, False], #0.0001
                 "computers": [512, 3, 0.5, "fixed", "group", True, 500, 0.0001, 0.00075, False, True, False, True], #sage: group, #gcn: 0.01
                 "cs": [512, 2, 0.3, "fixed", "group", True, 500, 0.0001, 0.001, False, True, True, True],
    }

    runs = exp_args.runs
    missing_rate = exp_args.missing_rate
    exp_args.fill = exp_args.fill if missing_rate > 0 else None

    for data_name, args in data_args.items():
        fix_seed(42)
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

                print(f"\n\n{data_name} run {run + 1} / {runs}")

                test_acc, train_time = exp(exp_args.methood, mlp,
                                           imp=args[4], isomorphic=args[5],
                                           mlp_epoch=args[6], lr=args[7] * exp_args.lr_rate, mlp_lr=args[8], init=args[12])
                acc_dict[exp_args.methood][0].append(test_acc)
                acc_dict[exp_args.methood][1].append(train_time)

            torch.cuda.empty_cache()
            gc.collect()
            save_log(acc_dict[exp_args.methood], exp_args.methood, data_name,
                     exp_args.fill, exp_args.missing_rate, exp_args.classic, "log")


