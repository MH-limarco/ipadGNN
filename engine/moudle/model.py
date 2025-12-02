
__all__ = ['LimelightModule', 'mirror_projection', 'projection_layer']

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from limelight.utils.tool import get_func_args
import engine.moudle.layer as lime_layer

MIRROR_ATTR = ["layers", "cls_conv"]
Safety = True


def get_activation(activation, channels=None):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def get_norm(norm, channels, eps=1e-5):

    if not norm or norm.lower() == 'none' or norm is None:
        return None
    elif norm.lower() == 'batch':
        return nn.BatchNorm1d(channels, eps=eps)
    elif norm.lower() == 'layer':
        return nn.LayerNorm(channels, eps=eps)
    else:
        raise ValueError(f"Unknown normalization type: {norm}")


def get_conv(gnn_type, in_channels, out_channels,
             dropout=0.5, normalize=None, train_eps=False,
             cugraph=False):
    _gnn_type = gnn_type.lower()
    if 'gcn' in _gnn_type:
        normalize = normalize if normalize is not None else True
        conv = lime_layer.GCNConv(in_channels, out_channels, cached=False, normalize=normalize)
        if 'mlp' in _gnn_type:
            conv = lime_layer.GCNConvMLP(conv=conv)

    elif 'sage' in _gnn_type:
        normalize = normalize if normalize is not None else True
        if cugraph and lime_layer.CuGraphSAGEConv is not None:
            conv = lime_layer.CuGraphSAGEConv(in_channels, out_channels, normalize=normalize)
        else:
            conv = lime_layer.SAGEConv(in_channels, out_channels, normalize=normalize)

        if 'mlp' in _gnn_type:
            conv = lime_layer.SAGEConvMLP(conv=conv)

    elif 'gat' in _gnn_type:
        if cugraph and lime_layer.CuGraphSAGEConv is not None:
            conv = lime_layer.CuGraphGATConv(in_channels, out_channels//4, heads=4, concat=True, bias=False)
        else:
            conv = lime_layer.GATConv(in_channels, out_channels//4, heads=4, concat=True,
                           add_self_loops=False, bias=False)

        if 'mlp' in _gnn_type:
            conv = lime_layer.GATConvMLP(conv=conv)

    elif 'cheb' in _gnn_type:
        conv = lime_layer.ChebConv(in_channels, out_channels, K=3)
        if 'mlp' in _gnn_type:
            conv = lime_layer.ChebConvMLP(conv=conv)

    elif 'gin' in _gnn_type:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_channels, out_channels),
        )
        conv = lime_layer.GINConv(nn=mlp, eps=0.0, train_eps=train_eps)

        if 'mlp' in _gnn_type:
            conv = lime_layer.GINConvMLP(conv=conv)

    elif _gnn_type == "linear":
        conv = nn.Linear(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")

    return conv, 'edge_index' in get_func_args(conv.forward)


def projection_layer(conv, target, cugraph=False):
    _name = conv.__class__.__name__
    if "MLP" in _name:
        if target == "mlp":
            return conv, False

        _params = conv.kwargs
        if "sage" in _name.lower():
            _name = "SAGEConv"
            if cugraph and lime_layer.CuGraphSAGEConv is not None:
                _name = "CuGraphSAGEConv"

        elif "gat" in _name.lower():
            _name = "GATConv"
            if cugraph and lime_layer.CuGraphGATConv is not None:
                _name = "CuGraphGATConv"
                _params = {k : v for k, v in _params.items()
                           if k in get_func_args(lime_layer.CuGraphGATConv.__init__)}

        return getattr(lime_layer, _name.replace("MLP", ""))(**_params), True

    elif "Conv" in _name:
        if target == "gnn":
            return conv, True
        _name = _name.replace("CuGraph", "")
        return getattr(lime_layer, f"{_name}MLP")(conv=conv), False

    elif isinstance(conv, nn.Linear):
        return conv, False

    elif isinstance(conv, LimelightModule):
        return mirror_projection(conv, target=target, cugraph=cugraph), True

    else:
        raise NotImplementedError(f"Unsupported layer: {conv.__class__.__name__}")


def mirror_projection(module, target, cugraph=False, mirror_attr=None):
    if mirror_attr is None:
        mirror_attr = MIRROR_ATTR

    device = module.device if hasattr(module, "device") else "cuda"

    new_module = deepcopy(module)

    for attr in mirror_attr:
        if hasattr(new_module, attr):
            _module = getattr(new_module, attr)
            if isinstance(_module, nn.ModuleList):
                if attr == "layers":
                    _use_edges = new_module._use_edges
                for idx, layer in enumerate(_module):
                    l, _use_edge = projection_layer(layer, target, cugraph)
                    _module[idx] = l
                    if attr == "layers":
                        _use_edges[idx] = _use_edge

            elif isinstance(_module, nn.Module):
                setattr(new_module, attr, projection_layer(_module, target, cugraph)[0])

    new_module.reset_parameters()
    new_module._config_updated = False
    return new_module.to(device)


class LimelightModule(nn.Module):
    def __init__(self, in_channels=None, hidden_channels=None, out_channels=None,
                 num_layer=3, dropout=0.5,
                 gnn_type='gcn', gnn_cls=True,
                 act='relu', act_first=False,
                 norm=False, res=False,
                 normalize=True, train_eps=False, cugraph=False,
                 project=False, project_channels=1024,
                 pt=None,
                 **kwargs):
        super().__init__()
        self._load_strict = True
        self._config_updated = False

        if pt is not None:
            self._init_with_pt(pt=pt)

        elif not (in_channels is None or hidden_channels is None or out_channels is None):
            self._init_with_args(in_channels, hidden_channels, out_channels,
                                 num_layer, dropout, gnn_type, gnn_cls,
                                 act, act_first, norm, res, project, project_channels,
                                 normalize, train_eps, cugraph)
        else:
            raise ValueError("in_channels, hidden_channels, out_channels must be specified or "
                             "provide a pre-trained model")

        if hasattr(self, "_update_config"):
            self._update_config()


    def _init_with_pt(self, pt):
        if not isinstance(pt, str):
            raise ValueError("pt filename must be a string")

        if not (pt.endswith('.pt') or pt.endswith('.pth')):
            pt = f'{pt}.pt'

        check_point = torch.load(pt, weights_only=False)
        _config = check_point.get('config', None)
        _state_dict = check_point.get('state_dict', None)
        if _config is not None:
            self._module_config = deepcopy(_config)
            self._CuGraph = _config.pop("CuGraph", False)
            self.act_first = _config.pop("act_first", False)
            self.res = _config.get("res_layers", None) is not None
            self.dropout = _config.pop("dropout", 0.5)
            self._use_edges = _config.pop("use_edges", None)
            self._cls_use_edges = _config.pop("cls_use_edges")

            for key, cfg in _config.items():
                if key == "cls_conv":
                    _obj = getattr(lime_layer, cfg[0])
                    self.cls_conv = _obj(**cfg[1])

                else:
                    _module_list = nn.ModuleList()
                    for obj, kwargs in cfg:
                        _obj = getattr(lime_layer, obj) if obj not in [get_activation, get_norm]  else obj
                        if _obj == nn.Linear:
                            kwargs["in_features"] = kwargs.pop("in_channels", None)
                            kwargs["out_features"] = kwargs.pop("out_channels", None)

                        if _obj in [lime_layer.GATConvMLP]:
                            self._load_strict = False

                        _module_list.append(_obj(**kwargs))
                    setattr(self, key, _module_list)

            self._config_updated = True
            if _state_dict is not None:
                self.load_state_dict(_state_dict, strict=self._load_strict)
        else:
            raise ValueError("Invalid checkpoint file")

    def _init_with_args(self, in_channels, hidden_channels, out_channels,
                        num_layer, dropout, gnn_type, gnn_cls,
                        act, act_first, norm, res, project, project_channels,
                        normalize, train_eps, cugraph):

        self._CuGraph = False
        self.act_first = act_first
        self.dropout = dropout
        self.norm = True if isinstance(norm, str) and norm.lower() != 'none' else norm
        self.res = res


        num_layer = num_layer - 1 if gnn_cls else num_layer
        if num_layer > 0:
            self.layers = nn.ModuleList()
            self.acts = nn.ModuleList()
            self._use_edges = []

            if self.norm:
                self.norms = nn.ModuleList()

            if self.res:
                self.res_layers = nn.ModuleList()

        _channels = in_channels
        for idx in range(num_layer):
            _conv, _use_edge = get_conv(gnn_type, _channels, hidden_channels, dropout, normalize, train_eps, cugraph)

            self.layers.append(_conv)
            self._use_edges.append(_use_edge)
            self.acts.append(get_activation(act, hidden_channels))
            if hasattr(self, 'res_layers'):
                self.res_layers.append(nn.Linear(_channels, hidden_channels))
            if hasattr(self, 'norms'):
                self.norms.append(get_norm(norm, hidden_channels))
            _channels = hidden_channels

        if gnn_cls:
            self.cls_conv, self._cls_use_edges = get_conv(gnn_type, _channels, out_channels, dropout)
            #self._cls_use_edges = True
        else:
            self.cls_conv = nn.Linear(_channels, out_channels)
            self._cls_use_edges = False

        self.reset_parameters()
        self._update_config()

    def reset_parameters(self):
        self._CuGraph = False
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                if layer is not None and isinstance(layer, (lime_layer.CuGraphSAGEConv, lime_layer.CuGraphGATConv)):
                    self._CuGraph = True

        if hasattr(self, 'res_layers'):
            for layer in self.res_layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        if hasattr(self, 'norms'):
            for norm in self.layers:
                if hasattr(norm, 'reset_parameters'):
                    norm.reset_parameters()

        if hasattr(self, 'acts'):
            for act in self.acts:
                if hasattr(act, 'reset_parameters'):
                    act.reset_parameters()

        self.cls_conv.reset_parameters()

    def forward_single(self, idx, emb, edge_index=None, **kwargs):
        _layer, _use_edges = self.layers[idx], self._use_edges[idx]
        _layer_input = (emb, edge_index) if _use_edges else (emb,)

        if hasattr(self, "res_layers"):
            _emb = _layer(*_layer_input) + self.res_layers[idx](emb)
        else:
            _emb = _layer(*_layer_input)

        if self.act_first:
            _emb = self.acts[idx](_emb)

        if hasattr(self, "norms"):
            _emb = self.norms[idx](_emb)

        if not self.act_first:
            _emb = self.acts[idx](_emb)

        _emb = F.dropout(_emb, p=self.dropout, training=self.training)
        return _emb

    def forward(self, x, edge_index=None, **kwargs):
        emb = x.clone()

        if hasattr(self, 'layers'):
            for idx in range(len(self.layers)):
                emb = self.forward_single(idx, emb, edge_index, **kwargs)

        if self._cls_use_edges:
            logit = self.cls_conv(emb, edge_index)
        else:
            try:
                logit = self.cls_conv(emb)
            except:
                self._cls_use_edges = True
                logit = self.cls_conv(emb, edge_index)
        return logit

    def forward_with_hidden(self, x, edge_index=None, **kwargs):
        emb = x.clone()
        hidden = []
        if hasattr(self, 'layers'):
            for idx in range(len(self.layers)):
                emb = self.forward_single(idx, emb, edge_index, **kwargs)
                hidden.append(emb)

        if self._cls_use_edges:
            logit = self.cls_conv(emb.float(), edge_index)
        else:
            logit = self.cls_conv(emb)
        return logit, hidden

    def _update_config(self):
        self._CuGraph = False
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if layer is not None and isinstance(layer, (lime_layer.CuGraphSAGEConv, lime_layer.CuGraphGATConv)):
                    self._CuGraph = True

        _config = {"CuGraph": self._CuGraph,
                   "use_edges": self._use_edges,
                   "cls_use_edges": self._cls_use_edges,
                   "act_first": self.act_first,
                   "dropout": self.dropout,
                   }

        _module = deepcopy(self.cpu())
        mlp_module = mirror_projection(_module, target="mlp")

        if hasattr(self, 'layers'):
            _layers = []
            for obj, layer in zip(self.layers, mlp_module.layers):
                if hasattr(layer, 'kwargs'):
                    _kwargs = layer.kwargs
                elif isinstance(layer, nn.Linear):
                    _kwargs = {"in_channels": layer.in_features,
                               "out_channels": layer.out_features,
                               "bias": layer.bias is not None}

                _layers.append((obj.__class__.__name__, _kwargs))
            _config['layers'] = _layers

        if hasattr(self, 'res_layers'):
            _res_layers = []
            for obj, res_layer in zip(self.res_layers, mlp_module.res_layers):
                if isinstance(res_layer, nn.Linear):
                    _kwargs = {"in_channels": res_layer.in_features,
                               "out_channels": res_layer.out_features,
                               "bias": res_layer.bias is not None}
                    _res_layers.append((obj.__class__.__name__, _kwargs))
                else:
                    raise NotImplementedError(f"Unsupported res-layer: {res_layer.__class__.__name__}")
            _config['res_layers'] = _res_layers

        if hasattr(self, 'norms'):
            _norms = []
            for obj, norm in zip(self.norms, mlp_module.norms):
                if hasattr(norm, 'weight'):
                    channels = norm.weight.shape
                else:
                    raise NotImplementedError(f"Unsupported norm-layer: {norm.__class__.__name__}")
                _norm_name = obj.__class__.__name__.lower()
                _kwargs = {"norm": _norm_name.replace("norm", "").replace("1d", ""),
                           "channels": channels,
                           "eps": norm.eps}

                _norms.append((get_norm, _kwargs))
            _config['norms'] = _norms

        if hasattr(self, 'acts'):
            _acts = []
            for obj, act in zip(self.acts, mlp_module.acts):
                channels = obj.channels if hasattr(obj, "channels") else None
                _kwargs = {"activation": obj.__class__.__name__.lower(),
                           "channels": channels}
                _acts.append((get_activation, _kwargs))
            _config['acts'] = _acts

        if hasattr(self, 'cls_conv'):
            if isinstance(mlp_module.cls_conv, nn.Linear):
                _kwargs = {"in_features": mlp_module.cls_conv.in_features,
                           "out_features": mlp_module.cls_conv.out_features,
                           "bias": mlp_module.cls_conv.bias is not None}
            elif hasattr(mlp_module.cls_conv, 'kwargs'):
                _kwargs = mlp_module.cls_conv.kwargs

            else:
                raise NotImplementedError(f"Unsupported cls-layer: {self.cls_conv.__class__.__name__}")

            _cls_conv = [self.cls_conv.__class__.__name__, _kwargs]

            _config['cls_conv'] = _cls_conv

        del mlp_module
        self._module_config = _config
        self._config_updated = True

    @property
    def config(self):
        if not self._config_updated:
            self._update_config()

        return self._module_config

    @property
    def device(self):
        return next(self.parameters()).device

    def save(self, path, full=True):
        if not (path.endswith('.pt') or path.endswith('.pth')):
            path = f'{path}.pt'

        if full:
            check_point = {"state_dict": self.state_dict(),
                           "config": self.config,
                           }
        else:
            check_point = self.state_dict()

        torch.save(check_point, path)

    def load(self, path):
        if not (path.endswith('.pt') or path.endswith('.pth')):
            path = f'{path}.pt'

        check_point = torch.load(path, weights_only=False)
        if "config" in check_point:
            if not self._config_updated:
                self._update_config()

            if self._module_config == check_point['config']:
                self.load_state_dict(check_point['state_dict'])
            else:
                raise ValueError("Checkpoint config does not match the model config. "
                                 "Please use the same config to initialize the model.")
        else:
            self.load_state_dict(check_point)

if __name__ == "__main__":
    import time

    gnn_type = "gcn"
    gnn_cls = True
    res = True
    norm = "layer"

    start = time.perf_counter()
    a = LimelightModule(16, 32, 8, num_layer=2,
                        gnn_type=gnn_type, gnn_cls=gnn_cls, res=res, norm=norm)
    print(a)
    #
    print(f"Model created in: {1000 * (time.perf_counter() - start):.2f}ms")

    start = time.perf_counter()
    a.save('test.pt')
    print(f"Model saved in: {1000 * (time.perf_counter() - start):.2f}ms")

    start = time.perf_counter()
    b = LimelightModule(pt='test.pt')
    print(f"Model rebuild in: {1000 * (time.perf_counter() - start):.2f}ms")

    d = LimelightModule(16, 32, 8, num_layer=2,
                        gnn_type=gnn_type, gnn_cls=gnn_cls, res=res, norm=norm)

    start = time.perf_counter()
    d.load('test.pt')
    print(f"Model loaded in: {1000 * (time.perf_counter() - start):.2f}ms")
