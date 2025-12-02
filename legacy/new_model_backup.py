
__all__ = ['MainModule', 'mirror_projection', 'projection_layer', "cugraph"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import CuGraphGATConv, GATConv
from torch_geometric import EdgeIndex
from copy import deepcopy
import gc

from limelight.utils.tool import get_func_args
from new_layer import *

try:
    import pylibcugraphops
    cugraph = True

except ImportError:
    cugraph = False

cugraph = True


MIRROR_ATTR = ["layers", "cls_conv"]


def get_activation(activation, channels):
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

def get_norm(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm1d(channels)
    elif norm == 'layer':
        return nn.LayerNorm(channels)
    elif not norm or norm.lower() == 'none' or norm is None:
        return None
    else:
        raise ValueError(f"Unknown normalization type: {norm}")


def get_conv(gnn_type, in_channels, out_channels,
             dropout=0.5, normalize=None, train_eps=False,
             cugraph=False):
    _gnn_type = gnn_type.lower()
    if 'gcn' in _gnn_type:
        normalize = normalize if normalize is not None else True
        conv = GCNConv(in_channels, out_channels, cached=False, normalize=normalize)
        if 'mlp' in _gnn_type:
            conv = GCNConvMLP(conv=conv)
    elif 'sage' in _gnn_type:
        conv = None
        normalize = normalize if normalize is not None else True
        if cugraph and CuGraphSAGEConv is not None:
            conv = CuGraphSAGEConv(in_channels, out_channels, normalize=normalize)

        if conv is None:
            conv = SAGEConv(in_channels, out_channels, normalize=normalize)

        if 'mlp' in _gnn_type:
            conv = SAGEConvMLP(conv=conv)
    elif 'gat' in _gnn_type:
        if cugraph:
            if cugraph:
                conv = CuGraphGATConv(in_channels, out_channels)
        else:
            conv = GATConv(in_channels, out_channels)

    elif 'cheb' in _gnn_type:
        conv = ChebConv(in_channels, out_channels, K=3)
        if 'mlp' in _gnn_type:
            conv = ChebConvMLP(conv=conv)

    elif 'gin' in _gnn_type:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            #torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_channels, out_channels),
            #torch.nn.BatchNorm1d(out_channels),
            #torch.nn.ReLU(),
            #torch.nn.Dropout(dropout),
            #torch.nn.Linear(out_channels, out_channels),
        )
        conv = GINConv(nn=mlp, eps=0.0, train_eps=train_eps)

        if 'mlp' in _gnn_type:
            conv = GINConvMLP(conv=conv)
    elif _gnn_type == "linear":
        conv = nn.Linear(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")

    return conv, 'edge_index' in get_func_args(conv.forward)


class MainModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layer=3, dropout=0.5,
                 gnn_type='gcn', gnn_cls=True,
                 act='relu', act_first=False,
                 norm=False, res=False,
                 normalize=True, train_eps=False, cugraph=False,
                 **kwargs):
        super().__init__()
        self._CuGraph = False
        self.act_first = act_first
        self.dropout = dropout
        self.norm = True if isinstance(norm, str) and norm.lower != 'none' else norm
        self.res = res

        if num_layer > 0:
            self.layers = nn.ModuleList()
            self.acts = nn.ModuleList()
            self._use_edges = []

            if self.norm:
                self.norms = nn.ModuleList()

            if res:
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
            self.cls_conv, _ = get_conv(gnn_type, _channels, out_channels, dropout)
        else:
            self.cls_conv = nn.Linear(_channels, out_channels)

        _cls_use_edges = 'edge_index' in get_func_args(self.cls_conv.forward)
        self._use_edges.append(_cls_use_edges)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                if CuGraphSAGEConv is not None and isinstance(layer, CuGraphSAGEConv):
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

    def forward(self, x, edge_index=None, **kwargs):

        #if isinstance(edge_index, torch.Tensor) and self._CuGraph:
        #    edge_index = EdgeIndex(edge_index)

        if hasattr(self, 'layers'):
            x_out = 0 if len(self.layers) > 0 else x
            for idx, layer in enumerate(self.layers):
                if self.res:
                    x = layer(x, edge_index) if self._use_edges[idx] else layer(x) + self.res_layers[idx](x)
                else:
                    x = layer(x, edge_index) if self._use_edges[idx] else layer(x)

                #if self.act_first:
                #    x = self.acts[idx](x)

                if hasattr(self, 'norms'):
                    x = self.norms[idx](x)

                #if not self.act_first:
                x = self.acts[idx](x)

                #if not isinstance(layer, (nn.Linear, GCNConvMLP, SAGEConvMLP, GINConvMLP)):
                x = F.dropout(x, p=self.dropout, training=self.training)

                x_out = x

        if self._use_edges[-1]:
            x_out = self.cls_conv(x_out, edge_index)
        else:
            x_out = self.cls_conv(x_out)
        return x_out

def projection_layer(conv, gnn2mlp=True, cugraph=False, debug=False):
    _name = conv.__class__.__name__
    if "MLP" in _name:
        if gnn2mlp:
            return conv
        _params = conv.kwargs
        if "SAGEConv" in _name:
            if cugraph and cugraph:
                _name = "CuGraphSAGEConv"
            else:
                _name = "SAGEConv"
        return globals()[_name.replace("MLP", "")](**_params)

    elif "Conv" in _name:
        if not gnn2mlp:
            return conv
        if _name == "CuGraphSAGEConv":
            _name = "SAGEConv"
        return globals()[f"{_name}MLP"](conv=conv)

    elif isinstance(conv, nn.Linear):
        return conv

    else:
        if debug:
            return conv
        else:
            raise NotImplementedError(f"Unsupported layer: {conv.__class__.__name__}")


def mirror_projection(module, gnn2mlp=True, cugraph=False, mirror_attr=None, load_state=False):
    if mirror_attr is None:
        mirror_attr = MIRROR_ATTR

    new_module = deepcopy(module)

    state = new_module.state_dict() if load_state else None
    for attr in mirror_attr:
        if hasattr(new_module, attr):
            _module = getattr(new_module, attr)
            if isinstance(_module, nn.ModuleList):
                for idx, layer in enumerate(_module):
                    _module[idx] = projection_layer(layer, gnn2mlp, cugraph)
            elif isinstance(_module, nn.Module):
                setattr(new_module, attr, projection_layer(_module, gnn2mlp, cugraph))

    if state is not None:
        new_module.load_state_dict(state)
    else:
        new_module.reset_parameters()

    return new_module


if __name__ == "__main__":
    in_dim = 512


    x = torch.randn(1024, in_dim)
    edge_index = torch.randint(0, in_dim, (2, 1024))

    a = MainModule(in_dim, 256, 30, 6
                   ,dropout=0.6, res=True, norm='layer',gnn_type='arma', gnn_cls=True)

    b = mirror_projection(a, True)

    print(a.layers)
    print(b.layers)
    print(a(x, edge_index).shape)
    print(b(x, edge_index).shape)

    # b = mirror_projection(a, False)

