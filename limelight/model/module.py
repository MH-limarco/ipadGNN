
__all__ = ["NormModule", "BoostModule"]

import torch
import torch_geometric
import torch.nn.functional as F
from torch import nn

from typing import Dict
import argparse
import inspect

from limelight.model.utils import get_activation, get_normalize
from limelight.model.mpnn import SingleMPNN
from limelight.utils import parse_args

from limelight.api import parse_config


CONFIG = parse_config()
NUMBER_LAYER = CONFIG.num_layers
DROPOUT = CONFIG.dropout
CONV_MAP = torch_geometric.nn.conv.__all__

class BaseModule(nn.Module):
    def __init__(self, args: [Dict, argparse.Namespace], **kwargs):
        super(BaseModule, self).__init__()
        self.args = parse_args(args)
        if not (self.args["conv_type"] in CONV_MAP or
                self.args["conv_type"].lower() in ["mlp", "linear"]):
            raise ValueError(f"Unknown layer: {self.args['conv_type']}. "
                             f"Available: {['MLP', 'Linear'] + CONV_MAP}")

        self.conv_type = "mpnn" if self.args["conv_type"] in CONV_MAP else "mlp"
        self.num_layers = self.args.get("num_layers", NUMBER_LAYER)
        self.dropout = self.args.get("dropout", DROPOUT)
        self.res = self.args.get("res", False)
        self.jk = self.args.get("jk", False)

        self.input_type = None
        self.layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()

        self._read_args()
        self._build_input_layer()
        self._build_module()
        self._build_pred_layer()

        self.check_input_type()
        self.reset_parameters()

    def build_layer(self, in_channels, out_channels):
        if self.conv_type == "mpnn":
            return SingleMPNN(self.args, in_channels, out_channels)
        return nn.Linear(in_channels, out_channels)

    def __call__(self, data: [torch.Tensor, torch_geometric.data.Data], **kwargs):
        if self.input_type is None:
            raise ValueError("Input type is not defined.")

        elif self.input_type == "graph" and isinstance(data, torch.Tensor):
            raise ValueError("Input type is `graph` but input is tensor.")

        elif isinstance(data, torch_geometric.data.Data):
            kwargs["edge_index"] = data.edge_index

        _input = data.x
        return self.forward(_input, **kwargs)

    def reset_parameters(self,):
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x, edge_index=None, **kwargs):
        raise NotImplementedError

    def _read_args(self):
        self.in_channels = self.args["in_channels"]
        self.hidden_channels = self.args["hidden_channels"]
        self.out_channels = self.args["num_classes"]

        self.pre_linear = self.args.get("pre_linear", False)
        self.norm_type = self.args.get("norm_type", None)
        self.act_type = self.args.get("act_type", None)

    def _build_module(self):
        raise NotImplementedError

    def _build_input_layer(self):
        if self.pre_linear:
            self.input_layer = nn.Linear(self.in_channels, self.hidden_channels)
            self.in_channels = self.hidden_channels
        else:
            self.input_layer = None

    def _build_pred_layer(self):
        self.pred_layer = nn.Linear(self.in_channels, self.out_channels)

    def check_input_type(self):
        self.input_type = self._check_input_type()
        #print(f"Input type: {self.input_type}")

    def _check_input_type(self):
        if len(self.layers) > 0:
            if inspect.signature(self.layers[0].forward).parameters.get("edge_index", None):
                return "graph"
        return "feature"


class NormModule(BaseModule):
    def _build_module(self):
        for i in range(self.num_layers):
            if self.res:
                self.res_layers.append(nn.Linear(self.in_channels, self.hidden_channels))

            self.layers.append(self.build_layer(self.in_channels, self.hidden_channels))
            if self.norm_type not in [None, "identity"]:
                self.norms.append(get_normalize(self.norm_type, in_channels=self.hidden_channels))

            self.acts.append(get_activation(self.act_type, in_channels=self.hidden_channels))
            self.in_channels = self.hidden_channels

    def forward(self, x, edge_index=None, **kwargs):
        if self.pre_linear:
            x = self.input_layer(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_out = 0 if len(self.layers) > 0 else x
        for idx, layer in enumerate(self.layers):
            _input = (x, edge_index) if self.input_type == "graph" else (x,)
            if self.res:
                x = layer(*_input) + self.res_layers[idx](x)
            else:
                x = layer(*_input)

            if self.norm_type not in [None, "identity"]:
                x = self.norms[idx](x)
            x = self.acts[idx](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk:
                x_out = x_out + x
            else:
                x_out = x

        return self.pred_layer(x_out)


class BoostModule(BaseModule):
    pass

if __name__ == "__main__":
    model_map = {"in_channels": 256, "hidden_channels": 256,
                 "num_classes": 1, "num_layers": 3,
                 "res": True, "jk": False, "pre_linear": False,
                 "conv_type": "linear", "heads": 4,
                 "K": 1,
                 "norm_type": "layer", "act_type": "swiglu"}
    a = NormModule(model_map)

    x = torch_geometric.data.Data(x=torch.randn(2, 64),
                                  edge_index=torch.tensor([[0, 1], [1, 0]]))
    print(a(x).shape)
