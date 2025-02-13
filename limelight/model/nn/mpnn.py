
from torch import nn
from torch_geometric import nn as pygnn
from torch_geometric.data import Data
import torch

from typing import Union, overload, Dict, Any
import argparse, inspect

__all__ = ["SingleMPNN"]

class SingleMPNN(nn.Module):
    @overload
    def __init__(self, args: Dict[str, Any], in_channels: int, out_channels: int, **kwargs): ...

    @overload
    def __init__(self,
                 args: argparse.Namespace, in_channels: int, out_channels: int, **kwargs): ...

    def __init__(self, args, in_channels, out_channels, **kwargs):
        super(SingleMPNN, self).__init__()

        if isinstance(args, argparse.Namespace):
            args = vars(args)

        self.conv_type = args.get("conv_type", "GCNConv")

        if hasattr(pygnn.conv, self.conv_type):
            _conv = getattr(pygnn.conv, self.conv_type)
        else:
            raise ValueError(f"Unknown gnn layer: {self.conv_type}. Available: {dir(pygnn.conv)}")

        conv_params = inspect.signature(_conv.__init__).parameters

        if "nn" in conv_params:
            raise ValueError(f"Not support GINConv and GINEConv yet.")
        elif "conv" in conv_params:
            raise ValueError(f"Not support GPSConv yet.")
        elif "heads" in conv_params:
            args["heads"] = args.get("heads", 1)
            out_channels = out_channels // args["heads"]

        kwargs.update({k: v for k, v in args.items() if k in conv_params and k not in ["in_channels", "out_channels"]})
        self.conv = _conv(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        self.conv.reset_parameters()

    @overload
    def forward(self, input_data:torch.Tensor, edge_index:torch.Tensor, **kwargs): ...

    @overload
    def forward(self, input_data:Data, **kwargs): ...

    def forward(self, input_data: Union[torch.Tensor, Data], edge_index: torch.Tensor = None, **kwargs):
        if isinstance(input_data, Data):
            x = input_data.x
            edge_index = input_data.edge_index
        else:
            x = input_data

        return self.conv(x, edge_index, **kwargs)


if __name__ == "__main__":
    _args = {"conv_type": "GATv2Conv",
             "heads": 4,
             "K": 1,}
    dim = 16
    node = 10
    model = SingleMPNN(_args, dim, dim)
    input = torch.randn(node, dim)
    edge = torch.randint(0, node, (2, 32))
    out = model(input, edge)


