
__all__ = ["SAGEConvMLP", "GCNConvMLP", "GINConvMLP", "ChebConvMLP", "GATConvMLP",
           "SAGEConv", "CuGraphSAGEConv", "GCNConv", "GINConv", "ChebConv", "GATConv", "CuGraphGATConv",
           "Linear"]

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv, ChebConv, MessagePassing
from torch_geometric.nn.inits import zeros

from limelight.utils.tool import pack_args, extract_init_args

try:
    from torch_geometric.nn import CuGraphSAGEConv, CuGraphGATConv

except ImportError:
    CuGraphSAGEConv, CuGraphGATConv= None, None


class BaseConvMLP(nn.Module):
    pruning_dim = None
    prunable = True
    def __init__(self):
        super().__init__()
        self.params = SimpleNamespace(**pack_args(2))
        self._get_kwargs()

    def _get_kwargs(self):
        self._kwargs = tuple(vars(self.params).keys())

    def _parse_conv(self, conv):
        _conv_args = extract_init_args(conv)
        self.params.__dict__.update(_conv_args)

    @property
    def kwargs(self):
        kwargs = {k: getattr(self.params, k) if k not in ['in_channels', 'out_channels'] else getattr(self, k)
                  for k in self._kwargs if k != 'conv'}
        return kwargs

    @property
    def unwrapped_parameters(self):
         return [(param, self.pruning_dim if self.pruning_dim is not None else param.dim() - 1)
                 for name, param in self.named_parameters(recurse=False)]


class GCNConvMLP(BaseConvMLP):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 improved: bool = False,
                 normalize: bool = True,
                 bias: bool = True,
                 conv: MessagePassing = None,
                 **kwargs):

        super().__init__()
        if not isinstance(conv, GCNConv) and (in_channels is None or out_channels is None):
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"`in_channels` and `out_channels` to be "
                             f"specified if `conv` not `nn.GCNConv`.")

        if isinstance(conv, GCNConv):
            self._parse_conv(conv)
            self.params.bias = True if self.params.bias is not None else False

        self.normalize = self.params.normalize


        in_channels = self.params.in_channels
        out_channels = self.params.out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

        if self.params.bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    @property
    def in_channels(self):
        return self.lin.in_features

    @property
    def out_channels(self):
        return self.lin.out_features

    def forward(self, x, *args):
        out = x
        out = self.lin(out)
        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, bias={self.params.bias})'


class SAGEConvMLP(BaseConvMLP):
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 normalize: bool = False,
                 root_weight: bool = True,
                 project: bool = False,
                 bias: bool = True,
                 conv: MessagePassing = None,
                 **kwargs):
        super().__init__()
        if not(isinstance(conv, SAGEConv) or  isinstance(conv, CuGraphSAGEConv)) and (in_channels is None or out_channels is None):
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"`in_channels` and `out_channels` to be "
                             f"specified if `conv` is not `SAGEConv`.")

        if isinstance(conv, SAGEConv) or isinstance(conv, CuGraphSAGEConv):
            self._parse_conv(conv)

        self.normalize = self.params.normalize
        self.root_weight = self.params.root_weight
        self.project = self.params.project
        self.bias = self.params.bias

        in_channels = self.params.in_channels
        out_channels = self.params.out_channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")

            self.lin = nn.Linear(in_channels[0], in_channels[0], bias=True)

        self.lin_l = torch.nn.Linear(in_channels[0], out_channels, bias=self.params.bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    @property
    def in_channels(self):
        return self.lin_l.in_features

    @property
    def out_channels(self):
        return self.lin_l.out_features

    def forward(self, x, *args):
        """"""
        if isinstance(x, torch.Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        out, x_r = x[0], x[1]
        out = self.lin_l(out)
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, bias={self.params.bias})'


class GATConvMLP(BaseConvMLP):
    prunable = ["lin"]
    def __init__(self,
                 in_channels = None,
                 out_channels = None,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 edge_dim = None,
                 fill_value = 'mean',
                 bias: bool = True,
                 residual: bool = False,
                 conv: MessagePassing = None,
                 **kwargs):

        super().__init__()
        if not isinstance(conv, (GATConv, CuGraphGATConv)) and (in_channels is None or out_channels is None):
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"`in_channels` and `out_channels` to be "
                             f"specified if `conv` not `nn.GATConv`.")

        if isinstance(conv, (GATConv, CuGraphGATConv)):
            self._parse_conv(conv)
            self.params._out_channels = self.params.out_channels // self.params.heads
            self.params.bias = True if self.params.bias is not None else False

        self.heads = self.params.heads
        self.concat = self.params.concat

        in_channels = self.params.in_channels
        out_channels = self.params.out_channels
        residual = self.params.residual
        total_out_channels = out_channels * (self.heads if self.params.concat else 1)


        self.lin = nn.Linear(in_channels, self.heads * out_channels, bias=False)
        if residual:
            self.res = nn.Linear(in_channels
                                if isinstance(in_channels, int) else in_channels[1],
                                total_out_channels,
                                bias=False)
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)
        #self.bias = nn.Parameter(torch.empty(total_out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        #if self.params.bias:
        #    zeros(self.bias)
        zeros(self.bias)

    @property
    def in_channels(self):
        return self.lin.in_features

    @property
    def out_channels(self):
        return self.lin.out_features // (self.heads if self.concat else 1)

    def forward(self, x, *args):
        H, C = self.heads, self.out_channels
        res = None

        if isinstance(x, torch.Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x  = self.lin(x).view(1, H, -1)

        else:
            x_src, x_dst = x
            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            x = self.lin(x_src).view(-1, H, C)

        out = x
        if self.concat:
            out = out.view(-1, H * C)

        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads}, bias={self.params.bias})'


class GINConvMLP(BaseConvMLP):
    def __init__(self,
                 nn = None,
                 eps = 0.0,
                 train_eps: bool = True,
                 conv: MessagePassing = None,
                 **kwargs):

        super().__init__()
        if not isinstance(conv, GINConv) and (nn is None):
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"`nn` to be specified if `conv` is not `GINConv`.")

        if isinstance(conv, GINConv):
            self._parse_conv(conv)
            self.params.train_eps = self.params.eps.requires_grad
            self.params.eps = self.params.eps[0]

        self.initial_eps = self.params.eps
        if self.params.train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        self.nn = self.params.nn
        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.nn.inits import reset
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    @property
    def in_channels(self):
        for layer in self.nn:
            if hasattr(layer, 'in_features'):
                return layer.in_features
            elif hasattr(layer, 'in_channels'):
                return layer.in_channels

    @property
    def out_channels(self):
        for layer in self.nn[::-1]:
            if hasattr(layer, 'out_features'):
                return layer.out_features
            elif hasattr(layer, 'out_channels'):
                return layer.out_channels

    def forward(self, x):
        out = x + (1 + self.eps) * x
        out = self.nn(out)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class ChebConvMLP(BaseConvMLP):
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 K: int = 3,
                 normalization= 'sym',
                 bias=True,
                 conv=None,
                 **kwargs):

        super().__init__()
        if not isinstance(conv, ChebConv) and (nn is None):
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"`nn` to be specified if `conv` is not `GINConv`.")

        if isinstance(conv, ChebConv):
            self._parse_conv(conv)
            self.params.bias = True if self.params.bias is not None else False
            self.params.K = len(conv.lins)

        self.K = self.params.K

        in_channels = self.params.in_channels
        out_channels = self.params.out_channels

        self.lins = torch.nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)])
        if self.params.bias:
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
        from torch_geometric.nn.inits import zeros, glorot
        for lin in self.lins:
            glorot(lin.weight)
        zeros(self.bias)

    def forward(self, x):

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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, bias={self.params.bias})'
