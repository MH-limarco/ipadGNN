

from limelight.utils import parse_args
from limelight.model import *


def build_model(args):
    args = parse_args(args)
    model_type = args['model_type'].lower()
    if model_type == "boost":
        return BoostModule(args)
    else:
        return NormModule(args)

if __name__ == "__main__":
    import torch_geometric
    import torch

    args = {"model_type": "mpnn",
            "conv_type": "linear",
            "in_channels": 64, "hidden_channels": 256,
            "out_channels": 1, "num_layers": 3,
            "heads": 4, "K": 1,
            "res": True, "jk": False, "pre_linear": True,
            "norm_type": "layer", "act_type": "swiglu"}

    model = build_model(args)
    x = torch_geometric.data.Data(x=torch.randn(2, 64),
                                  edge_index=torch.tensor([[0, 1], [1, 0]]))

    print(model(x))
