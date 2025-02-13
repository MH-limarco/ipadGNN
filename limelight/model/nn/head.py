
import torch
from torch import nn
from limelight.model.nn.activation import get_activation


class BaseModule(nn.Module):
    def __init__(self,):
        super(BaseModule, self).__init__()
        pass

    def forward(self,):
        raise NotImplementedError


class MPNNs(BaseModule):
    def __init__(self,):
        from limelight.model.nn.mpnn import SingleMPNN


class MLP(BaseModule):
    def __init__(self,):
        pass

class Boosting(BaseModule):
    def __init__(self,):
        pass