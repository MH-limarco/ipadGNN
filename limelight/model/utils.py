
__all__ = ["get_activation", "get_normalize"]

import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, Optional, overload

from limelight.api import parse_config

CONFIG = parse_config()
DEFAULT_ACTIVATION = CONFIG.activation
DEFAULT_NORMALIZE = CONFIG.normalize

activation_map = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": lambda: nn.Softmax(dim=-1),
    "log_softmax": lambda: nn.LogSoftmax(dim=-1),
    "swiglu": lambda in_channels, out_channels=None: SwiGLU(in_channels, out_channels)
}

normalize_map = {
    "identity": nn.Identity,
    "batch": nn.BatchNorm1d,
    "layer": nn.LayerNorm,
    "instance": nn.InstanceNorm1d,
}


class SwiGLU(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(SwiGLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.fc = nn.Linear(in_channels, 2 * self.out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        gate, x = self.fc(x).chunk(2, dim=-1)
        return F.silu(gate) * x

def init_flow(cls_map, default_name: str, name: str):
    default_name = default_name.lower()
    if default_name not in cls_map:
        pass
        #raise ValueError(f"Unknown Default function: {default_name}. "
        #                 f"Available options: {list(cls_map.keys())}")

    name = default_name if name is None else name.lower()
    if name not in cls_map:
        raise ValueError(f"Unknown function: {name}. "
                         f"Available options: {list(cls_map.keys())}")

    return cls_map[name], name


@overload
def get_activation(name: str, return_class: bool) -> Type[nn.Module]: ...

@overload
def get_activation(name: str, return_class: bool = False, **kwargs) -> nn.Module: ...

def get_activation(name: Optional[str] = None, return_class: bool = False, **kwargs) -> Union[nn.Module, Type[nn.Module], None]:
    """
    根據名稱返回對應的激活函數

    :param name: 激活函數名稱（大小寫皆可），如果為 None 則回傳 None。
    :param return_class: 是否回傳 nn.Module 類別，而不是已初始化的函數
    :return: 對應的 nn.Module 激活函數或類別
    """
    activation_cls, name = init_flow(activation_map, DEFAULT_ACTIVATION, name)

    if name == "swiglu" and not return_class:
        if "in_channels" not in kwargs:
            raise ValueError("SWiGLU requires `in_channels` and `out_channels` as an argument.")
        return activation_cls(**kwargs)

    return activation_cls if return_class else activation_cls()


@overload
def get_normalize(name: str, return_class: bool = False, in_channels: int = None, **kwargs) -> nn.Module: ...

def get_normalize(name: Optional[str] = None, return_class: bool = False, in_channels: int = None, **kwargs) -> Union[nn.Module, Type[nn.Module], None]:
    """
    根據名稱返回對應的正規層

    :param name: 正規層名稱（大小寫皆可），如果為 None 則回傳 None
    :param return_class: 是否回傳 nn.Module 類別，而不是已初始化的函數
    :param in_channels: 正規層的輸入維度
    :return: 對應的 nn.Module 正規層或類別
    """
    normalize_cls, name = init_flow(normalize_map, DEFAULT_NORMALIZE, name)

    if not return_class:
        if in_channels is None:
            raise ValueError("Normalization layer requires `in_channels` as an argument.")
        return normalize_cls(in_channels, **kwargs)

    return normalize_cls


if __name__ == "__main__":
    import torch
    x = torch.randn(2, 64)
    act = get_activation("SWiGLU", in_channels=64, out_channels=128)
    print(act(x).shape)

    norm = get_normalize("layer", in_channels=64)
    print(norm(x).shape)
