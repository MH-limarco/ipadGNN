
__all__ = ["get_activation"]

import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, Optional, overload


_activation_map = {
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
    "swiglu": lambda in_channels, out_channels=None: SWiGLU(in_channels, out_channels)
}


class SWiGLU(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(SWiGLU, self).__init__()
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
    name = "identity" if name is None else name.lower()
    if name not in _activation_map:
        raise ValueError(f"Unknown activation function: {name}. "
                         f"Available options: {list(_activation_map.keys())}")

    activation_cls = _activation_map[name]

    if name == "swiglu" and not return_class:
        if "in_channels" not in kwargs:
            raise ValueError("SWiGLU requires `in_channels` and `out_channels` as an argument.")
        return activation_cls(**kwargs)

    return activation_cls if return_class else activation_cls()


if __name__ == "__main__":
    import torch
    x = torch.randn(2, 64)
    act = get_activation("SWiGLU", in_channels=64, out_channels=128)
    print(act(x).shape)
