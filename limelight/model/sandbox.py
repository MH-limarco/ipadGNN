

import torch
import torch_geometric
from tqdm import tqdm

from limelight.model import *
from limelight.utils import Timer
from limelight.api import parse_config

config = parse_config()
NUMBER_NODE = config.number_node
NUMBER_EDGE = config.number_edge
NUMBER_ITERATIONS = config.number_iterations

class BaseSandBox:
    def __init__(self, module: torch.nn.Module, in_dim, num_iterations=NUMBER_ITERATIONS, warm_iter=None, device="cuda"):
        self.timer = Timer()
        self.device = "cuda" if torch.cuda.is_available() and device.lower() == "cuda" else "cpu"

        self.module = module.to(self.device)
        self.in_dim = in_dim

        self.num_iter = num_iterations
        self.warmup_iter = warm_iter if warm_iter is not None else round(self.num_iter * 0.1)

        print(f"\nWarm-up iterations: {self.warmup_iter}")

    def build_input(self):
        raise NotImplementedError

    def run(self):
        x = self.build_input()
        self.timer.reset()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        for i in tqdm(range(self.num_iter)):
            if i == self.warmup_iter:
                self.timer.start()
            _ = self.module(x)

        self.timer.stop()
        total_time = (self.num_iter - self.warmup_iter) / self.timer.get_time()
        print(f"Avg Speed: {total_time:.2f} iter/s")

class LayerSandBoxTest(BaseSandBox):
    def build_input(self):
        return torch.rand(NUMBER_NODE, self.in_dim).to(self.device)

class ModuleSandBoxTest(BaseSandBox):
    def build_input(self):
        return torch_geometric.data.Data(x=torch.rand(NUMBER_NODE, self.in_dim).to(self.device),
                                         edge_index=torch.randint(0, NUMBER_NODE, (2, NUMBER_EDGE)).to(self.device))

if __name__ == "__main__":
    test = LayerSandBoxTest(get_activation("swiglu", in_channels=128, out_channels=128), 128)
    test.run()


    model_map = {"in_channels": 128, "hidden_channels": 256,
                 "out_channels": 1, "num_layers": 3,
                 "conv_type": "Linear", #GCNConv, ChebConv, GATConv, SAGEConv, Linear
                 "heads": 4, "K": 1,
                 "norm_type": "layer", "act_type": "relu"}

    test = ModuleSandBoxTest(NormModule(model_map), 128)
    test.run()