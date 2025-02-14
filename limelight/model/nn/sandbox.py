
from tqdm import tqdm
import torch_geometric

from limelight.model.nn import *
from limelight.utils.timer import Timer

class BaseSandBox:
    def __init__(self, module, in_dim, num_iterations, warm_iter, device):
        self.timer = Timer()
        self.device = "cuda" if torch.cuda.is_available() and device.lower() == "cuda" else "cpu"

        self.module = module.to(self.device)
        self.in_dim = in_dim

        self.num_iter = num_iterations
        self.warmup_iter = warm_iter if warm_iter is not None else self.num_iter // 10

        print(f"\nWarm-up iterations: {self.warmup_iter}")

    def run(self):
        raise NotImplementedError

class LayerSandBoxTest(BaseSandBox):
    def __init__(self, layer, in_dim, num_iterations=1000, warm_iter=None, device="cuda"):
        super(LayerSandBoxTest, self).__init__(layer, in_dim, num_iterations, warm_iter, device)

    def run(self):
        self.timer.reset()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        x = torch.rand(128, self.in_dim).to(self.device)
        for i in tqdm(range(self.num_iter)):
            if i == self.warmup_iter:
                self.timer.start()
            _ = self.module(x)

        self.timer.stop()
        total_time = (self.num_iter - self.warmup_iter)/ self.timer.get_time()
        print(f"Avg Speed: {total_time:.2f} iter/s")

class ModuleSandBoxTest(BaseSandBox):
    def __init__(self, module, in_dim, num_iterations=1000, warm_iter=None, device="cuda"):
        super(ModuleSandBoxTest, self).__init__(module, in_dim, num_iterations, warm_iter, device)

    def run(self):
        self.timer.reset()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        x = torch_geometric.data.Data(x=torch.rand(128, self.in_dim),
                                      edge_index=torch.randint(0, 128, (2, 32))).to(self.device)
        for i in tqdm(range(self.num_iter)):
            if i == self.warmup_iter:
                self.timer.start()
            _ = self.module(x)

        self.timer.stop()
        total_time = (self.num_iter - self.warmup_iter)/ self.timer.get_time()
        print(f"Avg Speed: {total_time:.2f} iter/s")

if __name__ == "__main__":
    test = LayerSandBoxTest(get_activation("selu", in_channels=128, out_channels=128), 128)
    test.run()


    model_map = {"in_channels": 128, "hidden_channels": 256,
                 "out_channels": 1, "num_layers": 3,
                 "conv_type": "Linear", "heads": 4,
                 "K": 1,
                 "norm_type": "layer", "act_type": "swiglu"}

    test = ModuleSandBoxTest(NormModule(model_map), 128)
    test.run()