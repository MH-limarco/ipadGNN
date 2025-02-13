
from tqdm import tqdm
import torch

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

class SandBoxTest(BaseSandBox):
    def __init__(self, module, in_dim, num_iterations=100, warm_iter=None, device="cuda"):
        super(SandBoxTest, self).__init__(module, in_dim, num_iterations, warm_iter, device)

    def run(self):
        self.timer.reset()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        x = torch.rand(128, self.in_dim).to(self.device)
        for i in tqdm(range(self.num_iter)):
            if i == self.warmup_iter:
                self.timer.start()
            _ = self.module(x)

        self.timer.stop()
        total_time = self.num_iter / self.timer.get_time()
        print(f"Avg Speed: {total_time:.2f} iter/s")

if __name__ == "__main__":
    test = SandBoxTest(get_activation("selu", in_channels=128, out_channels=128), 128)
    test.run()