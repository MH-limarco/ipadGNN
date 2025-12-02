import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_flops(model, dataset):
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dataset.x, dataset.edge_index)

    FLOPS = flop_counter.get_total_flops()
    if FLOPS > 1e9:
        FLOPS = f"{FLOPS / 1e9:.2f}G"
    elif FLOPS > 1e6:
        FLOPS = f"{FLOPS / 1e6:.2f}M"
    elif FLOPS > 1e3:
        FLOPS = f"{FLOPS / 1e3:.2f}K"
    return FLOPS

def _get_runtime_vram(model, dataset, epochs=100):
    from time import perf_counter
    times = []
    torch.clear_autocast_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for e in range(epochs):
        torch.cuda.synchronize()
        start_epoch = perf_counter()
        model(dataset.x, dataset.edge_index)
        torch.cuda.synchronize()
        end_epoch = perf_counter()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)

    max_vram = torch.cuda.max_memory_reserved(DEVICE) / 1024 ** 2
    return (sum(times[int(epochs * 0.2):]) / (epochs - int(epochs * 0.2))) * 1000, max_vram

def check_float_and_time(model, dataset):
    model.eval()
    model = model.to(DEVICE)
    dataset = dataset.to(DEVICE)
    _float = _get_flops(model, dataset)
    _time, _max_vram = _get_runtime_vram(model, dataset)
    print(f"float: {_float} RunTime: {_time:.3f}ms VRAM: {_max_vram:.2f}MB")
    return _float, _time, _max_vram


from limelight.imputer.apcf_imputer import APCFIFilling
class LearnFPFilling:
    def __init__(self, dim, **kwargs):
        super(LearnFPFilling, self).__init__()#(alpha=alpha, beta=beta, star=star, batch=batch, debug=debug, **kwargs)
        from torch import nn
        self.dim = dim
        self.diffusion_cov = nn.Linear(dim, dim, bias=False).to(DEVICE)

    def reset_parameters(self):
        self.diffusion_cov.reset_parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, dataset):
        #from limelight.api.imputer import filling_data
        #dataset = filling_data(dataset.clone(), "fp", _print=False).detach()
        dataset = dataset.detach()
        dataset.x = self.diffusion_cov(dataset.x)
        return dataset