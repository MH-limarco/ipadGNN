# prune, cls, rkd

from copy import deepcopy
import gc

from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric import EdgeIndex

from new_rkd import ReliableProbModel
from new_model import MainModule, mirror_projection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

imp_dict = {"group": tp.importance.GroupMagnitudeImportance,
            "lamp": tp.importance.LAMPImportance,
            "fpgm": tp.importance.FPGMImportance,
            "random": tp.importance.RandomImportance
            }

pruner_dict = {"base": tp.pruner.BasePruner,
               "bnscale": tp.pruner.BNScalePruner,
               "group": tp.pruner.GroupNormPruner
               }


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


def _get_unwrapped_parameters(layer, dim=None):
    return [(param, dim if dim is not None else param.dim() - 1)
            for name, param in layer.named_parameters(recurse=False)]

def prune(model, dataset, prune_rate, pruner_type, imp_type, isomorphic, global_pruning=True, device=None):
    device = device if device is not None else DEVICE
    num_class = dataset.num_classes

    ignored_layers = []
    unwrapped_parameters = []
    for m in model.modules():
        if hasattr(m, "prunable"):
            if not m.prunable:
                raise ValueError(f"Layer {m.__class__.__name__} is not prunable.")

        if hasattr(m, "out_channels"):
            out_features = m.out_channels
        elif hasattr(m, "out_features"):
            out_features = m.out_features
        else:
            continue

        if out_features == num_class:
            ignored_layers.append(m)

        if not isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, nn.Linear)):
            if hasattr(m, "unwrapped_parameters"):
                unwrapped_parameters.extend(m.unwrapped_parameters)
            else:
                unwrapped_parameters.extend(_get_unwrapped_parameters(m))

    imp_func = imp_dict[imp_type]()
    pruner_func = pruner_dict[pruner_type]

    model.eval()
    model = model.to(device)
    dataset = dataset.to(device)
    pruner = pruner_func(model,
                         example_inputs=[torch.rand_like(dataset.x), dataset.edge_index],
                         importance=imp_func,
                         isomorphic=isomorphic,
                         global_pruning=global_pruning,
                         pruning_ratio=prune_rate,
                         ignored_layers=ignored_layers,
                         unwrapped_parameters=unwrapped_parameters,
                         )
    pruner.step()
    return model


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == label).sum().item()
    acc = correct / len(label)
    return acc


class Trainer:
    def __init__(self, model, epochs, early_stop=None,
                 optim='Adam', warnup=10, T_0=None, T_mult=1, eta_min=1e-5,
                 amp=True, device=None, gpu_idx=0, _print=True):
        device = device if device else DEVICE
        self.device = f"{device}:{gpu_idx}" if device == "cuda" else device
        self.model = model.to(self.device)
        self.early_stop = early_stop
        self.epochs = epochs
        self._print = _print
        self.optim = optim
        self.amp = amp

        self.scheduler_kwargs = {"warmup_epochs": warnup, "T_0": T_0, "T_mult": T_mult, "eta_min": eta_min}


    def fit(self, data):
        raise NotImplementedError("Please implement the fit method in your subclass.")

    def predict(self, data):
        raise NotImplementedError("Please implement the predict method in your subclass.")

    def loss_func(self, out, data, mask):
        raise NotImplementedError("Please implement the loss_func method in your subclass.")

    def _reset_fit_memory(self):
        self.best_state_dict = None
        self._max_patience = self.early_stop if self.early_stop is not None else -1
        self._best_metrics_val = 0.0
        self._best_metrics_test = 0.0
        self._best_epoch = 0
        self._patience = 0

    def _fit(self, data, lr, weight_decay, scheduler):
        self._reset_fit_memory()
        if hasattr(self.model, "_CuGraph") and self.model._CuGraph:
            data.edge_index = EdgeIndex(data.edge_index)

        torch.cuda.reset_peak_memory_stats(self.device)
        optimizer = self._get_optimizer(lr, weight_decay)
        scaler = torch.amp.GradScaler(device=self.device, enabled=self.amp)
        scheduler = self._get_scheduler(optimizer) if scheduler else None

        self.qbar = tqdm(range(1, self.epochs + 1), desc="Training", unit="epoch") \
                         if self._print else range(1, self.epochs + 1)
        for epoch in self.qbar:
            loss, metrics, _stop = self.train_step(epoch, data, optimizer, scaler, scheduler)
            if self._print:
                self.qbar.set_postfix(loss=round(loss[0], 3), acc=round(metrics[-1], 3), best_acc=self._best_metrics_test)
            if _stop:
                break

        max_vram = torch.cuda.max_memory_reserved(self.device) / 1024 ** 2
        if self._print:
            print(f"Max VRAM: {max_vram:.2f}MB")
        return self.best_state_dict, self._best_epoch, self._best_metrics_val, self._best_metrics_test

    def _get_optimizer(self, lr, weight_decay):
        optimizer = getattr(torch.optim, self.optim)(self.model.parameters(),
                                                     lr=lr, weight_decay=weight_decay)
        return optimizer

    def _get_scheduler(self, optimizer):
        warmup_epochs = self.scheduler_kwargs["warmup_epochs"]
        if 1.0 > warmup_epochs > 0:
            warmup_epochs = int(self.epochs * warmup_epochs)

        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=1.0 / max(warmup_epochs - 1, 1.0),
                                    end_factor=1.0,
                                    total_iters=warmup_epochs)

        T_0 = self.scheduler_kwargs.get("T_0", None)
        T_0 = T_0 if T_0 is not None else int(self.epochs // 5 if self.epochs > 100 else self.epochs // 3)
        T_mult = self.scheduler_kwargs["T_mult"]
        eta_min = self.scheduler_kwargs["eta_min"]
        cosine_scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                       T_0=T_0,
                                                       T_mult=T_mult,
                                                       eta_min=eta_min)

        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_epochs]  # 第 warmup_epochs 之後切到 cosine_scheduler
                                 )
        return scheduler
    
    def train_step(self, epoch, data, optimizer, scaler, scheduler):
        loss, metrics = self._train_step(data, optimizer, scaler, scheduler)
        _stop = self._early_stop(metrics, epoch)
        self._last_state_dict = self.model.state_dict()
        return loss, metrics, _stop

    def _train_step(self, data, optimizer, scaler, scheduler):
        train_loss = self.train(data, optimizer, scaler, scheduler)
        val_loss, metrics_train, metrics_val, metrics_test, _ = self.evaluate(data)
        return [train_loss, val_loss], [metrics_train, metrics_val, metrics_test]

    def train(self, data, optimizer, scaler, scheduler):
        raise NotImplementedError("Please implement the train method in your subclass.")

    def evaluate(self, data):
        raise NotImplementedError("Please implement the evaluate method in your subclass.")


    def _backward(self, loss, optimizer, scaler, scheduler):
        _loss = loss.detach().cpu().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if scaler is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    @staticmethod
    def _metrics(out, data, mask):
        raise NotImplementedError("Please implement the _metrics method in your subclass.")

    def _early_stop(self, metrics, epoch):
        metrics_train, metrics_val, metrics_test = metrics
        if metrics_val > self._best_metrics_val:
            self._best_epoch = epoch
            self._best_metrics_val = metrics_val
            self._best_metrics_test = metrics_test
            self.best_state_dict = self.model.state_dict()
            self._patience = 0

        else:
            self._patience += 1
            if self._patience == self._max_patience:
                print(f"Early stopping at epoch {epoch}")
                return True
        return False


class TrainerCLS(Trainer):
    def __init__(self, args, model,
                 epochs, lr, weight_decay=0.0001, early_stop=None, scheduler=False,
                 optim='Adam', warnup=10, T_0=50, T_mult=1, eta_min=1e-5,
                 amp=False, device=None, gpu_idx=0, _print=True, **kwargs):
        super().__init__(model=model, epochs=epochs, early_stop=early_stop, optim=optim, warnup=warnup, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
                         amp=amp, device=device, gpu_idx=gpu_idx, _print=_print)

        self.training_kwargs = {"lr": lr, "weight_decay": weight_decay,
                                "scheduler": scheduler}

    def fit(self, data):
        self.criterion = nn.CrossEntropyLoss() if data.num_classes > 2 else nn.BCEWithLogitsLoss()

        data = data.to(self.device)
        if hasattr(self.model, "_CuGraph") and self.model._CuGraph:
            data.edge_index = EdgeIndex(data.edge_index)

        state_dict, best_epoch, metrics_val, metrics_test = self._fit(data, **self.training_kwargs)
        self.model.load_state_dict(self.best_state_dict)


        if self._print or self._print == 0:
            print(f"Cls-Best epoch: {best_epoch}, "
                  f"Best val acc: {metrics_val:.4f}, "
                  f"Best test acc: {metrics_test:.4f}")

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            return pred

    def loss_func(self, out, data, mask):
        out = out.float() if self.amp else out
        loss = self.criterion(out[mask], data.y[mask])
        return loss

    def train(self, data, optimizer, scaler, scheduler):
        self.model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.bfloat16):
            out = self.model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.train_mask)

        self._backward(loss, optimizer, scaler, scheduler)
        return loss.detach().cpu().item()

    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.bfloat16 ): #
            out = self.model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.val_mask)

        metrics_train = self.metrics(out, data, data.train_mask)
        metrics_val = self.metrics(out, data, data.val_mask)
        metrics_test = self.metrics(out, data, data.test_mask)
        return loss.detach().cpu().item(), metrics_train, metrics_val, metrics_test, out

    @staticmethod
    def metrics(out, data, mask):
        metrics = accuracy(out[mask], data.y[mask])
        return metrics


class TrainerPrune(TrainerCLS):
    def __init__(self, args, model,
                 epochs, lr=0.001, weight_decay=0.0001, early_stop=None, scheduler=False,
                 mpp_step=5, prune_rate=0.5,
                 prune_type="base", imp_type="group", isomorphic=True, global_pruning=True,
                 optim='Adam', warnup=10, T_0=None, T_mult=2, eta_min=1e-5,
                 amp=False, device=None, gpu_idx=0, _print=False, **kwargs):

        super().__init__(args=args, model=model, epochs=round(epochs//mpp_step), lr=lr, weight_decay=weight_decay,
                         early_stop=early_stop, scheduler=scheduler, optim=optim, warnup=warnup, gpu_idx=gpu_idx,
                         T_0=T_0, T_mult=T_mult, eta_min=eta_min, amp=amp, device=device, _print=_print)
        self.mpp_step = max(mpp_step, 1)

        epoch_prune_rate = (1.0 - (1.0 - prune_rate) ** (1.0 / self.mpp_step))
        self.pruning_kwargs = {"prune_rate": epoch_prune_rate, "pruner_type": prune_type, "imp_type": imp_type,
                               "isomorphic": isomorphic, "global_pruning": global_pruning}

    def fit(self, data):
        self.criterion = nn.CrossEntropyLoss() if data.num_classes > 2 else nn.BCEWithLogitsLoss()

        data = data.to(self.device)

        for i in range(self.mpp_step):
            state_dict, best_epoch, metrics_val, metrics_test = self._fit(data, **self.training_kwargs)
            self.model.load_state_dict(self.best_state_dict)
            self.model = prune(self.model, data, **self.pruning_kwargs)

        if self._print or self._print == 0:
            print(f"Prune-Best epoch: {best_epoch}, "
                  f"Best val acc: {metrics_val:.4f}, "
                  f"Best test acc: {metrics_test:.4f}")

class TrainerRKD(TrainerCLS):
    def __init__(self, args, teacher, model,
                 epochs, momentum, updata=50, lr=0.001, weight_decay=0.0001, early_stop=None, scheduler=False,
                 optim='Adam', warnup=10, T_0=50, T_mult=1, eta_min=1e-5, lamb=0.4,
                 amp=False, device=None, gpu_idx=0, _print=True, **kwargs):
        super().__init__(args=args, model=model, epochs=epochs, lr=lr, weight_decay=weight_decay,
                         early_stop=early_stop, scheduler=scheduler, optim=optim, warnup=warnup,
                         T_0=T_0, T_mult=T_mult, eta_min=eta_min, amp=amp, device=device,
                         gpu_idx=gpu_idx, _print=_print)
        self.lamb = lamb
        self.momentum = momentum
        self.teacher = teacher.to(self.device)

        self.kd_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.sampler = ReliableProbModel(self.device, momentum)
        self._last_out = None

        self.updata_epoch = updata

    def initialization(self, teacher, data):
        self.sampler.initialization(teacher, data)
        self._logits_teacher = self.sampler.logits_teacher.to(self.device).detach()

    def _reset_fit_memory(self):
        super()._reset_fit_memory()
        self._last_out = None

    def updata_sampler(self, logits_student):
        self.sampler.updata(logits_student)

    def fit(self, data, smoothing=0.1):
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=smoothing) if data.num_classes > 2 else nn.BCEWithLogitsLoss()
        data = data.to(self.device)

        self.initialization(self.teacher, data)
        state_dict, best_epoch, metrics_val, metrics_test = self._fit(data, **self.training_kwargs)
        self.model.load_state_dict(self.best_state_dict)

        if self._print or self._print == 0:
            print(f"RKD-Best epoch: {best_epoch}, "
                  f"Best val acc: {metrics_val:.4f}, "
                  f"Best test acc: {metrics_test:.4f}")

    def criterion_kd(self, out, data):
        check_list = add_self_loops(data.check_list)[0] if hasattr(data, "check_list") else data.edge_index

        _loss = self.kd_loss(F.log_softmax(out[check_list[0]], dim=1),
                             F.log_softmax(self._logits_teacher[check_list[1]], dim=1))
        return _loss

    def loss_func(self, out, data, mask):
        out = out.float() if self.amp else out
        loss_cls = self.criterion_cls(out[mask], data.y[mask])
        loss_kd = self.criterion_kd(out, data)
        loss = self.lamb * loss_cls + (1 - self.lamb) * loss_kd
        if self._print:
            self.qbar.set_description(f"loss: {loss:.3f}, loss_kd={loss_kd.item():.3f}, power: {self.sampler.power:.2f}")
        return loss

    def train(self, data, optimizer, scaler, scheduler):
        self.model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.bfloat16):
            out = self.model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.train_mask)

        self._backward(loss, optimizer, scaler, scheduler)
        return loss.detach().cpu().item()

    def sampling(self, data):
        reliable_prob = self.sampler.predict_prob()
        if reliable_prob is None:
            return data

        sampling_mask = torch.bernoulli(reliable_prob[data.edge_index[1]]).bool()
        data.check_list = data.edge_index[:, sampling_mask]
        return data

    def train_step(self, epoch, data, optimizer, scaler, scheduler):
        data.edge_index = remove_self_loops(data.edge_index)[0]
        if self.sampler.ready:
            data = self.sampling(data)

        if epoch >= self.updata_epoch and epoch > 1:
            self.updata_sampler(self._last_out)

        loss, metrics, _stop = super().train_step(epoch, data, optimizer, scaler, scheduler)
        return loss, metrics, _stop

    def evaluate(self, data):
        val_loss, metrics_train, metrics_val, metrics_test, self._last_out = super().evaluate(data)
        return val_loss, metrics_train, metrics_val, metrics_test, self._last_out


if __name__ == "__main__":
    from limelight.api.dataset import load_dataset
    from limelight.api.imputer import filling_data
    from limelight.utils import fix_seed
    from time import perf_counter
    from torch_geometric.data import Data
    from new_baselines.polynormer import Polynormer
    from new_baselines.sgformer import SGFormer


    dataname = "cora"
    split = "fixed" if dataname not in ["cora", "citeseer", "pubmed"] else "class"

    gnn_type = "gcn"
    epochs = 500
    hidden = 256
    dropout = 0.7
    num_layers = 4
    norm = None ##"layer"
    res = False
    seed = 42 if dataname not in ["cora", "citeseer", "pubmed"] else 123

    pruner_step = 2
    pruner_type = "base"
    pruner_imp = "group"
    prune_rate = 0.5

    lr = 0.001 if prune_rate == 0.0 else 0.002630164
    weight_decay = 5e-4 if prune_rate == 0.0 else 0.000568157
    scheduler = False
    warnup: None

    pruner_lr = 0.00112937
    pruner_weight_decay = 4.80E-05
    prune_epochs = 150 * pruner_step
    prune_scheduler = True
    prune_warnup = 0.45

    fix_seed(seed)

    dataset, _ = load_dataset(dataname, split, missing_rate=0.0, label_num_per_class=20)

    start = perf_counter()

    fix_seed(seed)
    _gnn_type = f"{gnn_type}_mlp" if prune_rate > 0.0 else gnn_type

    gnn = MainModule(dataset.x.size(1), hidden, hidden, num_layers,
                       dropout=dropout, res=res, norm=norm, gnn_type="gcn",
                       gnn_cls=True)
    model = SGFormer(dataset.x.size(1), hidden, dataset.num_classes, num_layers=num_layers,
                     gnn=gnn, use_graph=True,
                     dropout=dropout)


    gnn_trainer = TrainerCLS(args=None, model=model, scheduler=False,
                               epochs=epochs, lr=lr, weight_decay=weight_decay,
                               early_stop=-1, optim="Adam", gpu_idx=0,
                               amp=False, device=None, _print=True)
    gnn_trainer.fit(dataset)
    train_time = perf_counter() - start
    print(f"{prune_rate} - Train Time: {train_time:.3f}s")

    fin_model = gnn_trainer.model
    del gnn_trainer
    check_float_and_time(fin_model, dataset)















