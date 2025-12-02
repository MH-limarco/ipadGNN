### we need support tunedGNN and baselines
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric import EdgeIndex

from moudle.kd import ReliableProbModel
from engine.moudle import kd_loss

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

def _get_unwrapped_parameters(layer, dim=None):
    return [(param, dim if dim is not None else param.dim() - 1)
            for name, param in layer.named_parameters(recurse=False)]

def prune(model, dataset, prune_rate, pruner_type, imp_type, isomorphic, global_pruning=True, device=None):
    device = device if device is not None else DEVICE
    num_class = dataset.num_classes

    ignored_layers = []
    unwrapped_parameters = []
    num_heads = {}
    for m in model.modules():
        if hasattr(m, "prunable"):
            if not m.prunable:
                raise ValueError(f"Layer {m.__class__.__name__} is not prunable.")

            if isinstance(m.prunable, list):
                for _p in m.prunable:
                    if hasattr(m, _p):
                        if type(getattr(m, _p)) in [nn.ModuleList]:
                            print(m, type(getattr(m, _p)), getattr(m, _p))
                            for idx, sub_m in enumerate(getattr(m, _p)):
                                num_heads[sub_m] = m.heads

                        else:
                            num_heads[getattr(m, _p)] = m.heads

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
                         num_heads=num_heads,
                         ignored_layers=ignored_layers,
                         unwrapped_parameters=unwrapped_parameters,
                         )
    pruner.step()
    model._config_updated = False
    print(model)
    return model


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == label).sum().item()
    acc = correct / len(label)
    return acc


class Trainer:
    def __init__(self, model, epochs, early_stop=None, label_smoothing=None,
                 optim='Adam', scheduler=False, warmup=None, T_0=None, T_mult=1, eta_min=1e-5,
                 amp=True, device=None, gpu_idx=None, _print=True):
        self._print = _print

        device = device if device else DEVICE
        gpu_idx = gpu_idx if gpu_idx is not None else 0
        self.device = f"{device}:{gpu_idx}" if device == "cuda" else device
        self.model = model.to(self.device)
        self.amp = amp

        self.epochs = epochs
        self.optim = optim
        self._scheduler = scheduler
        self.early_stop = early_stop

        self.label_smoothing = label_smoothing if label_smoothing is not None else 0.0
        self.scheduler_kwargs = {"warmup": warmup if warmup is not None else 0,
                                 "T_0": T_0, "T_mult": T_mult, "eta_min": eta_min}

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
        self._stop_epoch = 0

    def _fit(self, data, lr, weight_decay):
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing) if data.num_classes > 2 else nn.BCEWithLogitsLoss()
        self._reset_fit_memory()
        if hasattr(self.model, "_CuGraph") and self.model._CuGraph and isinstance(data.edge_index, torch.Tensor):
            data.edge_index = EdgeIndex(data.edge_index)

        torch.cuda.reset_peak_memory_stats(self.device)
        optimizer = self._get_optimizer(lr, weight_decay)
        scaler = torch.amp.GradScaler(device=self.device, enabled=self.amp)
        scheduler = self._get_scheduler(optimizer)

        self.qbar = tqdm(range(1, self.epochs + 1), desc="Training", unit="epoch") \
            if self._print else range(1, self.epochs + 1)

        for epoch in self.qbar:
            if hasattr(self.model, "start_global"):
                self.model.start_global(epoch)

            loss, metrics, _stop = self.train_step(epoch, data, optimizer, scaler, scheduler)
            if self._print:
                self.qbar.set_postfix(loss=round(loss[0], 3), acc=round(metrics[-1], 3),
                                      best_acc=self._best_metrics_test)
            if _stop:
                break
        if locals().get("epoch") is not None:
            self._stop_epoch = epoch
        max_vram = torch.cuda.max_memory_reserved(self.device) / 1024 ** 2
        if self._print:
            print(f"Max VRAM: {max_vram:.2f}MB")

    def _get_optimizer(self, lr, weight_decay):
        optimizer = getattr(torch.optim, self.optim)(self.model.parameters(),
                                                     lr=lr, weight_decay=weight_decay)
        return optimizer

    def _get_scheduler(self, optimizer):
        warmup = self.scheduler_kwargs["warmup"]
        if isinstance(warmup, float):
            if 0.0 <= warmup < 1.0:
                warmup_epochs = round(warmup * self.epochs)
            else:
                raise ValueError(f"Invalid warmup value: {warmup}. It should be a float between 0.0 and 1.0.")

        elif isinstance(warmup, int):
            if 0 <= warmup < self.epochs:
                warmup_epochs = warmup
            else:
                raise ValueError(f"Invalid warmup value: {warmup}. It should be an int between 0 and {self.epochs}.")
        else:
            raise ValueError(f"Invalid warmup value: {warmup}. It should be an int or a float.")

        warmup_scheduler = LinearLR(optimizer,
                                    total_iters=warmup_epochs)

        if self._scheduler:
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
        else:
            scheduler = warmup_scheduler

        return scheduler

    def train_step(self, epoch, data, optimizer, scaler, scheduler):
        loss, metrics = self._train_step(data, optimizer, scaler, scheduler)
        _stop = self._early_stop(metrics, epoch)
        self._last_state_dict = self.model.state_dict()
        return loss, metrics, _stop

    def _train_step(self, data, optimizer, scaler, scheduler):
        train_loss = self.train(data, optimizer, scaler, scheduler)
        val_loss, metrics_train, metrics_val, metrics_test, _ = self.evaluate(data, loss_flag=True)
        return [train_loss, val_loss], [metrics_train, metrics_val, metrics_test]

    def train(self, data, optimizer, scaler, scheduler):
        raise NotImplementedError("Please implement the train method in your subclass.")

    def test(self, data):
        self.model = self.model.to(self.device)
        if hasattr(self.model, "_CuGraph") and self.model._CuGraph and isinstance(data.edge_index, torch.Tensor):
            data.edge_index = EdgeIndex(data.edge_index)

        metrics_train, metrics_val, metrics_test, _ = self.evaluate(data.to(self.device), loss_flag=False)
        return metrics_train, metrics_val, metrics_test

    def evaluate(self, data, loss_flag):
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.float16):  #
            out = self.model(data.x, data.edge_index)
            if loss_flag:
                loss = self.loss_func(out, data, data.val_mask)

        metrics_train = self.metrics(out, data, data.train_mask)
        metrics_val = self.metrics(out, data, data.val_mask)
        metrics_test = self.metrics(out, data, data.test_mask)
        if loss_flag:
            return loss.detach().cpu().item(), metrics_train, metrics_val, metrics_test, out
        else:
            return metrics_train, metrics_val, metrics_test, out

    def _backward(self, loss, optimizer, scaler, scheduler):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
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
            self.model.eval()
            self.best_state_dict = deepcopy(self.model.state_dict())
            self._patience = 0

        else:
            self._patience += 1
            if self._patience == self._max_patience:
                print(f"Early stopping at epoch {epoch}")
                return True

        return False

    def metrics(self, out, data, mask):
        raise NotImplementedError("Please implement the metrics method in your subclass.")


class TrainerCLS(Trainer):
    def __init__(self, args, model,
                 epochs, lr, weight_decay=0.0001, early_stop=None, scheduler=False,
                 optim='Adam', warmup=None, T_0=None, T_mult=2, eta_min=1e-5,
                 label_smoothing=None, 
                 amp=False, device=None, gpu_idx=0, _print=True, **kwargs):
        super().__init__(model=model, epochs=epochs, scheduler=scheduler, early_stop=early_stop, optim=optim, 
                         warmup=warmup, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
                         label_smoothing=label_smoothing, amp=amp, device=device, gpu_idx=gpu_idx, _print=_print)

        self.training_kwargs = {"lr": lr, "weight_decay": weight_decay}

    def fit(self, data):
        data = data.to(self.device)
        if hasattr(self.model, "_CuGraph") and self.model._CuGraph:
            data.edge_index = EdgeIndex(data.edge_index)

        self._fit(data, **self.training_kwargs)
        self.model.load_state_dict(self.best_state_dict)

        if self._print or self._print == 0:
            print(f"{self.__class__.__name__}-Best epoch: {self._best_epoch}, "
                  f"Best val acc: {self._best_metrics_val:.4f}, "
                  f"Best test acc: {self._best_metrics_test:.4f}")

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
        return pred

    def loss_func(self, out, data, mask):
        out = out.float() if self.amp else out
        loss = self.criterion_cls(out[mask], data.y[mask])
        return loss

    def train(self, data, optimizer, scaler, scheduler):
        self.model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.float16):
            out = self.model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.train_mask)

        self._backward(loss, optimizer, scaler, scheduler)
        return loss.detach().cpu().item()

    @staticmethod
    def metrics(out, data, mask):
        metrics = accuracy(out[mask], data.y[mask])
        return metrics


class TrainerPrune(TrainerCLS):
    def __init__(self, args, model,
                 epochs, lr=0.001, weight_decay=0.0001, early_stop=None, scheduler=False,
                 mpp_step=1, prune_rate=0.5,
                 prune_type="base", imp_type="group", isomorphic=True, global_pruning=True,
                 optim='Adam', warmup=None, T_0=None, T_mult=2, eta_min=1e-5,
                 amp=False, device=None, gpu_idx=0, _print=True, **kwargs):

        super().__init__(args=args, model=model, epochs=round(epochs//mpp_step), lr=lr, weight_decay=weight_decay,
                         early_stop=early_stop, scheduler=scheduler, optim=optim, warmup=warmup, gpu_idx=gpu_idx,
                         T_0=T_0, T_mult=T_mult, eta_min=eta_min, amp=amp, device=device, _print=_print)
        self.mpp_step = max(mpp_step, 1)

        epoch_prune_rate = (1.0 - (1.0 - prune_rate) ** (1.0 / self.mpp_step))
        self.pruning_kwargs = {"prune_rate": epoch_prune_rate, "pruner_type": prune_type, "imp_type": imp_type,
                               "isomorphic": isomorphic, "global_pruning": global_pruning}

    def fit(self, data):
        data = data.to(self.device)

        for i in range(self.mpp_step):
            self._fit(data, **self.training_kwargs)
            self.model.load_state_dict(self.best_state_dict)
            self.model = prune(self.model, data, **self.pruning_kwargs)

        if self._print or self._print == 0:
            print(f"{self.__class__.__name__}-Best epoch: {self._best_epoch}, "
                  f"Best val acc: {self._best_metrics_val:.4f}, "
                  f"Best test acc: {self._best_metrics_test:.4f}")


class TrainerRKD(TrainerCLS):
    def __init__(self, args, teacher, student,
                 epochs, momentum, updata=50, lr=0.001, weight_decay=0.0001, early_stop=None, scheduler=False,
                 optim='Adam', warmup=None, T_0=None, T_mult=2, eta_min=1e-5, lamb=0.4, feat_loss=None, temperature=1.0,
                 alpha=0.0, label_smoothing=None, amp=False, device=None, gpu_idx=0, _print=True, krd=True, **kwargs):
        super().__init__(args=args, model=student, epochs=epochs, lr=lr, weight_decay=weight_decay,
                         early_stop=early_stop, scheduler=scheduler, optim=optim, warmup=warmup,
                         T_0=T_0, T_mult=T_mult, eta_min=eta_min, amp=amp, device=device,
                         label_smoothing=label_smoothing, gpu_idx=gpu_idx, _print=_print)
        self.krd = krd
        self.lamb = lamb
        self.temperature = temperature
        self.feat_loss = getattr(kd_loss, f"{feat_loss.upper()}Loss")(temperature=temperature) if feat_loss is not None else None
        self.alpha = alpha
        self.momentum = momentum
        self.teacher = teacher.to(self.device)

        self.kd_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.sampler = ReliableProbModel(self.device, momentum)
        self._last_out = None

        self.update_epoch = updata

    def _initialization(self, teacher, data):
        self.sampler.initialization(teacher, data)
        if self.feat_loss is not None and hasattr(teacher, "forward_with_hidden"):
            self._logit_teacher, self._hidden_teacher = teacher.forward_with_hidden(data.x, data.edge_index)
        else:
            self._logit_teacher = teacher(data.x, data.edge_index)

        self._logit_teacher = self._logit_teacher.detach()

    def _reset_fit_memory(self):
        super()._reset_fit_memory()
        self._last_out = None

    def fit(self, data):
        self._initialization(self.teacher, data.to(self.device))
        super().fit(data)

    def criterion_kd(self, out, data):
        if self.krd:
            check_list = add_self_loops(data.check_list)[0] if hasattr(data, "check_list") else data.edge_index
            _loss = self.kd_loss(F.log_softmax(out[check_list[0]], dim=1),
                                 F.log_softmax(self._logit_teacher[check_list[1]], dim=1))
        else:
            _loss = self.kd_loss(F.log_softmax(out, dim=1),
                                 F.log_softmax(self._logit_teacher, dim=1))

        return _loss


    def loss_func(self, out, data, mask, hidden=None):
        out = out.float() if self.amp else out
        loss_cls = self.criterion_cls(out[mask], data.y[mask])
        loss_kd = self.criterion_kd(out, data)
        if self.feat_loss is not None:
            loss_feat = self.criterion_feat(hidden)
            loss = loss_cls + self.lamb * loss_kd + self.alpha * loss_feat
            if self._print:
                self.qbar.set_description(
                    f"loss_feat={loss_feat.item():.3f}, loss_kd={loss_kd.item():.3f}, power: {self.sampler.power:.2f}")

        else:

            loss = (1 - self.lamb) * loss_cls +  self.lamb * loss_kd
            if self._print:
                self.qbar.set_description(
                    f"loss_kd={loss_kd.item():.3f}, power: {self.sampler.power:.2f}")

        return loss

    def sampling(self, data):
        reliable_prob = self.sampler.predict_prob()
        if reliable_prob is None:
            return data

        sampling_mask = torch.bernoulli(reliable_prob[data.edge_index[1]]).bool()
        data.check_list = data.edge_index[:, sampling_mask]
        return data

    def _updata_sampler(self, logits_student):
        self.sampler.updata(logits_student)

    def train_step(self, epoch, data, optimizer, scaler, scheduler):
        data.edge_index = remove_self_loops(data.edge_index)[0]
        if self.sampler.ready:
            data = self.sampling(data)

        if epoch >= self.update_epoch and epoch > 1:
            self._updata_sampler(self._last_out)

        return super().train_step(epoch, data, optimizer, scaler, scheduler)

    def train(self, data, optimizer, scaler, scheduler):
        self.model.train()
        optimizer.zero_grad()
        hidden = None
        with torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.float16):
            if self.feat_loss is not None and hasattr(self.model, "forward_with_hidden"):
                out, hidden = self.model.forward_with_hidden(data.x, data.edge_index)
            else:
                out = self.model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.train_mask, hidden=hidden)

        self._backward(loss, optimizer, scaler, scheduler)
        return loss.detach().cpu().item()

    def evaluate(self, data, loss_flag):
        self.model.eval()
        hidden = None
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.float16):  #
            if self.feat_loss is not None and hasattr(self.model, "forward_with_hidden"):
                out, hidden = self.model.forward_with_hidden(data.x, data.edge_index)
            else:
                out = self.model(data.x, data.edge_index)
            if loss_flag:
                loss = self.loss_func(out, data, data.train_mask, hidden=hidden)

        metrics_train = self.metrics(out, data, data.train_mask)
        metrics_val = self.metrics(out, data, data.val_mask)
        metrics_test = self.metrics(out, data, data.test_mask)
        self._last_out = out
        if loss_flag:
            return loss.detach().cpu().item(), metrics_train, metrics_val, metrics_test, out
        else:
            return metrics_train, metrics_val, metrics_test, out

    def criterion_feat(self, hidden):
        loss = 0.0
        for idx, (student_hid, teacher_hid) in enumerate(zip(hidden, self._hidden_teacher)):
            loss += self.feat_loss(student_hid, teacher_hid)

        #for idx in [0, -1]:
        #   loss += self.feat_loss(hidden[idx], self._hidden_teacher[idx])
        return loss


def read_config(data_name, dir, config_root="./best_config"):
    import yaml, os
    with open(os.path.join(config_root, dir, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def read_best_config(data_name, model_type, prune_rate=0.0):
    gnn_dir = "gnns/prune_cfg" if prune_rate > 0 else "gnns/best_cfg"
    gnn_config = read_config(data_name, gnn_dir)

    config = {"seed": gnn_config.get("seed", 42)}

    if prune_rate > 0:
        _gnn_config = gnn_config[model_type.lower()]
        config.update(_gnn_config[prune_rate])
        _gnn_config.pop(0.5)
        _gnn_config.pop(0.6)
        _gnn_config.pop(0.7)
        config.update(_gnn_config)
        config["prune_epochs"] = int(150 * config["prune_step"])
        config = {k: float(v) if "lr" in k else v for k, v in config.items()}
        config = {k: float(v) if "weight_decay" in k else v for k, v in config.items()}
        config = {k: int(v) if "epochs" in k else v for k, v in config.items()}

    else:
        config.update(gnn_config[model_type.lower()])
        config["epochs"] = int(config["epochs"])
        config["lr"] = float(config["lr"])
        config["weight_decay"] = float(config["weight_decay"])
    return config

def P(model):
    mp_model = mirror_projection(model, target="mlp", mirror_attr=["gnn"])

    prune_trainer = TrainerPrune(args=config, model=mp_model,
                                 epochs=600, lr=0.001,
                                 weight_decay=5e-4,
                                 prune_rate=0.6, mpp_step=1,
                                 prune_type="group", imp_type="group",
                                 isomorphic=False,
                                 scheduler=False, warmup=0.1, T_0=1,
                                 early_stop=-1, amp=False, gpu_idx=0, _print=True)

    prune_trainer.fit(dataset)

    model = mirror_projection(prune_trainer.model, target="gnn", mirror_attr=["gnn"])
    return model


if __name__ == "__main__":
    from limelight.api.dataset import load_dataset
    from limelight.utils import fix_seed
    from engine.moudle.model import mirror_projection
    from copy import deepcopy
    from engine.save_log.baselines.polynormer import Polynormer

    dataname = "photo"
    prune_rate = 0.6

    kdmlp = True
    best = False

    split = "fixed" if dataname not in ["cora", "citeseer", "pubmed"] else "class"

    config = {"seed": 42, "hidden": 64, "local_layers": 7, "global_layers": 2, "heads": 8,
              "dropout": 0.7, "in_dropout": 0.2}

    fix_seed(config["seed"])
    dataset, _ = load_dataset(dataname, split, missing_rate=0.0, label_num_per_class=20)
    #dataset = filling_data(dataset, "fp")


    model = Polynormer(in_channels=dataset.x.size(1), hidden_channels=config["hidden"], out_channels=dataset.num_classes,
                       local_layers=config["local_layers"], global_layers=config["global_layers"],
                       dropout=config["dropout"], in_dropout=config["in_dropout"],
                       heads=config["heads"])

    model = P(model)
    print(model)

    gnn_trainer = TrainerCLS(args=None, model=model,
                             epochs=2000, lr=0.001, weight_decay=5e-5,
                             early_stop=-1, optim="Adam", gpu_idx=0,
                             T_0=1, scheduler=False, warmup=0.00,
                             amp=False, device=None, _print=True)

    gnn_trainer.fit(dataset)

