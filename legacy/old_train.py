

import gc
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
import torch_pruning as tp
from tqdm import tqdm


from new_model import MainModule, mirror_projection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNE_RATE = 0.95

def pruner_func(_model, data, prune_type="base", imp_type="group",
           prune_rate=PRUNE_RATE, isomorphic=True, device=None, _print=False):

    model = deepcopy(_model)
    model.eval()
    device = device if device is not None else DEVICE
    num_class = data.num_classes

    model = model.to(device)
    data = data.to(device)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.out_features == num_class:
                ignored_layers.append(m)

    unwrapped_parameters = []
    for name, param in model.named_parameters():
        _name = name.split('.')
        if len(_name) > 2 and (_name[0] not in ("norms", "res_layers") and (_name[2] =="bias")):
            unwrapped_parameters.append([param, 0])

    if imp_type.lower() == "group":
        imp = tp.importance.GroupMagnitudeImportance()

    elif imp_type.lower() == "lamp":
        imp = tp.importance.LAMPImportance()

    elif imp_type.lower() == "fpgm":
        imp = tp.importance.FPGMImportance()

    elif imp_type.lower() == "bnscale":
        imp = tp.importance.BNScaleImportance()

    else:
        raise ValueError("Unknown prune type. Please choose from 'group' or 'lamp'.")

    #BNScalePruner, BasePruner
    if prune_type.lower() == "base":
        pruner_api = tp.pruner.BasePruner
    elif prune_type.lower() == "bnscale":
        pruner_api = tp.pruner.BNScalePruner
    elif prune_type.lower() == "group":
        pruner_api = tp.pruner.GroupNormPruner
    else:
        raise ValueError("Unknown prune type. Please choose from 'base' or 'bnscale'.")

    pruner = pruner_api(model,
                          example_inputs=[torch.rand_like(data.x), data.edge_index],
                          importance=imp,
                          isomorphic=isomorphic,
                          global_pruning=True,
                          pruning_ratio=prune_rate,
                          ignored_layers=ignored_layers,
                          unwrapped_parameters=unwrapped_parameters,
                          )

    if _print:
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, [torch.rand_like(data.x), data.edge_index])
        print(
            f"Full:\nmacs:{base_macs:.2f} | params:{base_nparams}")  # print the number of MACs and parameters before pruning

        if _print:
            macs, nparams = tp.utils.count_ops_and_params(model, [torch.rand_like(data.x), data.edge_index])
            print(f"Prune:\nmacs:{macs:.2f} | params:{nparams}")  # print the number of MACs and parameters after pruning
    else:
        pruner.step()
    return model

def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == label).sum().item()
    acc = correct / len(label)
    return acc

class Trainer:
    def __init__(self,
                 optim='Adam', warnup=10, T_0=None, T_mult=1, eta_min=1e-5,
                 amp=True, device=None):
        self.amp = amp
        self.device = device if device else DEVICE
        self.optim = optim
        self.scheduler_kwargs = {"warmup_epochs": warnup, "T_0": T_0, "T_mult": T_mult, "eta_min": eta_min}


    def fit(self, data):
        raise NotImplementedError("Please implement the fit method in your subclass.")

    def predict(self, data):
        raise NotImplementedError("Please implement the predict method in your subclass.")

    def loss_func(self, out, data, mask):
        raise NotImplementedError("Please implement the loss_func method in your subclass.")

    def _train_loop(self, model, data, lr, weight_decay, epochs, scheduler, early_stop=None):
        self.best_state_dict = None
        self._max_patience = early_stop if early_stop is not None else -1
        self._best_metrics_val = 0.0
        self._best_metrics_test = 0.0
        self._best_epoch = 0
        self._patience = 0
        torch.cuda.reset_peak_memory_stats(self.device)

        optimizer = self._get_optimizer(model, lr, weight_decay)
        scaler = torch.amp.GradScaler(device=self.device, enabled=self.amp)
        scheduler = self._get_scheduler(optimizer) if scheduler else None

        #

        qbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
        for epoch in qbar:
            qbar.set_postfix(best_acc=self._best_metrics_test)
            loss, metrics = self._train_loop_step(model, data, optimizer, scaler, scheduler)

            if self._early_stop(metrics, epoch, model):
                break

        max_vram = torch.cuda.max_memory_reserved(self.device) / 1024 ** 2
        print(f"Max VRAM: {max_vram:.2f}MB")
        return self.best_state_dict, self._best_epoch, self._best_metrics_val, self._best_metrics_test

    def _get_optimizer(self, model, lr, weight_decay):
        optimizer = getattr(torch.optim, self.optim)(model.parameters(),
                                                     lr=lr, weight_decay=weight_decay)
        return optimizer

    def _get_scheduler(self, optimizer):
        warmup_epochs = self.scheduler_kwargs["warmup_epochs"]
        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=0.05,
                                    end_factor=1.0,
                                    total_iters=warmup_epochs)

        T_0 = self.scheduler_kwargs.get("T_0", 30)
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

    def _train_loop_step(self, model, data, optimizer, scaler, scheduler):
        train_loss, metrics_train = self._train(model, data, optimizer, scaler, scheduler)
        val_loss, metrics_val, metrics_test = self._evaluate(model, data)
        return [train_loss, val_loss], [metrics_train, metrics_val, metrics_test]

    def _train(self, model, data, optimizer, scaler, scheduler):
        #model.train()
        #optimizer.zero_grad()
        #with torch.autocast(device_type=self.device, enabled=self.amp):
        #    out = model(data.x, data.edge_index)
        #    loss = self.loss_func(out, data, data.train_mask)

        #self._backward(loss, model.parameters(), optimizer, scaler, scheduler)
        #metrics_train = self._metrics(out, data, data.train_mask)
        #return loss.detach().cpu().item(), metrics_train
        raise NotImplementedError("Please implement the _train method in your subclass.")

    def _evaluate(self, model, data):
        #model.eval()
        #with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.amp):
        #    out = model(data.x, data.edge_index)
        #    loss = self.loss_func(out, data, data.val_mask)

        #metrics_val = self._metrics(out, data, data.val_mask)
        #metrics_test = self._metrics(out, data, data.test_mask)
        #return loss.detach().cpu().item(), metrics_val, metrics_test
        raise NotImplementedError("Please implement the _evaluate method in your subclass.")

    @staticmethod
    def _backward(loss, parameters, optimizer, scaler, scheduler, pruner=None, model=None):
        _loss = loss.detach().cpu().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

        if pruner is not None and model is not None:
            pruner.regularize(model)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    @staticmethod
    def _metrics(out, data, mask):
        #metrics = accuracy(out[mask], data.y[mask])
        #return metrics
        raise NotImplementedError("Please implement the _metrics method in your subclass.")

    def _early_stop(self, metrics, epoch, model):
        metrics_train, metrics_val, metrics_test = metrics
        if metrics_val > self._best_metrics_val:
            self._best_epoch = epoch
            self._best_metrics_val = metrics_val
            self._best_metrics_test = metrics_test
            self.best_state_dict = model.state_dict()
            self._patience = 0

        else:
            self._patience += 1
            if self._patience == self._max_patience:
                print(f"Early stopping at epoch {epoch}")
                return True

        return False


class TrainerCLS(Trainer):
    def __init__(self, args, model,
                 epochs, lr=0.001, weight_decay=0.0001,
                 early_stop= None, scheduler=False,
                 mpp=True, mpp_epochs=200, mpp_lr=0.001, mpp_weight_decay=5e-4,
                 mpp_early_stop=200, mpp_scheduler=False, mpp_step=5, mpp_load=False,
                 prune_rate=0.5, prune_type="base", imp_type="group", isomorphic=True, #bnscale
                 optim='Adam', warnup=10, T_0=50, T_mult=1, eta_min=1e-5,
                 amp=True, device=None):
        super().__init__(optim=optim, warnup=warnup, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
                         amp=amp, device=device)
        self.args = args
        self.model = model
        self.mmp = mpp
        self.mpp_step = mpp_step
        self.mpp_load = mpp_load

        self.scheduler_kwargs = {"warmup_epochs": warnup, "T_0": T_0, "T_mult": T_mult, "eta_min": eta_min}

        self.model_training_kwargs = {"epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                                      "early_stop": early_stop, "scheduler": scheduler}
        if self.mmp:
            self.mpp_training_kwargs = {"epochs": mpp_epochs, "lr": mpp_lr, "weight_decay": mpp_weight_decay,
                                        "early_stop": mpp_early_stop, "scheduler": mpp_scheduler}

            self.prune_kwargs = {"prune_rate": prune_rate, "isomorphic": isomorphic,
                                 "prune_type": prune_type, "imp_type":imp_type,}

        #if self.krd:
        #    self.krd_training_kwargs = {"epochs": krd_epochs, "lr": krd_lr, "weight_decay": krd_weight_decay,
        #                                "early_stop": krd_early_stop, "scheduler": krd_scheduler}

    def fit(self, data):
        self.criterion = nn.CrossEntropyLoss() if data.num_classes > 2 else nn.BCEWithLogitsLoss()
        data = data.to(self.device)
        from torch_geometric import EdgeIndex
        data.edge_index = EdgeIndex(data.edge_index, sparse_size=(data.x.size(0), data.x.size(0)),sort_order='row',is_undirected=True,)
        data = data.to(self.device)

        if self.mmp:
            _mpp_model = mirror_projection(self.model, gnn2mlp=True).to(self.device)

            _prune_kwargs = deepcopy(self.prune_kwargs)
            _mpp_training_kwargs = deepcopy(self.mpp_training_kwargs)

           # prune_rate =
            _prune_kwargs["prune_rate"] = (1.0 - (1.0 - self.prune_kwargs["prune_rate"]) ** (1.0 / self.mpp_step))

            _mpp_training_kwargs["epochs"] = self.mpp_training_kwargs["epochs"]

            for i in range(self.mpp_step):
                state_dict, _, _, _metrics_test = self._train_loop(_mpp_model, data, **_mpp_training_kwargs)
                _mpp_model.load_state_dict(state_dict)
                _mpp_model = pruner_func(_mpp_model, data, **_prune_kwargs, _print=False)
                print(f"\nmetrics: {_metrics_test*100:.2f} {_mpp_model.layers[1].in_channels}", )

            gnn_model = mirror_projection(_mpp_model, gnn2mlp=False).to(self.device)
            if self.mpp_load:
                gnn_model.load_state_dict(_mpp_model.state_dict())

            #print(gnn_model.layers)

            state_dict, best_epoch, metrics_val, metrics_test = self._train_loop(gnn_model, data, **self.model_training_kwargs)

        else:
            self.model = self.model.to(self.device)
            state_dict, best_epoch, metrics_val, metrics_test = self._train_loop(self.model, data, **self.model_training_kwargs)

        print(f"Best epoch: {best_epoch}, "
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

    def _train(self, model, data, optimizer, scaler, scheduler):
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.bfloat16):
            out = model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.train_mask)

        self._backward(loss, model.parameters(), optimizer, scaler, scheduler)
        metrics_train = self.metrics(out, data, data.train_mask)
        return loss.detach().cpu().item(), metrics_train

    def _evaluate(self, model, data):
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device, enabled=self.amp):
            out = model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.val_mask)

        metrics_val = self.metrics(out, data, data.val_mask)
        metrics_test = self.metrics(out, data, data.test_mask)
        return loss.detach().cpu().item(), metrics_val, metrics_test

    @staticmethod
    def metrics(out, data, mask):
        metrics = accuracy(out[mask], data.y[mask])
        return metrics


class TrainerSP(TrainerCLS):
    def fit(self, data):
        self.criterion = nn.CrossEntropyLoss() if data.num_classes > 2 else nn.BCEWithLogitsLoss()
        data = data.to(self.device)
        from torch_geometric import EdgeIndex
        data.edge_index = EdgeIndex(data.edge_index, sparse_size=(data.x.size(0), data.x.size(0)),sort_order='row',is_undirected=True,)


        if self.mmp:
            _mpp_model = mirror_projection(self.model, gnn2mlp=True).to(self.device)

            state_dict, _, _, _ = self._train_loop(_mpp_model, data, **self.mpp_training_kwargs)
            _mpp_model.load_state_dict(state_dict)


    def _sp_trainer(self, model, data, prune_rate, isomorphic, prune_type, imp_type,):
        num_class = data.num_classes

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == num_class:
                    ignored_layers.append(m)

        unwrapped_parameters = []
        for name, param in model.named_parameters():
            _name = name.split('.')
            if len(_name) > 2 and (_name[0] not in ("norms", "res_layers") and (_name[2] == "bias")):
                unwrapped_parameters.append([param, 0])

        if imp_type.lower() == "group":
            imp = tp.importance.GroupMagnitudeImportance()

        elif imp_type.lower() == "lamp":
            imp = tp.importance.LAMPImportance()

        elif imp_type.lower() == "fpgm":
            imp = tp.importance.FPGMImportance()

        elif imp_type.lower() == "bnscale":
            imp = tp.importance.BNScaleImportance()

        else:
            raise ValueError("Unknown prune type. Please choose from 'group' or 'lamp'.")

        # BNScalePruner, BasePruner
        if prune_type.lower() == "base":
            pruner_api = tp.pruner.BasePruner
        elif prune_type.lower() == "bnscale":
            pruner_api = tp.pruner.BNScalePruner
        elif prune_type.lower() == "group":
            pruner_api = tp.pruner.GroupNormPruner
        else:
            raise ValueError("Unknown prune type. Please choose from 'base' or 'bnscale'.")

        pruner_api = tp.pruner.BNScalePruner
        pruner = pruner_api(model,
                            example_inputs=[torch.rand_like(data.x), data.edge_index],
                            importance=imp,
                            isomorphic=isomorphic,
                            global_pruning=True,
                            pruning_ratio=prune_rate,
                            ignored_layers=ignored_layers,
                            unwrapped_parameters=unwrapped_parameters,
                            )
        return pruner

    def _train_loop_step(self, model, data, optimizer, scaler, scheduler):
        self.pruner = self._sp_trainer(model, data, **self.prune_kwargs)
        return super()._train_loop_step(model, data, optimizer, scaler, scheduler)

    def _train(self, model, data, optimizer, scaler, scheduler):
        model.train()
        self.pruner.update_regularizer()
        optimizer.zero_grad()
        with torch.autocast(device_type=self.device, enabled=self.amp, dtype=torch.bfloat16):
            out = model(data.x, data.edge_index)
            loss = self.loss_func(out, data, data.train_mask)

        self._backward(loss, model.parameters(), optimizer, scaler, scheduler, pruner=self.pruner, model=model)
        metrics_train = self.metrics(out, data, data.train_mask)
        return loss.detach().cpu().item(), metrics_train

    def _evaluate(self, model, data):
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, [torch.rand_like(data.x), data.edge_index])
        print(
            f"Full:\nmacs:{base_macs:.2f} | params:{base_nparams}")  # print the number of MACs and parameters before pruning
        orgs_list = []
        for layer in model.layers:
            org_count = (torch.abs(layer.lin.weight) > 0).sum()
            orgs_list.append(round(((org_count / layer.lin.weight.reshape(-1).size(0)) * 100).item(), 2))

        return super()._evaluate(model, data)

class TrainerRDK(Trainer):
    def __init__(self, args, teach_model,
                 epochs, lr=0.001, weight_decay=0.0001,
                 early_stop=None, scheduler=False,
                 prune_rate=0.5, prune_type="group", isomorphic=True,
                 krd=False, krd_epochs=200, krd_lr=0.001, krd_weight_decay=0.0001,
                 krd_early_stop=None, krd_scheduler=False,
                 optim='Adam', warnup=10, T_0=50, T_mult=1, eta_min=1e-5,
                 amp=True, device=None):

        super().__init__(optim=optim, warnup=warnup, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
                         amp=amp, device=device)

        self.krd = krd
        self.krd_training_kwargs = {"epochs": krd_epochs, "lr": krd_lr,
                                    "weight_decay": krd_weight_decay,
                                    "early_stop": krd_early_stop,
                                    "scheduler": krd_scheduler}

if __name__ == "__main__":
    from limelight.api.dataset import load_dataset
    from limelight.api.imputer import filling_data
    from limelight.utils import fix_seed
#bnscale_lamp_0.95, avg: 93.91 + 3.55
   # 94.07,3.48
    dataset = "photo"
    split = "fixed"
    missing_rate = 0.0

    seed = 42
    hid_dim = 256
    num_layers = 6
    dropout = 0.5
    res = True
    norm = "layer" # "layer", None, "batch"
    gnn_type = 'gcn' # "gcn", "sage", "gin"
    mpp_epochs = 150
    mpp_lr = 0.001
    epochs = 1000
    lr = 0.001 # 0.005
    prune_lr = 0.0025 # 0.005
    weight_decay = 5e-5 # 5e-5
    mpp_early_stop = -1

    scheduler = False
    prune_scheduler = False
    mpp_scheduler = True
    prune_type = "group" # "base", "bnscale", "group"
    imp_type = "group" # "group", "lamp", "fpgm"
    prune_rate = 0.75
    mpp_step = 3
    mpp_load = True
    isomorphic = True
    mpp = True
    amp = True
    full = False
    runs = 3

    fix_seed(seed)
    dataset, mask = load_dataset(dataset, split, label_num_per_class=20, missing_rate=missing_rate)
    if missing_rate > 0:
        dataset = filling_data(dataset, 'fp')

    print(dataset.train_mask.sum().item(), dataset.val_mask.sum().item(), dataset.test_mask.sum().item())

    fix_seed(seed)
    model = MainModule(dataset.x.size(1), hid_dim, dataset.num_classes, num_layers
                       , dropout=dropout, res=res, norm=norm, gnn_type=gnn_type, gnn_cls=False)

    a = TrainerCLS(args=None, model=model, epochs=epochs, lr=prune_lr, weight_decay=weight_decay,
                 early_stop= None, scheduler=prune_scheduler if mpp else scheduler,
                 mpp=mpp, mpp_epochs=mpp_epochs, mpp_lr=mpp_lr, mpp_weight_decay=weight_decay,
                 mpp_early_stop=mpp_early_stop, mpp_scheduler=mpp_scheduler,mpp_load=mpp_load, mpp_step=mpp_step,
                 prune_rate=prune_rate, prune_type=prune_type, imp_type=imp_type, isomorphic=isomorphic, #bnscale
                 optim='Adam', warnup=10, T_0=20, T_mult=2, eta_min=1e-5,
                 amp=True, device=None)
    a.fit(dataset)


    fix_seed(seed)
    model = MainModule(dataset.x.size(1), hid_dim, dataset.num_classes, num_layers
                       , dropout=dropout, res=res, norm=norm, gnn_type=gnn_type, gnn_cls=False)

    b = TrainerCLS(args=None, model=model, epochs=epochs, lr=prune_lr, weight_decay=weight_decay,
                   early_stop=None, scheduler=prune_scheduler if mpp else scheduler,
                   mpp=mpp, mpp_epochs=mpp_epochs, mpp_lr=mpp_lr, mpp_weight_decay=weight_decay,
                   mpp_early_stop=mpp_early_stop, mpp_scheduler=mpp_scheduler, mpp_load=False, mpp_step=mpp_step,
                   prune_rate=prune_rate, prune_type=prune_type, imp_type=imp_type, isomorphic=isomorphic,  # bnscale
                   optim='Adam', warnup=10, T_0=20, T_mult=2, eta_min=1e-5,
                   amp=True, device=None)
    b.fit(dataset)

    fix_seed(seed)
    model = MainModule(dataset.x.size(1), hid_dim, dataset.num_classes, num_layers
                       , dropout=dropout, res=res, norm=norm, gnn_type=gnn_type, gnn_cls=False)

    b = TrainerCLS(args=None, model=model, epochs=epochs, lr=prune_lr, weight_decay=weight_decay,
                   early_stop=None, scheduler=prune_scheduler if mpp else scheduler,
                   mpp=mpp, mpp_epochs=1, mpp_lr=0.0, mpp_weight_decay=weight_decay,
                   mpp_early_stop=mpp_early_stop, mpp_scheduler=mpp_scheduler, mpp_load=False, mpp_step=mpp_step,
                   prune_rate=prune_rate, prune_type=prune_type, imp_type=imp_type, isomorphic=isomorphic,  # bnscale
                   optim='Adam', warnup=10, T_0=20, T_mult=2, eta_min=1e-5,
                   amp=True, device=None)
    b.fit(dataset)


    fix_seed(seed)
    model = MainModule(dataset.x.size(1), hid_dim, dataset.num_classes, num_layers
                       , dropout=dropout, res=res, norm=norm, gnn_type=gnn_type, gnn_cls=False)

    b = TrainerCLS(args=None, model=model, epochs=epochs, lr=lr, weight_decay=weight_decay,
                   early_stop=None, scheduler=prune_scheduler if mpp else scheduler,
                   mpp=False, mpp_epochs=mpp_epochs, mpp_lr=mpp_lr, mpp_weight_decay=weight_decay,
                   mpp_early_stop=mpp_early_stop, mpp_scheduler=mpp_scheduler, mpp_load=False, mpp_step=mpp_step,
                   prune_rate=prune_rate, prune_type=prune_type, imp_type=imp_type, isomorphic=isomorphic,  # bnscale
                   optim='Adam', warnup=10, T_0=20, T_mult=2, eta_min=1e-5,
                   amp=True, device=None)
    b.fit(dataset)











