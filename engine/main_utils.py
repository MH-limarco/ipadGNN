
import os

from engine.moudle.model import LimelightModule, mirror_projection
from engine.trainer import TrainerPrune, TrainerCLS, TrainerRKD

def check_nan(value):
    import math
    return value is not None and not (isinstance(value, float) and math.isnan(value))

def _rename_config(config, rename_dict):
    new_config = {}
    for new_key, org_key in rename_dict.items():
        if isinstance(org_key, (tuple, list)):
            new_config[new_key] = config.get(*org_key)
        else:
            new_config[new_key] = config[org_key]
    return new_config

def _train_prune(config, dataset, model, early_stop=-1, amp=True, _print=False):
    rename_dict = {"epochs": "prune_epochs",
                   "lr": "prune_lr", "weight_decay": "prune_weight_decay",
                   "prune_rate": "prune_rate", "mpp_step": "prune_step",
                   "scheduler": "prune_scheduler", "warmup": "prune_warmup",
                   "T_0": "prune_T_0"}

    prune_config = _rename_config(config, rename_dict)

    imlp_trainer = TrainerPrune(args=None, model=model, _print=_print,
                                early_stop=early_stop, amp=amp, gpu_idx=0, **prune_config)

    imlp_trainer.fit(dataset)
    return imlp_trainer

def _train_cls(config, dataset, model, early_stop=-1, amp=True, _print=False):
    gnn_trainer = TrainerCLS(args=None, model=model,
                             epochs=config["epochs"], lr=config["lr"],
                             weight_decay=config["weight_decay"],
                             scheduler=config.get("scheduler", False), warmup=config.get("warmup", 0.0),
                             T_0=config.get("T_0", 0),
                             early_stop=early_stop, amp=amp, gpu_idx=0, _print=True)
    gnn_trainer.fit(dataset)
    return gnn_trainer

def _train_kd(config, dataset, teacher, student, early_stop=500, amp=True, _print=True):
    rename_dict = {"epochs": "kd_epochs", "lr": "kd_lr",
                   "weight_decay": "kd_weight_decay",
                   "scheduler": "kd_scheduler", "warmup": "kd_warmup",
                   "T_0": "kd_T_0", "label_smoothing": "label_smoothing",
                   "alpha": ("alpha", 0.0), "lamb": "lamb",
                   "momentum": "momentum", "feat_loss": ("feat_loss", None) ,
                   "temperature": ("temperature", 1.0),
                   }

    kd_config = _rename_config(config, rename_dict)
    kd_trainer = TrainerRKD(args=None, teacher=teacher, student=student,
                            early_stop=early_stop, amp=amp, gpu_idx=0, _print=_print,
                            **kd_config)
    kd_trainer.fit(dataset)
    return kd_trainer


def train_imlp(config, dataset, data_name, gnn_type, seed, load=True):
    _root = f"./save/prune_mlp/{data_name}"
    os.makedirs(_root, exist_ok=True)
    _save_path = f"{_root}/best_seed_{seed}_{data_name}_{gnn_type}_{config['prune_rate']}.pt"

    if load and os.path.exists(_save_path):
        trained_imlp = LimelightModule(pt=_save_path)

    else:
        imlp = LimelightModule(dataset.x.size(1), config["hidden"], dataset.num_classes, config["num_layers"],
                               dropout=config["dropout"], res=config["residual"], norm=config["norm"],
                               gnn_type=f"{gnn_type}_mlp", gnn_cls=False)

        imlp_trainer = _train_prune(config, dataset, imlp)
        if load:
            imlp_trainer.model.save(_save_path)
        trained_imlp = imlp_trainer.model

    return trained_imlp, None

def train_gnn(imlp, config, dataset, data_name, gnn_type, seed, early_stop=None, load=True):
    early_stop = 500 if early_stop is None else early_stop
    _root = f"./save/prune_gnn/{data_name}"
    os.makedirs(_root, exist_ok=True)
    _save_path = f"{_root}/best_seed_{seed}_{data_name}_{gnn_type}_{config['prune_rate']}.pt"

    if load and os.path.exists(_save_path):
        trained_gnn = LimelightModule(pt=_save_path)
        trained_gnn._CuGraph = config["cugraph"]
        gnn_trainer = TrainerCLS(args=None, model=trained_gnn,
                                 epochs=config["epochs"], lr=config["lr"],
                                 early_stop=-1, amp=False, gpu_idx=0, _print=True)

    else:
        pruned_gnn = mirror_projection(imlp, target="gnn", cugraph=config["cugraph"])
        gnn_trainer = _train_cls(config, dataset, pruned_gnn, early_stop=early_stop)
        if load:
            gnn_trainer.model.save(_save_path)
        trained_gnn = gnn_trainer.model

    _, _, gnn_test = gnn_trainer.test(dataset)
    return trained_gnn, gnn_test

def train_kd(teacher, student, config, dataset, use_imlp=True, early_stop=None):
    early_stop = 500 if early_stop is None else early_stop
    _config = config
    if use_imlp and config["reset"]:
        student.reset_parameters()

    if not use_imlp:
        student = LimelightModule(dataset.x.size(1), 512, dataset.num_classes, 1,
                                  dropout=0.5, res=False, norm="batch", gnn_type="linear",
                                  gnn_cls=False)

        _config["feat_loss"] = None

    kd_trainer = _train_kd(_config, dataset, early_stop=early_stop, teacher=teacher, student=student)

    _, _, metrics_test = kd_trainer.test(dataset)
    return kd_trainer.model, metrics_test

def train_flow(config, dataset, gnn_type, data_name, seed, early_stop=None, load=True):
    imlp, _ = train_imlp(config, dataset, data_name, gnn_type, seed, load=load)
    gnn, gnn_acc = train_gnn(imlp, config, dataset, data_name, gnn_type, seed, early_stop=early_stop, load=load)
    kd, kd_acc = train_kd(gnn, imlp, config, dataset, seed, early_stop=early_stop)
    return kd_acc, kd_acc
