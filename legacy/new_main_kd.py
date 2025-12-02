
import gc
from time import perf_counter
from copy import deepcopy
from functools import partial
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import optuna
import torch

from limelight.api.dataset import load_dataset
from limelight.api.imputer import filling_data
from limelight.utils import fix_seed

from new_train import TrainerCLS, TrainerPrune, TrainerRKD
from new_model import MainModule, mirror_projection

def read_config(data_name, config_root="./new_best_prune_cfg"):
    import yaml, os
    with open(os.path.join(config_root, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def get_config(data_name, model_type, config_root="./new_best_prune_cfg", prune_rate=0.5):
    _config = read_config(data_name, config_root)
    config = {"dataset": data_name,
              "seed": _config.get("seed", 42)}
    _model_cfg = _config[model_type.lower()]
    _prune_cfg = _model_cfg[round(prune_rate, 1)]
    _model_cfg.pop(0.5)
    _model_cfg.pop(0.6)
    _model_cfg.pop(0.7)

    config.update(_prune_cfg)
    config.update(_model_cfg)
    config.update(_config["dataset"])
    config["epochs"] = int(config["epochs"])
    config["lr"] = float(config["lr"])
    config["weight_decay"] = float(config["weight_decay"])

    if config.get('warnup', False) and config["warnup"] is not None:
        config["warnup"] = float(config["warnup"])
    else:
        config["warnup"] = None

    config["prune_step"] = int(config["prune_step"])
    config["prune_lr"] = float(config["prune_lr"])
    config["prune_weight_decay"] = float(config["prune_weight_decay"])

    if config.get('prune_warmup', False) and config["prune_warmup"] is not None:
        config["prune_warnup"] = float(config["prune_warmup"])
    else:
        config["prune_warnup"] = None

    return config


def experiment_main(data_name, gnn_type,

                    mpp, prune_mlp, missing_rate, prune_rate,

                    kd_dropout, kd_residual, kd_norm,
                    kd_lr, kd_weight_decay, kd_scheduler, kd_warnup,
                    kd_lamb, kd_momentum, kd_smoothing,
                    lr=None, weight_decay=None,
                    runs=3, amp=True, mpnn_amp=False, cugraph=False,
                    optim="Adam",
                    ):

    de_config = get_config(data_name, gnn_type, prune_rate=prune_rate)
    mpp_epochs = de_config["prune_step"] * 150
    kd_epochs = int(de_config["epochs"] * 1.5)

    if lr is not None:
        de_config["lr"] = lr
    if weight_decay is not None:
        de_config["weight_decay"] = weight_decay

    return experiment_loop(mpp=mpp, prune_mlp=prune_mlp,
                           optim=optim, gnn_type=gnn_type, missing_rate=missing_rate,

                           mpp_epochs=mpp_epochs, prune_rate=prune_rate,

                           kd_dropout=kd_dropout, kd_residual=kd_residual, kd_norm=kd_norm,
                           kd_epochs=kd_epochs, kd_lr=kd_lr, kd_weight_decay=kd_weight_decay, kd_scheduler=kd_scheduler,
                           kd_lamb=kd_lamb, kd_warnup=kd_warnup, kd_smoothing=kd_smoothing, kd_momentum=kd_momentum,

                           runs=runs, amp=amp, mpnn_amp=mpnn_amp, cugraph=cugraph, **de_config)



def experiment_loop(seed, dataset, split,
                    missing_rate, optim,
                    mpp, prune_mlp,
                    gnn_type, hidden, num_layers, dropout, residual, norm,
                    epochs, lr, weight_decay, scheduler, warnup,

                    prune_step, prune_rate, pruner_type, imp_type, isomorphic,
                    mpp_epochs, prune_lr, prune_weight_decay, prune_scheduler, prune_warmup,

                    kd_dropout, kd_residual, kd_norm,
                    kd_epochs, kd_lr, kd_weight_decay, kd_scheduler, kd_warnup,
                    kd_lamb, kd_momentum, kd_smoothing,
                    label_num_per_class=20, valid_num=500, test_num=1000,
                    runs=3, amp=True, mpnn_amp=False, cugraph=False, **kwargs):

    fix_seed(seed)
    dataset, mask = load_dataset(dataset, split, missing_rate=missing_rate,
                                 label_num_per_class=label_num_per_class, valid_num=valid_num, test_num=test_num,
                                 )
    if missing_rate > 0.0:
        dataset = filling_data(dataset, "fp")

    torch.cuda.empty_cache()

    exp_acc, exp_time = [], []
    for idx in range(runs):
        best_acc, train_time = experiment_step(idx, seed, dataset, optim,
                                               mpp, prune_mlp,
                                               gnn_type, hidden, num_layers, dropout, residual, norm,
                                               epochs, lr, weight_decay, scheduler, warnup,

                                               prune_step, prune_rate, pruner_type, imp_type, isomorphic,
                                               mpp_epochs, prune_lr, prune_weight_decay, prune_scheduler, prune_warmup,

                                               kd_dropout, kd_residual, kd_norm,
                                               kd_epochs, kd_lr, kd_weight_decay, kd_scheduler, kd_warnup,
                                               kd_lamb, kd_momentum, kd_smoothing,

                                               amp, mpnn_amp, cugraph)

        exp_acc.append(best_acc)
        exp_time.append(train_time)

        torch.cuda.empty_cache()
        gc.collect()
    return sum(exp_acc) / len(exp_acc)

def experiment_step(run_idx, seed, dataset, optim,
                    mpp, prune_mlp,
                    gnn_type, hidden, num_layers, dropout, residual, norm,
                    epochs, lr, weight_decay, scheduler, warnup,

                    prune_step, prune_rate, prune_type, imp_type, isomorphic,
                    mpp_epochs, mpp_lr, mpp_weight_decay, mpp_scheduler, mpp_warnup,

                    kd_dropout, kd_residual, kd_norm,
                    kd_epochs, kd_lr, kd_weight_decay, kd_scheduler, kd_warnup,
                    kd_lamb, kd_momentum, kd_smoothing,

                    amp=True, mpnn_amp=False, cugraph=False):

    run_seed = seed * (run_idx + 1)
    fix_seed(run_seed)
    start = perf_counter()

    model = MainModule(dataset.x.size(1), hidden, dataset.num_classes, num_layers, cugraph=cugraph,
                       dropout=dropout, res=residual, norm=norm, gnn_type=gnn_type, gnn_cls=False)

    if mpp:
        gnnmlp = mirror_projection(model, gnn2mlp=True, cugraph=cugraph)
        prune_trainer = TrainerPrune(args=None, model=gnnmlp,
                         epochs=mpp_epochs, lr=mpp_lr, weight_decay=mpp_weight_decay,
                         early_stop=-1, scheduler=mpp_scheduler,
                         mpp_step=prune_step, prune_rate=prune_rate,
                         prune_type=prune_type, imp_type=imp_type, isomorphic=isomorphic, global_pruning=True,
                         optim=optim, warnup=mpp_warnup,
                         amp=amp, device=None, _print=False)

        prune_trainer.fit(dataset)
        if prune_mlp:
            sample_mlp = deepcopy(prune_trainer.model)
            sample_mlp.reset_parameters()

        model_pruned = mirror_projection(prune_trainer.model, gnn2mlp=False, cugraph=cugraph)

        gnn_trainer = TrainerCLS(args=None, model=model_pruned,
                       epochs=epochs, lr=lr, weight_decay=weight_decay,
                       early_stop=-1, scheduler=scheduler,
                       optim=optim, warnup=warnup,
                       amp=mpnn_amp, device=None, _print=True)

        gnn_trainer.fit(dataset)

    else:
        gnn_trainer = TrainerCLS(args=None, model=model,
                       epochs=epochs, lr=lr, weight_decay=weight_decay,
                       early_stop=-1, scheduler=scheduler,
                       optim=optim, warnup=0.1,
                       amp=mpnn_amp, device=None, _print=True)

        gnn_trainer.fit(dataset)

    gnn_trainer.model.load_state_dict(gnn_trainer.best_state_dict)
    if not (mpp and prune_mlp):
        sample_mlp = MainModule(dataset.x.size(1), 512, dataset.num_classes, 1, cugraph=cugraph,
                                dropout=kd_dropout, res=kd_residual, norm=kd_norm, gnn_type="linear", gnn_cls=False, act="relu")

    RKD_trainer = TrainerRKD(args=None, teacher=gnn_trainer.model, model=sample_mlp, lamb=kd_lamb,
                   epochs=kd_epochs, lr=kd_lr, weight_decay=kd_weight_decay,
                   early_stop=-1, scheduler=kd_scheduler,
                   optim=optim, warnup=kd_warnup, momentum=kd_momentum,
                   amp=amp, device=None, _print=True)

    RKD_trainer.fit(dataset, kd_smoothing)
    RKD_trainer.model.load_state_dict(RKD_trainer.best_state_dict)

    train_time = perf_counter() - start

    return RKD_trainer._best_metrics_test, train_time


def _get_gnn_suggest(trial, mpp=True):
    if mpp:
        prune_rate = trial.suggest_float("prune_rate", 0.5, 0.7, step=0.1)

    else:
        prune_rate = None

    args = {
            "mpp": mpp,
            "prune_rate":prune_rate,
            }

    return args

def _kd_mlp_suggest(trial):
    prune_mlp = trial.suggest_categorical("prune_mlp", [True, False])
    if not prune_mlp:
        kd_dropout = trial.suggest_float("kd_dropout", 0.1, 0.8, step=0.05)
        kd_residual = trial.suggest_categorical("kd_residual", [True, False])
        kd_norm = trial.suggest_categorical("kd_norm", ["layer", "batch", None])
    else:
        kd_dropout = None
        kd_residual = None
        kd_norm = None

    kd_mlp_args = {"prune_mlp": prune_mlp, "kd_dropout": kd_dropout,
                   "kd_residual": kd_residual,"kd_norm": kd_norm}
    return kd_mlp_args

def _get_kd_suggest(trial):
    kd_smoothing = trial.suggest_float("kd_smoothing", 0.0, 0.3, step=0.05)
    kd_lamb = trial.suggest_float("kd_lamb", 0.1, 0.9, step=0.1)
    kd_momentum = trial.suggest_categorical("kd_momentum", [0.995, 0.99, 0.9, 0.8])

    kd_lr = trial.suggest_float("kd_lr", 1e-5, 1e-2, log=True)
    kd_weight_decay = trial.suggest_float("kd_weight_decay", 1e-5, 1e-0, log=True)

    kd_scheduler = trial.suggest_categorical("kd_scheduler", [True, False])
    if kd_scheduler:
        kd_warnup = trial.suggest_float("kd_warnup", 0.0, 0.5, step=0.05)
    else:
        kd_warnup = None

    kd_args = {"kd_smoothing": kd_smoothing,
               "kd_lamb": kd_lamb, "kd_momentum": kd_momentum,
               "kd_lr": kd_lr, "kd_weight_decay": kd_weight_decay,
               "kd_scheduler": kd_scheduler, "kd_warnup": kd_warnup}
    return kd_args


def optuna_objective(trial, dataset_name, model_name, runs):
    args = {"missing_rate": 0.0}

    # ---- 常見連續 / 離散參數 ----

    #mpp = trial.suggest_categorical("mpp", [True, False])
    mpp = True


    # ---- 條件式參數：MPP ----
    gnn_args = _get_gnn_suggest(trial, mpp=mpp)

    # ---- 條件式參數：KD fullmlp ----
    kd_mlp_args = _kd_mlp_suggest(trial)

    # ---- 條件式參數：KD ----
    kd_args = _get_kd_suggest(trial)

    kwargs = {**args, **gnn_args, **kd_mlp_args, **kd_args}
    avg_acc = experiment_main(data_name=dataset_name, gnn_type=model_name,
                                        **kwargs,
                                        runs=runs, amp=True, mpnn_amp=False,
                                        cugraph= True if dataset_name == "physics" else False,
                                        optim="Adam",
                                        )

    print(f"\nTrial {trial.number} finished with kd accuracy: {avg_acc:.4f}\n")
    avg_acc = avg_acc if np.isfinite(avg_acc) else 0.0
    return avg_acc

def optuna_optimize(dataset_name, gnn_type, runs=3, n_trials=50):
    import os
    root = "new_optuna_kd"
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "db", dataset_name.lower()), exist_ok=True)
    os.makedirs(os.path.join(root, "csv", dataset_name.lower()), exist_ok=True)

    if os.path.exists(f"./{root}/csv/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv"):
        print(f"File: {root}/csv/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv already exists. Skipping optimization.")
        return


    study = optuna.create_study(
        study_name=f"{dataset_name.lower()}_{gnn_type.lower()}",
        directions=["maximize"],
        storage=f"sqlite:///./{root}/db/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.db",  # 本地儲存路徑
        load_if_exists=True  # 避免重複建立新 study
    )

    trial = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trial: {trial} / {n_trials}")
    if trial < n_trials:
        t = 0
        try:
            study.optimize(partial(optuna_objective, dataset_name=dataset_name,
                                   model_name=gnn_type, runs=runs), n_trials=n_trials)
        except:
            t += 1
            if t < 2:
                optuna_optimize(dataset_name, gnn_type, runs=runs, n_trials=n_trials)
            else:
                return


    df = study.trials_dataframe()
    df.to_csv(f"./{root}/csv/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv", index=False)
        #optuna.visualization.plot_pareto_front(study).show()


if __name__ == "__main__":
    for dataset in ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]: #[::-1]
        for conv_name in ["gcn", "sage"]:
            optuna_optimize(dataset_name=dataset, gnn_type=conv_name, runs=3, n_trials=49)

