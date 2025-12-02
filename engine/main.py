

import os
import gc
import time

import torch
import numpy as np
import pandas as pd

from limelight.utils import fix_seed
from limelight.api.dataset import load_dataset
from limelight.api.imputer import filling_data

from engine.moudle.model import LimelightModule, mirror_projection
from engine.trainer import TrainerPrune, TrainerCLS, TrainerRKD


use_imlp = True

def check_nan(value):
    import math
    return value is not None and not (isinstance(value, float) and math.isnan(value))

def round_floats(d):
    import re
    for k, v in d.items():
        if isinstance(v, float):
            # 找到第一個非零數字的位置，並往後保留一位
            v_str = f"{v:.16e}"
            match = re.search(r"([1-9]\d*)(\.\d+)?e([+-]?\d+)", v_str)
            if match:
                # 計算有效數字的總長度 +1
                significant_length = len(match.group(1)) + 1
                # 根據有效數字長度進行 round
                d[k] = round(v, significant_length - int(match.group(3)))
        elif isinstance(v, dict):
            round_floats(v)
    return d

def _read_cfg(data_name, root, path):
    import yaml, os
    with open(os.path.join(root, path, f"{data_name}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def train_imlp(config, dataset, model):
    imlp_trainer = TrainerPrune(args=None, model=model,
                                epochs=config["prune_epochs"], lr=config["prune_lr"],
                                weight_decay=config["prune_weight_decay"],
                                prune_rate=config["prune_rate"], mpp_step=config["prune_step"],
                                prune_type=config["pruner_type"], imp_type=config["imp_type"],
                                isomorphic=config["isomorphic"],
                                scheduler=config["prune_scheduler"], warmup=config["prune_warmup"],
                                T_0=config["prune_T_0"],
                                early_stop=-1, amp=True, gpu_idx=0, _print=False)

    imlp_trainer.fit(dataset)
    return imlp_trainer

def train_gnn(config, dataset, model, early_stop=500):
    gnn_trainer = TrainerCLS(args=None, model=model,
                              epochs=config["epochs"], lr=config["lr"],
                              weight_decay=config["weight_decay"],
                              scheduler=config.get("scheduler", False), warmup=config.get("warmup", 0.0),
                              T_0=config.get("T_0", 0),
                              early_stop=early_stop, amp=False, gpu_idx=0, _print=True)
    gnn_trainer.fit(dataset)
    return gnn_trainer

def train_kd(config, dataset, teacher, student, early_stop=500, krd=True):
    if not krd:
        config["lamb"] = 0.1
        config["label_smoothing"] = 0.0
        config["kd_lr"] = 0.005
        config["kd_epochs"] = min(config["kd_epochs"], 1500)

    kd_trainer = TrainerRKD(args=None, teacher=teacher, student=student,
                             epochs=config["kd_epochs"], momentum=config["momentum"],
                             lamb=config["lamb"], alpha=config.get("alpha", 0.0),
                             feat_loss=config.get("feat_loss", None), temperature=config.get("temperature", 1.0),
                             lr=config["kd_lr"], weight_decay=config["kd_weight_decay"],
                             scheduler=config["kd_scheduler"], warmup=config["kd_warmup"],
                             T_0=config["kd_T_0"], label_smoothing=config["label_smoothing"],
                             early_stop=early_stop, krd=krd, amp=False, gpu_idx=0, _print=True)
    kd_trainer.fit(dataset)
    return kd_trainer


def experiment_prune_one_step(config, dataset, gnn_type, data_name, load, seed, gnn_cls=False, return_time=False, early_stop=500, krd=True, use_org=False):
    imlp_root = f"./save/prune_mlp/{data_name}"
    os.makedirs(imlp_root, exist_ok=True)
    imlp_save_path = f"{imlp_root}/best_seed_{seed}_{data_name}_{gnn_type}_{config['prune_rate']}.pt"

    if use_org:
        #config["prune_epochs"] = 1
        config["prune_rate"] = 1.0

    print(f"prune_rate : {config['prune_rate']:.2f}")
    if load and os.path.exists(imlp_save_path):
        trained_imlp = LimelightModule(pt=imlp_save_path)


    else:
        imlp = LimelightModule(dataset.x.size(1), config["hidden"], dataset.num_classes, config["num_layers"],
                               dropout=config["dropout"], res=config["residual"], norm=config["norm"],
                               gnn_type=f"{gnn_type}_mlp", gnn_cls=gnn_cls)

        imlp_trainer = train_imlp(config, dataset, imlp)
        if load:
            imlp_trainer.model.save(imlp_save_path)
        trained_imlp = imlp_trainer.model

    gnn_root = f"./save/prune_gnn/{data_name}"
    os.makedirs(gnn_root, exist_ok=True)
    gnn_save_path = f"{gnn_root}/best_seed_{seed}_{data_name}_{gnn_type}_{config['prune_rate']}.pt"
    if load and os.path.exists(gnn_save_path) and not return_time:
        trained_gnn = LimelightModule(pt=gnn_save_path)
        trained_gnn._CuGraph = config["cugraph"]
        gnn_trainer = TrainerCLS(args=None, model=trained_gnn,
                                 epochs=config["epochs"], lr=config["lr"],
                                 early_stop=-1, amp=False, gpu_idx=0, _print=True)

    else:
        pruned_gnn = mirror_projection(trained_imlp, target="gnn", cugraph=config["cugraph"])
        _start = time.perf_counter()
        gnn_trainer = train_gnn(config, dataset, pruned_gnn, early_stop=early_stop)
        _train_time_gnn = time.perf_counter() - _start

        if load:
            gnn_trainer.model.save(gnn_save_path)
        trained_gnn = gnn_trainer.model

    _start = time.perf_counter()
    _, _, gnn_test = gnn_trainer.test(dataset)
    _inference_time_gnn = time.perf_counter() - _start

    if config["reset"]:
        trained_imlp.reset_parameters()

    student = trained_imlp

    if not use_imlp:
        student = LimelightModule(dataset.x.size(1), 512, dataset.num_classes, 1,
                              dropout=0.5, res=False, norm="batch", gnn_type="linear",
                              gnn_cls=False)
        config["feat_loss"] = None



    _start = time.perf_counter()
    kd_trainer = train_kd(config, dataset, teacher=trained_gnn, student=student, early_stop=early_stop, krd=krd)
    _train_time_kd = time.perf_counter() - _start

    _start = time.perf_counter()
    metrics_train, metrics_val, metrics_test = kd_trainer.test(dataset)
    _inference_time_kd = time.perf_counter() - _start

    if return_time:
        return ([(kd_trainer._stop_epoch + 1) / _train_time_kd, _inference_time_kd],
                [(gnn_trainer._stop_epoch + 1) / _train_time_gnn, _inference_time_gnn])
    return metrics_test, gnn_test

def experiment_full_one_step(config, dataset, gnn_type, data_name, load, seed, gnn_cls=False, return_time=False, early_stop=500):
    gnn_root = f"./save/full_gnn/{data_name}"
    os.makedirs(gnn_root, exist_ok=True)
    gnn_save_path = f"{gnn_root}/best_seed_{seed}_{data_name}_{gnn_type}_full.pt"
    if load and os.path.exists(gnn_save_path) and not return_time:
        trained_gnn = LimelightModule(pt=gnn_save_path)
        gnn_trainer = TrainerCLS(args=None, model=trained_gnn,
                                 epochs=config["epochs"], lr=config["lr"],
                                 early_stop=-1, amp=False, gpu_idx=0, _print=True)

    else:
        gnn = LimelightModule(dataset.x.size(1), config["hidden"], dataset.num_classes, config["num_layers"],
                               dropout=config["dropout"], res=config["residual"], norm=config["norm"], cugraph=config["cugraph"],
                               gnn_type=gnn_type, gnn_cls=gnn_cls)

        _start = time.perf_counter()
        gnn_trainer = train_gnn(config, dataset, gnn, early_stop=early_stop)
        _train_time_gnn = time.perf_counter() - _start


        if load:
            gnn_trainer.model.save(gnn_save_path)
        trained_gnn = gnn_trainer.model

    _start = time.perf_counter()
    _, _, gnn_test = gnn_trainer.test(dataset)
    _inference_time_gnn = time.perf_counter() - _start

    if return_time:
        return [0.0, 0.0], [(gnn_trainer._stop_epoch + 1) / _train_time_gnn, _inference_time_gnn]
    return 0.0, gnn_test


def experiment(config, data_name, gnn_type, split, mode, missing_type, missing_rate, params,
               missing_edage_rate=None, runs=3, classic=False, return_time=False, early_stop=500, krd=True, prune=True):
    model_config = config[gnn_type]

    model_config["cugraph"] = True if gnn_type == "sage" and data_name in ["cs", "physics"] else False
    if mode != "full":
        model_config["kd_epochs"] = int(min(model_config["epochs"] * 2, 2000))
        model_config["prune_epochs"] = int(150 * model_config["prune_step"])
        model_config["prune_step"] = int(model_config["prune_step"])

    model_config = {k: float(v) if "lr" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: float(v) if "weight_decay" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: int(v) if "epochs" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: int(v) if "T_0" in k and check_nan(v) else v for k, v in model_config.items()}

    params["batch"] = 1024 if data_name == "physics" else None

    config["dataset"]["split_method"] = config["dataset"].pop("split")

    fix_seed(config["seed"])
    dataset, _ = load_dataset(data_name, **config["dataset"], missing_type=missing_type, missing_rate=missing_rate)
    print(f"{data_name} - model: {gnn_type} split: {split} mr: {missing_rate}")


    if missing_rate > 0.0:
        print(params)
        dataset = filling_data(dataset, split, **params)

    if missing_edage_rate > 0.0:
        edges_mask = torch.bernoulli(torch.full((dataset.edge_index.size(1),), 1 - missing_edage_rate)).bool()
        print(f"missing edge rate: {missing_edage_rate:0.2f}, edges kept: {edges_mask.sum()}/{dataset.edge_index.size(1)}")
        dataset.edge_index = dataset.edge_index[:, edges_mask]
        model_config["cugraph"] = False

    #load = missing_rate <= 0.0
    load = False
    if classic:
        #mode = "full"
        model_config["num_layers"] = min(model_config["num_layers"], 2)
        model_config["dropout"] = 0.0
        model_config["residual"] = False
        model_config["norm"] = None
        gnn_cls = True

    else:
        gnn_cls = False

    kd_acc, gnn_acc = [], []
    train_times = []

    for run in range(runs):
        print(f"run: {run}/ {runs}")
        _seed = config["seed"] * (run + 1)
        fix_seed(_seed)

        _start_full = time.perf_counter()
        if mode.lower() == "full":
            kd_test, gnn_test = experiment_full_one_step(model_config, dataset, gnn_type, data_name, load, _seed,
                                                         gnn_cls=gnn_cls, return_time=return_time, early_stop=early_stop)
        else:
            kd_test, gnn_test = experiment_prune_one_step(model_config, dataset, gnn_type, data_name, load, _seed,
                                                          gnn_cls=gnn_cls, return_time=return_time, early_stop=early_stop, krd=krd, use_org= not prune)
        train_times.append(time.perf_counter() - _start_full)
        kd_acc.append(kd_test)
        gnn_acc.append(gnn_test)

        torch.cuda.empty_cache()
        gc.collect()

    if return_time:
        kd_acc = np.array(kd_acc).T
        gnn_acc = np.array(gnn_acc).T
        _output =  (f"{kd_acc[0, 1:].mean():.2f} +- {kd_acc[0, 1:].std():.2f}" , f"{(kd_acc[1, 1:].mean())* 1000:.2f} +- {(kd_acc[1, 1:].std())* 1000:.2f}",
                    f"{gnn_acc[0, 1:].mean():.2f} +- {kd_acc[0, 1:].std():.2f}", f"{(gnn_acc[1, 1:].mean())* 1000:.2f} +- {(gnn_acc[1, 1:].std())* 1000:.2f}",
                    f"{sum(train_times) / len(train_times):.2f}")

        return _output
        #return (sum([i[0] for i in kd_acc]) / len(kd_acc), sum([i[1] for i in kd_acc]) / len(kd_acc),
        #        sum([i[0] for i in gnn_acc]) / len(gnn_acc), sum([i[1] for i in gnn_acc]) / len(gnn_acc),
        #        sum(train_times) / len(train_times))

    kd_acc, gnn_acc = np.array(kd_acc) * 100, np.array(gnn_acc) * 100
    return f"{kd_acc.mean():.3f} +- {kd_acc.std() :.4f}", f"{gnn_acc.mean() :.4f} +- {gnn_acc.std():.4f}"

def main_base(data_name, gnn_type, mode=None, classic=False, return_time=False, prune=True, early_stop=500, krd=True):
    mode = "ikd" if mode is None else mode
    use_imlp = True
    if mode == "kd_mlp":
        mode = "kd"
        use_imlp = False

    _text_name = f"kd_{mode}_{data_name}_{gnn_type}" if krd==True else f"glnn_{mode}_{data_name}_{gnn_type}"

    _text_name = _text_name
    if classic:
        _text_name = _text_name + "_classic"

    if not prune:
        _text_name = _text_name + "_full"
        #mode = "full"

    _type = "baselines"
    runs = 10

    if return_time:
        _type = "runtime"
        runs = 5

    _text_root = os.path.join(f"./", f"{_type}{'_classic' if classic else '' }{'_prune' if prune else '_full'}_log",
                              f"{data_name}")

    _text_name = f"{_text_name}.txt"
    os.makedirs(_text_root, exist_ok=True)
    _text_path = os.path.join(_text_root, _text_name)

    if os.path.exists(_text_path):
        print(f"File {_text_path} already exists. Skipping...")
        return

    root = os.path.join("./", "save_log", "best_config", "gnns")
    path = "kd_cfg" if mode == "kd" or (
            'kd' in mode and not use_imlp) else "ikd_cfg" if mode != "full" else "best_cfg"
    cfg = _read_cfg(data_name, root, path)

    output = experiment(config=cfg, data_name=data_name, gnn_type=gnn_type, split="zero",
                        mode=mode, missing_type="uniform", missing_rate=0.0,
                        missing_edage_rate=0.0,
                        params={}, runs=runs, classic=classic, return_time=return_time, early_stop=early_stop, krd=krd, prune=prune)

    if return_time:
        kd_train_time, kd_inference_time, gnn_train_time, gnn_inference_time, full_train_time = output
        print(f"kd:\n"
              f"train_epoch/s: {kd_train_time}, inference_time: {kd_inference_time}\n"
              
              f"gnn:\n"
              f"train_epoch/s: {gnn_train_time}, inference_time: {gnn_inference_time}\n"
              f"train_time: {full_train_time}\n"
              )
        print("save:", _text_root)
        open(_text_path, "a+", encoding="utf-8").write(f"{kd_train_time} {gnn_train_time}\n"
                                                       f"{kd_inference_time} {gnn_inference_time}\n"
                                                       f"{full_train_time}\n")

    else:
        kdacc, gnnacc = output
        print(f"kd acc: {kdacc}, gnn acc: {gnnacc}")
        print("save:", _text_root)
        open(_text_path, "a+", encoding="utf-8").write(f"{kdacc} {gnnacc}\n")


def main(data_name, gnn_type, split, missing_type, missing_rate, params,
          missing_edage_rate=None, mode=None, classic=False, return_time=False):

    mode = "ikd" if mode is None else mode
    use_imlp = True
    if mode == "kd_mlp":
        mode = "kd"
        use_imlp = False


    missing_edage_rate = 0.0 if missing_edage_rate is None else missing_edage_rate
    _text_name = f"kd_{mode}_{data_name}_{gnn_type}"

    if missing_rate == 0.0 and missing_edage_rate >= 0.0:
        _text_name = _text_name + f"_edge_{missing_edage_rate}"
        _text_root = os.path.join(f"./", f"edge_{missing_type}{'_classic' if classic else ''}_log",
                                  f"{data_name}")

    elif missing_rate > 0.0 and missing_edage_rate > 0.0:
        _text_name = _text_name + f"_feat{missing_rate}_edge_{missing_edage_rate}"
        _text_root = os.path.join(f"./", f"feat{missing_rate}_edge_{missing_type}{'_classic' if classic else ''}_log",
                                  f"{data_name}")

    else:
        _text_name = _text_name + f"_{missing_rate}"
        _text_root = os.path.join(f"./", f"{split}_{missing_type}{'_classic' if classic else ''}_log",
                                  f"{data_name}")

    if classic:
        _text_name = _text_name + "_classic"

    if return_time:
        _text_name = _text_name + "_time"

    _text_name = f"{_text_name}.txt"
    os.makedirs(_text_root, exist_ok=True)
    _text_path = os.path.join(_text_root, _text_name)

    if os.path.exists(_text_path):
        print(f"File {_text_path} already exists. Skipping...")
        return

    root = os.path.join("./", "save_log", "best_config", "gnns")
    path = "kd_cfg" if mode == "kd" or (
            'kd' in mode and not use_imlp) else "ikd_cfg" if mode != "full" else "best_cfg"
    cfg = _read_cfg(data_name, root, path)

    kdacc, gnnacc = experiment(config=cfg, data_name=data_name, gnn_type=gnn_type, split=split,
                               mode=mode, missing_type=missing_type, missing_rate=missing_rate,
                               missing_edage_rate=missing_edage_rate,
                               params=params, runs=10, classic=classic)

    print(f"kd acc: {kdacc}, gnn acc: {gnnacc}")
    print("save:", _text_root)
    open(_text_path, "a+", encoding="utf-8").write(f"{kdacc} {gnnacc}\n")

def main_runtime(full=True, classic=False, prune=True, mode=None, krd=True):
    _datasets = datasets if full else datasets[:-2]
    for gnn_type in gnn_types:
        for data_name in _datasets:
            main_base(data_name=data_name, gnn_type=gnn_type, classic=classic, prune=prune, mode=mode, krd=krd, early_stop=-1, return_time=True)

def main_baseline(full=True, classic=False, prune=True, mode=None, krd=True):
    _datasets = datasets if full else datasets[:-2]
    for gnn_type in gnn_types:
        for data_name in _datasets:
            main_base(data_name=data_name, gnn_type=gnn_type, classic=classic, prune=prune, mode=mode, krd=krd)

def main_missing_feat(full=True, classic=False, fast=False):
    missings = [0.3, 0.6, 0.9, 0.995] if not fast else [0.995]
    _datasets = datasets if full else datasets[:-2]
    for mt in missing_types:
        for gnn_type in gnn_types:
            for missing_rate in missings:
                for data_name in _datasets:
                    for split in splits:
                        if split == "apcfi":
                            n_trials = 50 if data_name == "physics" and gnn_type == "sage" else 75 if data_name == "physics" else 100

                            _root = os.path.join("./", f"optuna_apcfi_full_{mt}", "csv", f"{missing_rate}")
                            _path = os.path.join(_root, f"{data_name.lower()}_{gnn_type.lower()}_{n_trials}.csv")
                            _df = pd.read_csv(_path)
                            _best = _df.iloc[0]
                            _best_params = round_floats(_best.filter(like="params_").to_dict())
                            _best_params = {k.replace("params_", ""): v for k, v in _best_params.items()}

                        else:
                            _best_params = {}

                        main(data_name=data_name, gnn_type=gnn_type, missing_type=mt, split=split,
                              missing_rate=missing_rate, params=_best_params, classic=classic)

def main_missing_edges(full=True, classic=False):
    missings_edges = [0.3, 0.6, 0.9]
    _datasets = datasets if full else datasets[:-2]
    for gnn_type in gnn_types:
        for missing_edge_rate in missings_edges:
            for data_name in _datasets:
                main(data_name=data_name, gnn_type=gnn_type, missing_type="uniform", split="zero",
                      params={}, missing_rate=0.0, missing_edage_rate=missing_edge_rate, classic=classic)

def run_baseline():
    print("baseline")
    main_baseline(full=True, classic=True, prune=False)
    main_baseline(full=True, classic=True, prune=True)
    main_baseline(full=True, classic=True, prune=False, mode="kd_mlp")
    main_baseline(full=True, classic=True, prune=False, mode="kd_mlp", krd=False)

    main_baseline(full=True, classic=False, prune=False)
    main_baseline(full=True, classic=False, prune=True)
    main_baseline(full=True, classic=False, prune=True, mode="kd_mlp")
    main_baseline(full=True, classic=False, prune=True, mode="kd_mlp", krd=False)

def run_runtime():
    print("baseline")
    main_runtime(full=True, classic=True, prune=False)
    main_runtime(full=True, classic=True, prune=True)
    main_runtime(full=True, classic=True, prune=False, mode="kd_mlp")
    main_runtime(full=True, classic=True, prune=False, mode="kd_mlp", krd=False)

    main_runtime(full=True, classic=False, prune=False)
    #
    main_runtime(full=True, classic=False, prune=True)
    main_runtime(full=True, classic=False, prune=True, mode="kd_mlp", krd=False)


def main_missing_mix(full=True, classic=False, fast=True):
    print("mix")
    missings = [0.3, 0.6, 0.9, 0.995] if not fast else [0.995]
    _edage_rate = 0.6
    _splits = "apcfi"
    _datasets = datasets if full else datasets[:-2]
    _gnn_types = ["gcn"]
    _missing_types = ["uniform"]
    for mt in _missing_types:
        for gnn_type in _gnn_types:
            for missing_rate in missings:
                for data_name in _datasets:
                    n_trials = 50 if data_name == "physics" and gnn_type == "sage" else 75 if data_name == "physics" else 100
                    _root = os.path.join("./", "optuna_db", f"optuna_apcfi_full_{mt}", "csv", f"{missing_rate}")
                    _path = os.path.join(_root, f"{data_name.lower()}_{gnn_type.lower()}_{n_trials}.csv")
                    _df = pd.read_csv(_path)
                    _best = _df.iloc[0]
                    _best_params = round_floats(_best.filter(like="params_").to_dict())
                    _best_params = {k.replace("params_", ""): v for k, v in _best_params.items()}

                    main(data_name=data_name, gnn_type=gnn_type, missing_type=mt, split=_splits,
                         missing_rate=missing_rate, missing_edage_rate=_edage_rate, params=_best_params, classic=classic)

star = None
beta = None
alpha = None

splits = ["apcfi", "fp", "zero"]
missing_types = ["uniform", "structural"]
gnn_types = ["gcn", "sage"]
datasets = ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]


def done():
    run_runtime()
    run_baseline()

    print("feat")
    main_missing_feat(full=True, classic=True, fast=True)
    main_missing_feat(full=True, fast=True)

    main_baseline(full=True, classic=False, prune=False, mode="kd_mlp")
    main_runtime(full=True, classic=False, prune=False, mode="kd_mlp")

    print("edges")
    main_missing_edges(full=True)
    main_missing_edges(full=True, classic=True)

    main_missing_feat(full=True)
    main_missing_feat(full=True, classic=True)


if __name__ == "__main__":
    done()
    main_missing_mix(full=True, classic=False, fast=True)
    main_missing_mix(full=True, classic=True, fast=True)




