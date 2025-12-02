

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
from engine.utils import LearnFPFilling

limit_error_num = 5
use_imlp = True

def collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def check_nan(value):
    import math
    return value is not None and not (isinstance(value, float) and math.isnan(value))

def safe_rely(func, safe, **kwargs):
    global error_num

    try:
        _output = func(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        collect()
        error_num = 0
        return _output

    except RuntimeError as e:
        if safe and error_num < limit_error_num:
            error_num += 1
            print(
                f"Error occurred {error_num} / {limit_error_num}: {e}. Retrying with {func}.")

            collect()
            safe_rely(func, safe, **kwargs)

        else:
            if safe:
                print(f"Maximum error limit reached. Stopping execution. {error_num} / {limit_error_num}")
            raise e

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

def _read_cfg(data_name, root, path=None):
    import yaml, os
    _join = [root] if not path else [root, path]
    with open(os.path.join(*_join, f"{data_name}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def train_imlp(config, dataset, model, filler=None, amp=True):
    _amp = config.get("prune_amp", True) if amp and config.get("prune_amp", True) else False

    imlp_trainer = TrainerPrune(args=None, model=model,
                                epochs=config["prune_epochs"], lr=config.get("prune_lr", config["lr"]),
                                weight_decay=config.get("prune_weight_decay", config["weight_decay"]),
                                prune_rate=config["prune_rate"], mpp_step=config["prune_step"],
                                prune_type=config.get("pruner_type", "group"), imp_type=config.get("imp_type", "group"),
                                isomorphic=config.get("isomorphic", False),
                                scheduler=config.get("prune_scheduler", False), warmup=config.get("prune_warmup", False),
                                T_0=config.get("prune_T_0", 1),
                                early_stop=-1, amp=_amp, gpu_idx=0, _print=True, filler=filler)

    imlp_trainer.fit(dataset)

    return imlp_trainer

def train_gnn(config, dataset, model, early_stop=500, filler=None):
    gnn_trainer = TrainerCLS(args=None, model=model,
                              epochs=config["epochs"], lr=config["lr"],
                              weight_decay=config["weight_decay"],
                              scheduler=config.get("scheduler", False), warmup=config.get("warmup", 0.0),
                              T_0=config.get("T_0", 0),
                              early_stop=early_stop, amp=False, gpu_idx=0, _print=True, filler=filler)
    gnn_trainer.fit(dataset)
    return gnn_trainer

def train_kd(config, dataset, teacher, student, early_stop=500, krd=True, filler=None):
    if not krd:
        config["lamb"] = 0.1
        config["label_smoothing"] = 0.0
        config["kd_lr"] = 0.005
        config["kd_epochs"] = min(config["kd_epochs"], 1500)

    feat_loss = config.get("feat_loss", None)
    lamb = config.get("lamb", 2.0 if feat_loss is not None else 0.3)
    alpha = config.get("alpha", 0.1 if feat_loss is not None else 0.0)
    kd_trainer = TrainerRKD(args=None, teacher=teacher, student=student,
                             epochs=config["kd_epochs"], momentum=config.get("momentum", 0.9),
                             lamb=lamb , alpha=alpha,
                             feat_loss=feat_loss, temperature=config.get("temperature", 1.0),
                             lr=config.get("kd_lr", config["lr"]), weight_decay=config.get("kd_weight_decay", config["weight_decay"]),
                             scheduler=config.get("kd_scheduler", False), warmup=config.get("kd_warmup", False),
                             T_0=config.get("kd_T_0", 1.0), label_smoothing=config.get("label_smoothing", 0.0),
                             early_stop=early_stop, krd=krd, amp=False, gpu_idx=0, _print=True, filler=filler)
    kd_trainer.fit(dataset)
    return kd_trainer


def core(data_name, model_name, fill_method, missing_type,
        missing_rate=0.0, missing_edge_rate=0.0, prune_rate=0.0, kd_mode=None, runs=3, fill_kwargs=None, **kwargs):
    global error_num
    error_num = 0

    model_name = model_name.lower()
    _is_sota = model_name in sota_types

    kd_mode = "ikd" if kd_mode is None else kd_mode
    mp_mlp = True
    if kd_mode == "kd_mlp":
        kd_mode = "kd"
        mp_mlp = False

    _text_name = f"{model_name.lower()}"

    if missing_rate > 0.0 and missing_edge_rate > 0.0:
        _root = f"mix_missing_{missing_type}"
        _text_name += f"_{missing_type}_{fill_method.lower()}_{missing_rate}_{missing_edge_rate:.2f}"
        if data_name.lower() == "physics" and model_name in ["sage", "polyformer"]:
            runs = 3

    elif missing_rate > 0.0:
        _root = f"feat_missing_{missing_type}"
        _text_name += f"_{missing_type}_{fill_method.lower()}_{missing_rate}_{missing_edge_rate:.2f}"
        if data_name.lower() == "physics" and model_name in ["sage", "polyformer"]:
            runs = 3

    elif missing_edge_rate > 0.0:
        _root = "edge_missing"
        _text_name += f"_{missing_rate}_{missing_edge_rate:.2f}"

    else:
        _root = "full"

    _text_root = os.path.join("./", _root, data_name, kd_mode)

    if isinstance(fill_kwargs, dict):
        for k, v in fill_kwargs.items():
            _text_name += f"_{k}{v}"

    #os.path.join("./", "save_log", "best_config", "gnns")
    root = os.path.join("./", "save_log", "best_config", "gnns" if not _is_sota else "baselines")
    path = None if _is_sota else f"{kd_mode}_cfg" if prune_rate == 'auto' or prune_rate > 0.0 else "best_cfg"
    config = _read_cfg(data_name, root, path)

    if prune_rate == "auto" and model_name in gnn_types:
        prune_rate = config[model_name]["prune_rate"]
        config.update(**config[model_name])
        _text_name = "auto_" + _text_name

    _text_name = f"{f'pr{prune_rate}' if prune_rate > 0.0 else 'full'}_{_text_name}.txt"
    os.makedirs(_text_root, exist_ok=True)
    text_path = os.path.join(_text_root, _text_name)

    if os.path.exists(text_path):
        print(f"File {text_path} already exists. Skipping...")
        return

    result_str = experiment(config, data_name, model_name, missing_type, fill_method,
                 missing_rate=missing_rate, missing_edge_rate=missing_edge_rate, fill_kwargs=fill_kwargs,
                 prune_rate=prune_rate, mp_mlp=mp_mlp, runs=runs, early_stop=500, **kwargs)

    print(result_str)
    print("save to:", text_path)
    open(text_path, "a+", encoding="utf-8").write(result_str)
    collect()


def experiment(config, data_name, model_name, missing_type, fill_method, fill_kwargs=None,
               missing_rate=0.0, missing_edge_rate=0.0, prune_rate=0.0, mp_mlp=True, krd=True,runs=3, early_stop=500,
               safe=True, **kwargs):
    global error_num

    model_config = config[model_name]
    model_config["prune_amp"] = True

    config["dataset"]["split_method"] = config["dataset"].pop("split")

    if data_name == "physics" and fill_kwargs is not None:
        fill_kwargs["batch"] = 1024

    fix_seed(config["seed"])

    print(f"{data_name} - model: {model_name} {f'with {kwargs} ' if kwargs != {} else ''} fill: {fill_method} mr: {missing_rate}")

    dataset = load_and_fill_dataset(config, data_name, fill_method, missing_type,
                                    missing_rate, missing_edge_rate, fill_kwargs, safe=safe)

    filler = None
    if "fp_learning" == fill_method.lower():
        krd = False
        mp_mlp = False
        filler = LearnFPFilling(dim=dataset.x.size(1))


    if model_name in sota_types:
        model_config["prune_rate"] = prune_rate
        model_config = tidy_sota_cfg(model_config, prune_rate)
        model, model_config = build_sota(model_config, model_name=model_name, dataset=dataset, **kwargs)

    elif model_name in gnn_types:
        model_config["cugraph"] = True if model_name in ["sage", "gat"]  and missing_edge_rate== 0.0 and data_name in ["cs", "physics"] else False
        model_config = tidy_gnn_cfg(model_config, prune_rate)
        model, model_config = build_gnn(model_config, model_name=model_name, dataset=dataset, **kwargs)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    kd_acc_avg, gnn_acc_avg = [], []
    kd_train_avg, kd_inference_avg = [], []
    gnn_train_avg, gnn_inference_avg = [], []

    for run in range(runs):
        print(f"run: {run}/ {runs}")
        _seed = config["seed"] * (run + 1)
        fix_seed(_seed)

        gnn_acc, kd_acc, gnn_speed, kd_speed = safe_rely(_experiment_workflow, safe=safe,
                                                         model_config=model_config, model=model, dataset=dataset,
                                                         use_mp_mlp=mp_mlp, krd=krd, early_stop=early_stop, filler=filler)
        #gnn_acc, kd_acc, gnn_speed, kd_speed = _experiment_workflow(model_config, model, dataset, use_mp_mlp=mp_mlp, early_stop=early_stop, safe=safe, **kwargs)

        gnn_acc_avg.append(gnn_acc)
        kd_acc_avg.append(kd_acc)
        gnn_train_avg.append(gnn_speed[0])
        gnn_inference_avg.append(gnn_speed[1])
        kd_train_avg.append(kd_speed[0])
        kd_inference_avg.append(kd_speed[1])

    kd_acc_avg, gnn_acc_avg = np.array(kd_acc_avg) * 100, np.array(gnn_acc_avg) * 100
    kd_train_avg, kd_inference_avg = np.array(kd_train_avg), np.array(kd_inference_avg) * 1000
    gnn_train_avg, gnn_inference_avg = np.array(gnn_train_avg), np.array(gnn_inference_avg) * 1000

    output_res = (f"{np.mean(kd_acc_avg):.2f} +- {np.std(kd_acc_avg):.2f}, {np.mean(gnn_acc_avg):.2f} +- {np.std(gnn_acc_avg):.2f}\n" +
                  f"{np.mean(kd_train_avg):.2f} +- {np.std(kd_train_avg):.2f}, {np.mean(gnn_train_avg):.2f} +- {np.std(gnn_train_avg):.2f}\n"+
                  f"{np.mean(kd_inference_avg):.2f} +- {np.std(kd_inference_avg):.2f}, {np.mean(gnn_inference_avg):.2f} +- {np.std(gnn_inference_avg):.2f}\n"
                  )

    return output_res

def load_and_fill_dataset(config, data_name, fill_method, missing_type, missing_rate, missing_edge_rate, fill_kwargs=None, safe=True):
    dataset, _ = load_dataset(data_name, **config["dataset"], missing_type=missing_type, missing_rate=missing_rate)

    if missing_edge_rate > 0.0:
        if dataset.edge_index.size(1) % 2 == 0:
            edges_mask = torch.bernoulli(torch.full((dataset.edge_index.size(1) // 2,), 1 - missing_edge_rate)).bool()
            print(f"missing edge rate: {missing_edge_rate:0.2f}, edges kept: {edges_mask.sum()}/{dataset.edge_index.size(1) // 2}")
            _edge_index = dataset.edge_index[:, :dataset.edge_index.size(1)//2][:, edges_mask]
            dataset.edge_index = torch.cat([_edge_index, _edge_index.flip(0)], dim=1)
        else:
            raise ValueError(f"Edge index size {dataset.edge_index.size(1)} is not even, cannot apply edge masking.")
            #edges_mask = torch.bernoulli(torch.full((dataset.edge_index.size(1),), 1 - missing_edge_rate)).bool()
            #print(f"missing edge rate: {missing_edge_rate:0.2f}, edges kept: {edges_mask.sum()}/{dataset.edge_index.size(1)}")
            #dataset.edge_index = dataset.edge_index[:, edges_mask]

    if missing_rate > 0.0:
        print(f"{fill_method}_kwargs:", fill_kwargs)
        fill_method = fill_method.lower() if "learning" not in fill_method.lower() else fill_method.lower().replace("_learning", "")

        dataset = safe_rely(filling_data, safe=safe, data=dataset, method=fill_method, **fill_kwargs if fill_kwargs is not None else {})
        #dataset = filling_data(dataset, fill_method, **fill_kwargs if fill_kwargs is not None else {})

    return dataset

def _experiment_workflow(model_config, model, dataset, use_mp_mlp=True, krd=True, early_stop=500, filler=None):
    mp_mlp = mirror_projection(model, target="mlp", cugraph=model_config.get("cugraph", False),
                               mirror_attr=model_config.get("mirror_attr", None))

    if model_config["prune_rate"] > 0.0:
        mp_mlp_trainer = train_imlp(model_config, dataset, mp_mlp, filler=filler)
        mp_mlp = mp_mlp_trainer.model

        model = mirror_projection(mp_mlp, target="gnn",
                                  cugraph=model_config.get("cugraph", False),
                                  mirror_attr=model_config.get("mirror_attr", None))

    collect()
    _start = time.perf_counter()
    gnn_trainer = train_gnn(model_config, dataset, model, early_stop=early_stop, filler=filler)
    _train_time_gnn = time.perf_counter() - _start
    trained_gnn = gnn_trainer.model
    _start = time.perf_counter()
    for _ in range(5):
        _, _, gnn_test = gnn_trainer.test(dataset)
    _inference_time_gnn = (time.perf_counter() - _start) / 5

    if use_mp_mlp and model_config.get("reset", True):
        mp_mlp.reset_parameters()

    student = mp_mlp if use_mp_mlp and mp_mlp is not None else None

    if student is None:
        student = LimelightModule(dataset.x.size(1), 512, dataset.num_classes, 1,
                                  dropout=0.5, res=False, norm="batch", gnn_type="linear",
                                  gnn_cls=False)
        model_config["feat_loss"] = None
    if not use_mp_mlp and not krd:
        return (gnn_test, 0.0, [_train_time_gnn, _inference_time_gnn], [0.0,0.0])

    collect()
    _start = time.perf_counter()
    kd_trainer = train_kd(model_config, dataset, teacher=trained_gnn, student=student, early_stop=early_stop, krd=krd, filler=filler)
    _train_time_kd = time.perf_counter() - _start

    _start = time.perf_counter()
    for _ in range(5):
        _, _, kd_test = kd_trainer.test(dataset)
    _inference_time_kd = (time.perf_counter() - _start) / 5

    gnn_train_speed = (gnn_trainer._stop_epoch + 1) / _train_time_gnn
    kd_train_speed = (kd_trainer._stop_epoch + 1) / _train_time_kd

    del gnn_trainer, kd_trainer, trained_gnn, student
    collect()
    return (gnn_test, kd_test, [gnn_train_speed, _inference_time_gnn], [kd_train_speed, _inference_time_kd])

def tidy_gnn_cfg(model_config, prune_rate=None):
    model_config["kd_epochs"] = int(min(model_config["epochs"] * 2, 2000))

    model_config = {k: float(v) if "lr" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: float(v) if "weight_decay" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: int(v) if "epochs" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: int(v) if "T_0" in k and check_nan(v) else v for k, v in model_config.items()}

    if not model_config.get("prune_rate", False):
        model_config["prune_rate"] = prune_rate if isinstance(prune_rate, float) else float(prune_rate)
        model_config["prune_amp"] = True

    return model_config

def tidy_sota_cfg(model_config, prune_rate):
    model_config["kd_epochs"] = int(min(model_config["epochs"] * 2, 2000))
    model_config["feat_loss"] = model_config.get("feat_loss", "cwd")

    if prune_rate > 0.0:
        model_config["prune_step"] = 1
        model_config["prune_epochs"] = int(250 * model_config["prune_step"])

    model_config = {k: float(v) if "lr" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: float(v) if "weight_decay" in k and check_nan(v) else v for k, v in model_config.items()}

    return model_config


def build_gnn(model_config, model_name, dataset, **kwargs):
    model_config.update(**kwargs)
    model_config["kd_epochs"] = int(min(model_config["epochs"] * 2, 2000))

    if model_config["prune_rate"] > 0.0:
        model_config["prune_epochs"] = int(250 * model_config["prune_step"])
        model_config["prune_step"] = int(model_config["prune_step"])

    model = LimelightModule(dataset.x.size(1), model_config["hidden"], dataset.num_classes, model_config["num_layers"],
                            dropout=model_config["dropout"], res=model_config["residual"], norm=model_config["norm"],
                            gnn_type=model_name.replace("classic", ""), gnn_cls="classic" in model_name)

    return model, model_config

def build_sota(model_config, model_name, dataset, **kwargs):
    model_config.update(**kwargs)
    if model_name == "sgformer":
        from engine.save_log.baselines.sgformer import SGFormer
        gnn = LimelightModule(dataset.x.size(1), model_config["hidden"], model_config["hidden"], model_config["layers"],
                              dropout=model_config["dropout"], res=False, norm='batch', gnn_type="gcn",
                              gnn_cls=True)

        model = SGFormer(in_channels=dataset.x.size(1), hidden_channels=model_config["hidden"],
                         out_channels=dataset.num_classes,
                         num_layers=model_config["layers"], dropout=model_config["dropout"],
                         gnn=gnn, num_heads=model_config["num_heads"])

        model_config["mirror_attr"] = ["gnn"]
        #model_config["feat_loss"] = None

        if model_config["prune_rate"] > 0.0:
            model_config["prune_amp"] = False
            model_config["prune_epochs"] = 400 * model_config.get("prune_step", 1)
            model_config["epochs"] = int(min(model_config["epochs"] * 1.5, 2500))

    elif model_name == "polynormer":
        from engine.save_log.baselines.polynormer import Polynormer

        model = Polynormer(in_channels=dataset.x.size(1), hidden_channels=model_config["hidden"], out_channels=dataset.num_classes,
                           local_layers=model_config["local_layers"], global_layers=model_config["global_layers"],
                           heads=model_config["num_heads"], dropout=model_config["dropout"], in_dropout=model_config["in_dropout"],
                           g_epochs=model_config["g_epochs"],)

        model_config["mirror_attr"] = ["local_convs"]
        model_config["prune_epochs"] = 600 * model_config.get("prune_step", 1)
        model_config["feat_loss"] = None


        if model_config["prune_rate"] > 0.0:
            model_config["prune_amp"] = False
            model_config["prune_lr"] = 0.005
            model_config["prune_weight_decay"] = 5e-4
            model_config["isomorphic"] = True

            model_config["epochs"] = int(min(model_config["epochs"] * 1.5, 2500))

            if model_config["g_epochs"] > 0:
                model_config["lr"] = 0.01

    else:
        raise ValueError(f"Unsupported model: {model_name}")


    return model, model_config



def missing_exp(model_name, fill_method="zero", feat_missing_type="uniform",
                feat_missing=0.0, edge_missing=0.0, prune_rate=0.0, kd_mode="ikd", runs=5, fast=False, **kwargs):
    use_datasets = sota_datasets if model_name in sota_types else datasets

    if edge_missing > 0.0 and feat_missing > 0.0:
        use_datasets = mix_dataset

    if fast:
        _datasets = use_datasets[:-1] if not isinstance(fast, int) else use_datasets[:-fast]
    else:
        _datasets = use_datasets

    for data_name in _datasets:
        fill_kwargs = None
        if fill_method == "apcfi" and feat_missing > 0.0:
            _model_name = "gcn" if model_name in sota_types else model_name

            n_trials = 50 if data_name == "physics" and _model_name == "sage" else 75 if data_name == "physics" else 100
            _root = os.path.join("./", f"optuna_apcfi_full_{feat_missing_type}", "csv", f"{feat_missing}")
            _path = os.path.join(_root, f"{data_name.lower()}_{_model_name.lower()}_{n_trials}.csv")
            _df = pd.read_csv(_path)
            _best = _df.iloc[0]
            _best_params = round_floats(_best.filter(like="params_").to_dict())
            fill_kwargs = {k.replace("params_", ""): v for k, v in _best_params.items()}


        core(data_name, model_name, fill_method, feat_missing_type, feat_missing, edge_missing, prune_rate,
             kd_mode=kd_mode, fill_kwargs=fill_kwargs, runs=runs, **kwargs)


def mix_exp(model_name, fill_method='zero', prune_rate=0.0, feat_missing=0.995, kd_mode="ikd", runs=5, fast=False, full=False, **kwargs):
    missing_exp(model_name, fill_method=fill_method, feat_missing_type="uniform",
                feat_missing=feat_missing, edge_missing=0.6, prune_rate=prune_rate, kd_mode=kd_mode, runs=runs,
                fast=fast, **kwargs)
    if full:
        emrs = [0.3, 0.6, 0.9] if model_name == "gcn" else [0.9]
        for emr in emrs:
            missing_exp(model_name, edge_missing=emr, prune_rate=prune_rate, kd_mode=kd_mode, runs=runs, fast=fast, **kwargs)


def feat_exp(model_name, fill_method='zero', prune_rate=0.0, feat_missing=0.995, kd_mode="ikd", runs=5, fast=False, **kwargs):
    missing_exp(model_name, fill_method=fill_method, feat_missing_type="uniform",
                feat_missing=feat_missing, prune_rate=prune_rate, kd_mode=kd_mode, runs=runs, fast=fast, **kwargs)

    missing_exp(model_name, fill_method=fill_method, feat_missing_type="structural",
                feat_missing=feat_missing, prune_rate=prune_rate, kd_mode=kd_mode, runs=runs, fast=fast, **kwargs)


def exp(model_name, fill_method='zero', prune_rate=0.0, feat_missing=0.995, kd_mode="ikd", runs=5, fast=False, **kwargs):
    mix_exp(model_name, fill_method=fill_method, prune_rate=prune_rate, feat_missing=feat_missing, kd_mode=kd_mode,
            runs=runs, fast=fast, **kwargs)

    feat_exp(model_name, fill_method=fill_method, prune_rate=prune_rate, feat_missing=feat_missing, kd_mode=kd_mode, runs=runs, fast=fast, **kwargs)

    missing_exp(model_name, prune_rate=prune_rate, kd_mode=kd_mode, runs=runs, fast=fast, **kwargs)



error_num = None

star = None
beta = None
alpha = None

splits = ["apcfi", "fp", "zero"]
missing_types = ["uniform", "structural"]
sota_types = ["sgformer", "polynormer"]
gnn_types = ["gcn", "sage"]
sota_datasets = ["photo", "computers", "wikics", "cs", "physics"]
datasets = ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]
mix_dataset = ["photo", "computers", "wikics", "physics"]

def run_sota():
    exp("sgformer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5)
    exp("polynormer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5)

    exp("sgformer", fill_method="fp", prune_rate=0.6, kd_mode="ikd", runs=5)
    exp("polynormer", fill_method="fp", prune_rate=0.6, kd_mode="ikd", runs=5, fast=True)

    #exp("polynormer", fill_method="fp", prune_rate=0.6, kd_mode="ikd", runs=5, fast=True, g_epochs=-1)
    #exp("polynormer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, g_epochs=-1)

if __name__ == "__main__":
    #run_sota()
    feat_exp("gcn", fill_method="fp_learning", prune_rate=0.0, feat_missing=0.9, kd_mode="ikd", runs=5, fast=2)
    feat_exp("gcn", fill_method="fp", prune_rate=0.0, feat_missing=0.9, kd_mode="ikd", runs=5, fast=2)
    #run_sota()

    #exp("gcn", fill_method="apcfi", prune_rate="auto", kd_mode="ikd", runs=5, fast=False)
    #exp("sage", fill_method="apcfi", prune_rate="auto", kd_mode="ikd", runs=5, fast=False)

    #exp("gcn", fill_method="apcfi", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False)
    #exp("sage", fill_method="apcfi", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False)


    mix_exp("gcn", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False)
    mix_exp("gcn", fill_method="apcfi", prune_rate="auto", kd_mode="ikd", runs=5, fast=False)

    mix_exp("sage", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False)
    mix_exp("sage", fill_method="apcfi", prune_rate="auto", kd_mode="ikd", runs=5, fast=False)

    mix_exp("sgformer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False, full=False)
    mix_exp("polynormer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False, full=False)

    mix_exp("sgformer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False, full=True)
    mix_exp("polynormer", fill_method="fp", prune_rate=0.0, kd_mode="ikd", runs=5, fast=False, full=True)
