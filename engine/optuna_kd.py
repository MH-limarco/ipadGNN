
from functools import partial

import torch
from scipy.signal import ellip

from limelight.utils import fix_seed
from limelight.api.dataset import load_dataset

from engine.moudle.model import LimelightModule, mirror_projection
from engine.trainer import TrainerPrune, TrainerCLS, TrainerRKD

sorted_keys = ["prune_rate", "momentum", "momentum", "loss_type",
               "kd_lr", "kd_weight_decay", "lamb", "alpha",
               "kd_warmup", "kd_scheduler", "kd_T_0", "label_smoothing"]


def _read_config(data_name, dir_root, config_root="./best_config"):
    import yaml, os
    with open(os.path.join(config_root, dir_root, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def check_nan(value):
    import math
    return value is not None and not (isinstance(value, float) and math.isnan(value))

def _read_best_config(data_name, model_type, prune_rate=0.0):
    import math
    gnn_dir = "gnns/prune_cfg" if prune_rate > 0 else "gnns/new_best_cfg"
    gnn_config = _read_config(data_name, gnn_dir)

    config = {"seed": gnn_config.get("seed", 42)}

    if prune_rate > 0:
        _gnn_config = gnn_config[model_type.lower()]
        config.update(_gnn_config[prune_rate])
        _gnn_config.pop(0.5)
        _gnn_config.pop(0.6)
        _gnn_config.pop(0.7)
        config.update(_gnn_config)
        config["prune_epochs"] = int(150 * config["prune_step"])

    else:
        config.update(gnn_config[model_type.lower()])

    config["kd_epochs"] = int(min(config["epochs"] * 2, 2000))
    config["prune_step"] = int(config["prune_step"])

    config = {k: float(v) if "lr" in k and check_nan(v) else v for k, v in config.items()}
    config = {k: float(v) if "weight_decay" in k and check_nan(v) else v for k, v in config.items()}
    config = {k: int(v) if "epochs" in k and check_nan(v) else v for k, v in config.items()}
    config = {k: int(v) if "T_0" in k and check_nan(v) else v for k, v in config.items()}
    return config

def read_config(data_name, model_type, prune_rate):
    config = _read_best_config(data_name, model_type, prune_rate)
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

def train_gnn(config, dataset, model):
    gnn_trainer = TrainerCLS(args=None, model=model,
                              epochs=config["epochs"], lr=config["lr"],
                              weight_decay=config["weight_decay"],
                              scheduler=config["scheduler"], warmup=config["warmup"],
                              T_0=config["T_0"],
                              early_stop=-1, amp=False, gpu_idx=0, _print=True)
    gnn_trainer.fit(dataset)
    return gnn_trainer

def train_kd(config, dataset, teacher, student):
    kd_trainer = TrainerRKD(args=None, teacher=teacher, student=student,
                             epochs=config["kd_epochs"], momentum=config["momentum"],
                             lamb=config["lamb"], alpha=config["alpha"],
                             feat_loss=config["feat_loss"], temperature=config["temperature"],
                             lr=config["kd_lr"], weight_decay=config["kd_weight_decay"],
                             scheduler=config["kd_scheduler"], warmup=config["kd_warmup"],
                             T_0=config["kd_T_0"], label_smoothing=config["label_smoothing"],
                             early_stop=500, amp=False, gpu_idx=0, _print=True)
    kd_trainer.fit(dataset)
    return kd_trainer


def experiment_one_step(config, dataset, gnn_type, data_name, seed):
    import os
    imlp_root = f"./save/prune_mlp/{data_name}"
    os.makedirs(imlp_root, exist_ok=True)
    imlp_save_path = f"{imlp_root}/best_seed_{seed}_{data_name}_{gnn_type}_{config['prune_rate']}.pt"

    if os.path.exists(imlp_save_path):
        trained_imlp = LimelightModule(pt=imlp_save_path)


    else:
        imlp = LimelightModule(dataset.x.size(1), config["hidden"], dataset.num_classes, config["num_layers"],
                               dropout=config["dropout"], res=config["residual"], norm=config["norm"],
                               gnn_type=f"{gnn_type}_mlp", gnn_cls=False)

        imlp_trainer = train_imlp(config, dataset, imlp)

        imlp_trainer.model.save(imlp_save_path)
        trained_imlp = imlp_trainer.model

    gnn_root = f"./save/prune_gnn/{data_name}"
    os.makedirs(gnn_root, exist_ok=True)
    gnn_save_path = f"{gnn_root}/best_seed_{seed}_{data_name}_{gnn_type}_{config['prune_rate']}.pt"
    if os.path.exists(gnn_save_path):
        trained_gnn = LimelightModule(pt=gnn_save_path)
        trained_gnn._CuGraph = config["cugraph"]
        gnn_trainer = TrainerCLS(args=None, model=trained_gnn,
                                 epochs=config["epochs"], lr=config["lr"],
                                 early_stop=-1, amp=False, gpu_idx=0, _print=True)

        if not config["cugraph"]:
            _, _, gnn_test = gnn_trainer.test(dataset)
            print(f"GNN-Best acc: {gnn_test:.4f}")

    else:
        pruned_gnn = mirror_projection(trained_imlp, target="gnn", cugraph=config["cugraph"])
        gnn_trainer = train_gnn(config, dataset, pruned_gnn)
        gnn_trainer.model.save(gnn_save_path)
        trained_gnn = gnn_trainer.model

    if config["reset"]:
        trained_imlp.reset_parameters()
    kd_trainer = train_kd(config, dataset, teacher=trained_gnn, student=trained_imlp)
    metrics_train, metrics_val, metrics_test = kd_trainer.test(dataset)
    return metrics_test

def experiment(optuna_config, data_name, gnn_type, runs=3):
    import gc
    split = "fixed" if data_name not in ["cora", "citeseer", "pubmed"] else "class"
    config = read_config(data_name, gnn_type, optuna_config["prune_rate"])
    config.update(optuna_config)

    test_acc = []

    fix_seed(config["seed"])
    dataset, _ = load_dataset(data_name, split, missing_rate=config["missing_rate"], label_num_per_class=20)

    for run in range(runs):
        _seed = config["seed"] * (run + 1)
        fix_seed(_seed)
        metrics_test = experiment_one_step(config, dataset, gnn_type, data_name, _seed)
        test_acc.append(metrics_test)

        torch.cuda.empty_cache()
        gc.collect()

    return sum(test_acc) / len(test_acc)

def optuna_objective(trial, dataset_name, gnn_type, feat, runs):
    args = {"missing_rate": 0.0}

    #gnn
    args["prune_rate"] = float(trial.suggest_float("prune_rate", 0.5, 0.7, step=0.1))

    #kd
    args["reset"] = trial.suggest_categorical("reset", [False, True])
    args["momentum"] = float(trial.suggest_categorical("momentum", [0.995, 0.99, 0.9, 0.8]))
    args["feat_loss"] = trial.suggest_categorical("feat_loss", ["mse", "cwd"]) if feat else None
    #args["feat_loss"] = "cwd" if feat else None
    args["kd_lr"] = float(trial.suggest_float("kd_lr", 1e-5, 1e-2, log=True))
    args["kd_weight_decay"] = float(trial.suggest_float("kd_weight_decay", 1e-5, 1e-0, log=True))

    if args["feat_loss"] is not None:
        args["lamb"] = float(trial.suggest_float("lamb", 0.0, 5, step=0.25))
        args["alpha"] = float(trial.suggest_float("alpha", 0.0, 0.5, step=0.25))

    else:
        args["lamb"] = float(trial.suggest_float("lamb", 0.1, 0.9, step=0.1))
        args["alpha"] = None

    if args["feat_loss"] == "cwd":
        args["temperature"] = float(trial.suggest_float("temperature", 0.25, 5.0, step=0.25))
    else:
        args["temperature"] = None

    args["kd_warmup"] = float(trial.suggest_float("kd_warmup", 0.0, 0.15, step=0.05))
    args["kd_scheduler"] = trial.suggest_categorical("kd_scheduler", [True, False])
    args["kd_T_0"] = int(trial.suggest_int("kd_T_0", 30, 150, step=30)) if args["kd_scheduler"] else None
    args["label_smoothing"] = float(trial.suggest_float("label_smoothing", 0.0, 0.3, step=0.05))

    args["cugraph"] = True if dataset_name.lower() in ["cs", "physics"] and gnn_type == "sage" else False

    avg_acc = experiment(args, dataset_name, gnn_type, runs=runs)

    print(f"\nTrial {trial.number} finished with {args['prune_rate']:.2f}-prune gnn accuracy: {avg_acc:.4f}\n")

    return avg_acc

def optuna_optimize(dataset_name, gnn_type, feat=False, runs=3, n_trials=50):
    import optuna
    import os

    root = "optuna_kd" + ('' if not feat else "_feat")
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "db", dataset_name.lower()), exist_ok=True)
    os.makedirs(os.path.join(root, "csv", dataset_name.lower()), exist_ok=True)

    if os.path.exists(f"./{root}/csv/{dataset_name.lower()}/{dataset_name.lower()}_{gnn_type.lower()}_{n_trials}.csv"):
        print(f"File: {dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv already exists. Skipping optimization.")
        return

    study = optuna.create_study(
        study_name=f"{dataset_name.lower()}_{gnn_type.lower()}",
        directions=["maximize"],
        storage=f"sqlite:///./{root}/db/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.db",  # 本地儲存路徑
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=10,
            reduction_factor=3,
            min_early_stopping_rate=1
        ), load_if_exists=True)  # 避免重複建立新 study)

    trial = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if trial < n_trials:
        try:
            study.optimize(partial(optuna_objective, dataset_name=dataset_name,
                                   gnn_type=gnn_type, feat=feat, runs=runs), n_trials=n_trials)
        except:
            print(f"Trial {study.trials} failed. Retrying...")
            study.optimize(partial(optuna_objective, dataset_name=dataset_name,
                                    gnn_type=gnn_type, feat=feat, runs=runs), n_trials=n_trials)

    trial = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if trial >= n_trials:
        df = study.trials_dataframe()
        _sorted_keys = [f"params_{i}" for i in sorted_keys if f"params_{i}" in df.columns]
        remaining_cols = [col for col in df.columns if col not in _sorted_keys]
        final_cols = remaining_cols + _sorted_keys

        df_reordered = df[final_cols]
        second_col_name = df_reordered.columns[1]
        df_sorted = df_reordered.sort_values(by=second_col_name, ascending=False)
        df_sorted.to_csv(f"./{root}/csv/{dataset_name.lower()}/{dataset_name.lower()}_{gnn_type.lower()}_{n_trials}.csv", index=False)

        #build_best_prune_yaml(df_sorted, dataset_name, gnn_type)

def build_best_prune_yaml(df, data_name, gnn_type):
    from collections import OrderedDict
    import yaml, os, re
    params_columns = [col for col in df.columns if col.startswith("params_")]
    best_combinations = df.loc[df.groupby("params_prune_rate")["values_1"].idxmax(), params_columns]
    best_combinations.columns = [col.replace("params_", "") for col in best_combinations.columns]
    best_combinations.reset_index(drop=True, inplace=True)
    prune_rate_dict = best_combinations.set_index("prune_rate").to_dict(orient="index")

    gnn_config = _read_config(data_name, "gnns/new_best_cfg")

    gnn_config[gnn_type.lower()].pop("lr", None)
    gnn_config[gnn_type.lower()].pop("weight_decay", None)

    gnn_config[gnn_type.lower()].update(prune_rate_dict)

    gnn_path = os.path.join("save_log/best_config", "gnns", "prune_cfg", f"{data_name.lower()}.yaml")

    def round_floats(d):
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

    round_floats(prune_rate_dict)

    ordered_config = OrderedDict()
    if "dataset" in gnn_config:
        ordered_config["seed"] = gnn_config.pop("seed")
        ordered_config["dataset"] = gnn_config.pop("dataset")

    ordered_config.update(gnn_config)

    class CustomDumper(yaml.SafeDumper):
        pass

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    CustomDumper.add_representer(OrderedDict, dict_representer)

    with open(gnn_path, "w+") as f:
        yaml.dump(ordered_config, f, Dumper=CustomDumper, default_flow_style=False, indent=4)

    print(f"{data_name}_{gnn_type}YAML 已更新！")


if __name__ == "__main__":
    for dataset in ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]:  # [::-1]:
        for conv_name in ["sage", "gcn"]:
            optuna_optimize(dataset_name=dataset, gnn_type=conv_name, feat=True, runs=3, n_trials=125)








