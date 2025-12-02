
from functools import partial
from copy import deepcopy
import os

import torch
from limelight.utils import fix_seed
from limelight.api.dataset import load_dataset
from limelight.api.imputer import filling_data

from engine.moudle.model import LimelightModule, mirror_projection
from engine.trainer import TrainerPrune, TrainerCLS
from main_utils import train_flow, check_nan

sorted_keys = []

def _read_cfg(data_name, root, path):
    import yaml, os
    with open(os.path.join(root, path, f"{data_name}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def experiment(optuna_config, config, data_name, gnn_type, mode, missing_rate, runs=3, load=True):
    import gc
    _config = deepcopy(config)
    model_config = _config[gnn_type]

    model_config["cugraph"] = True if gnn_type == "sage" and data_name in ["cs", "physics"] else False
    if mode != "full":
        model_config["kd_epochs"] = int(min(model_config["epochs"] * 2, 2000))
        model_config["prune_epochs"] = int(150 * model_config["prune_step"])
        model_config["prune_step"] = int(model_config["prune_step"])

    model_config = {k: float(v) if "lr" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: float(v) if "weight_decay" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: int(v) if "epochs" in k and check_nan(v) else v for k, v in model_config.items()}
    model_config = {k: int(v) if "T_0" in k and check_nan(v) else v for k, v in model_config.items()}

    _config["dataset"]["split_method"] = _config["dataset"].pop("split")
    fix_seed(_config["seed"])
    dataset, _ = load_dataset(data_name, **_config["dataset"], missing_rate=missing_rate, missing_type=missing_type)
    if missing_rate > 0.0:
        dataset = filling_data(dataset, "apcfi", **optuna_config)

    kd_acc, gnn_acc = [], []
    for run in range(runs):
        _seed = config["seed"] * (run + 1)
        fix_seed(_seed)
        if mode == "full":
            pass
            #kd_test, gnn_test = experiment_full_one_step(model_config, dataset, gnn_type, data_name, mode, _seed)
        else:
            kd_test, gnn_test = train_flow(model_config, dataset, gnn_type, data_name, _seed, 500,load=load)
            kd_acc.append(kd_test)
            gnn_acc.append(gnn_test)

        torch.cuda.empty_cache()
        gc.collect()

    return sum(kd_acc) / len(kd_acc), sum(gnn_acc) / len(gnn_acc)

def optuna_objective(trial, config, dataset_name, gnn_type, mode, missing_rate, load, runs):
    args = {}
    _config = deepcopy(config)
    args["alpha"] = float(trial.suggest_float("alpha", 0.1, 0.9, step=0.05))
    args["beta"] = float(trial.suggest_float("beta", 1e-6, 1e-0, log=True))
    args["star"] = trial.suggest_categorical("star", [True, False])
    args["batch"] = 1024 if dataset_name == "physics" else None

    kd_acc, gnn_acc = experiment(args, _config, dataset_name, gnn_type, mode=mode,
                                 missing_rate=missing_rate, runs=runs, load=load)
    return kd_acc

def optuna_optimize(config, dataset_name, gnn_type, mode, missing_rate, runs=3, n_trials=50):
    import optuna
    import os
    root = f"optuna_apcfi_full_{missing_type}"
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "db", f"{missing_rate}"), exist_ok=True)
    os.makedirs(os.path.join(root, "csv", f"{missing_rate}"), exist_ok=True)

    study = optuna.create_study(
        study_name=f"{dataset_name.lower()}_{gnn_type.lower()}_{missing_rate}",
        directions=["maximize"],
        storage=f"sqlite:///./{root}/db/{missing_rate}/{dataset_name.lower()}_{gnn_type.lower()}_{n_trials}.db",  # 本地儲存路徑
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=10,
            reduction_factor=3,
            min_early_stopping_rate=1
        ), load_if_exists=True)  # 避免重複建立新 study)

    trial = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if trial < n_trials:
        print(f"{trial} / {n_trials}")
        try:
            study.optimize(partial(optuna_objective, mode=mode, config=config, dataset_name=dataset_name, load=False,
                                   gnn_type=gnn_type, missing_rate=missing_rate, runs=runs), n_trials=n_trials)
        except:
            print(f"Trial {study.trials} failed. Retrying...")
            study.optimize(partial(optuna_objective, mode=mode, config=config, dataset_name=dataset_name, load=False,
                                   gnn_type=gnn_type, missing_rate=missing_rate, runs=runs), n_trials=n_trials)

    trial = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if trial >= n_trials:
        df = study.trials_dataframe()
        _sorted_keys = [f"params_{i}" for i in sorted_keys if f"params_{i}" in df.columns]
        remaining_cols = [col for col in df.columns if col not in _sorted_keys]
        final_cols = remaining_cols + _sorted_keys

        df_reordered = df[final_cols]
        second_col_name = df_reordered.columns[1]
        df_sorted = df_reordered.sort_values(by=second_col_name, ascending=False)
        df_sorted.to_csv(
            f"./{root}/csv/{missing_rate}/{dataset_name.lower()}_{gnn_type.lower()}_{n_trials}.csv",
            index=False)


if __name__ == "__main__":
    mode = "ikd"
    missing_type = "structural" #uniform #structural
    use_imlp = True
    for data_name in ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]: #"physics"]:  # [::-1]:
        for gnn_type in ["gcn", "sage"]:
            for mr in [0.3, 0.6, 0.9, 0.995]:
                root = os.path.join("./", "best_config", "gnns")
                path = "kd_cfg" if mode == "kd" or (
                        'kd' in mode and not use_imlp) else "ikd_cfg" if mode != "full" else "best_cfg"
                cfg = _read_cfg(data_name, root, path)
                n_trials = 50 if data_name == "physics" and gnn_type == "sage" else 75 if data_name == "physics" else 100
                optuna_optimize(cfg, data_name, gnn_type, mode, missing_rate=mr, runs=3, n_trials=n_trials)
