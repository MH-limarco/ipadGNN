
from functools import partial

import torch
from limelight.utils import fix_seed
from limelight.api.dataset import load_dataset

from engine.moudle.model import LimelightModule, mirror_projection
from engine.trainer import TrainerPrune, TrainerCLS

sorted_keys = ["prune_rate", "lr", "weight_decay", "warmup", "scheduler", "T_0",
               "prune_lr", "prune_weight_decay", "prune_step", "prune_epochs",
               "prune_scheduler", "prune_T_0", "prune_warmup",
               "pruner_type", "imp_type", "isomorphic"]



def _read_config(data_name, dir, config_root="./best_config"):
    import yaml, os
    with open(os.path.join(config_root, dir, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def _read_best_config(data_name, model_type, prune_rate=0.0):
    gnn_dir = "gnns/new_best_prune_cfg" if prune_rate > 0 else "gnns/new_best_cfg"
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

    config = {k: float(v) if "lr" in k else v for k, v in config.items()}
    config = {k: float(v) if "weight_decay" in k else v for k, v in config.items()}
    config = {k: int(v) if "epochs" in k else v for k, v in config.items()}
    return config

def read_config(data_name, model_type):
    config = _read_best_config(data_name, model_type, 0.0)
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


def experiment_one_step(config, dataset, gnn_type):
    imlp = LimelightModule(dataset.x.size(1), config["hidden"], dataset.num_classes, config["num_layers"],
                           dropout=config["dropout"], res=config["residual"], norm=config["norm"],
                           gnn_type=f"{gnn_type}_mlp", gnn_cls=False)

    imlp_trainer = train_imlp(config, dataset, imlp)
    trained_imlp = imlp_trainer.model


    pruned_gnn = mirror_projection(trained_imlp, target="gnn", cugraph=config["cugraph"])
    gnn_trainer = train_gnn(config, dataset, pruned_gnn)

    _, _, metrics_test = gnn_trainer.test(dataset)
    print("Test accuracy:", metrics_test)
    return metrics_test

def experiment(optuna_config, data_name, gnn_type, runs=3):
    import gc
    split = "fixed" if data_name not in ["cora", "citeseer", "pubmed"] else "class"
    config = read_config(data_name, gnn_type)
    config.update(optuna_config)

    test_acc = []

    fix_seed(config["seed"])
    dataset, _ = load_dataset(data_name, split, missing_rate=config["missing_rate"], label_num_per_class=20)

    for run in range(runs):
        _seed = config["seed"] * (run + 1)
        fix_seed(_seed)
        metrics_test = experiment_one_step(config, dataset, gnn_type)
        test_acc.append(metrics_test)

        torch.cuda.empty_cache()
        gc.collect()

    return sum(test_acc) / len(test_acc)

def optuna_objective(trial, dataset_name, gnn_type, runs):
    args = {"missing_rate": 0.0}

    #gnn
    args["prune_rate"] = float(trial.suggest_float("prune_rate", 0.5, 0.7, step=0.1))
    args["lr"] = float(trial.suggest_float("lr", 1e-5, 1e-2, log=True))
    args["weight_decay"] = float(trial.suggest_float("weight_decay", 1e-5, 1e-0, log=True))
    args["warmup"] = float(trial.suggest_float("warmup", 0.0, 0.15, step=0.05))
    args["scheduler"] = trial.suggest_categorical("scheduler", [True, False])
    args["T_0"] = int(trial.suggest_int("T_0", 30, 150, step=30)) if args["scheduler"] else None

    #prune
    args["prune_lr"] = float(trial.suggest_float("prune_lr", 1e-5, 1e-2, log=True))
    args["prune_weight_decay"] = float(trial.suggest_float("prune_weight_decay", 1e-5, 1e-0, log=True))
    args["prune_step"] = int(trial.suggest_int("prune_step", 1, 3))
    args["prune_epochs"] = round(300 / args["prune_step"])

    args["prune_scheduler"] = trial.suggest_categorical("prune_scheduler", [True, False])
    args["prune_T_0"] = int(trial.suggest_int("prune_T_0", 30, 150, step=30)) if args["prune_scheduler"] else None
    args["prune_warmup"] = float(trial.suggest_float("prune_warmup", 0.0, 0.15, step=0.05))

    args["pruner_type"] = trial.suggest_categorical("pruner_type", ["base", "group", "bnscale"])
    args["imp_type"] = trial.suggest_categorical("imp_type", ["group", "lamp", "fpgm"])
    args["isomorphic"] = trial.suggest_categorical("isomorphic", [True, False])


    args["cugraph"] = True if dataset_name.lower() in ["cs", "physics"] and gnn_type == "sage" \
                      else False

    avg_acc = experiment(args, dataset_name, gnn_type, runs=runs)

    print(f"\nTrial {trial.number} finished with {args['prune_rate']:.2f}-prune gnn accuracy: {avg_acc:.4f}\n")

    return avg_acc, args["prune_rate"]

def optuna_optimize(dataset_name, gnn_type, runs=3, n_trials=50):
    import optuna
    import os
    root = "optuna_pruner"
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "db", dataset_name.lower()), exist_ok=True)
    os.makedirs(os.path.join(root, "csv", dataset_name.lower()), exist_ok=True)

    if os.path.exists(f"./{root}/csv/{dataset_name.lower()}/{dataset_name.lower()}_{gnn_type.lower()}_{n_trials}.csv"):
        print(f"File: {dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv already exists. Skipping optimization.")
        return

    study = optuna.create_study(
        study_name=f"{dataset_name.lower()}_{gnn_type.lower()}",
        directions=["maximize", "maximize"],
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
                                   gnn_type=gnn_type, runs=runs), n_trials=n_trials)
        except:
            print(f"Trial {study.trials} failed. Retrying...")
            study.optimize(partial(optuna_objective, dataset_name=dataset_name,
                                    gnn_type=gnn_type, runs=runs), n_trials=n_trials)

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

if __name__ == "__main__":
    for dataset in ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]:  # [::-1]:
        for conv_name in ["sage", "gcn"]:
            optuna_optimize(dataset_name=dataset, gnn_type=conv_name, runs=3, n_trials=100)








