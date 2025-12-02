
import gc
from time import perf_counter
from functools import partial

import optuna
import torch

from limelight.api.dataset import load_dataset
from limelight.api.imputer import filling_data
from limelight.utils import fix_seed

from new_train import TrainerCLS, TrainerPrune, TrainerRKD
from new_model import MainModule, mirror_projection

def read_config(data_name, config_root="./new_best_cfg"):
    import yaml, os
    with open(os.path.join(config_root, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def get_config(data_name, config_root="./new_best_cfg"):
    _config = read_config(data_name, config_root)
    config = {"dataset": data_name,
              "seed": _config.get("seed", 42)}

    config.update(_config["dataset"])

    _cfg = _config["gcn"]
    config["weight_decay"] = float(_cfg["weight_decay"])
    config["epochs"] = int(_cfg["epochs"])
    config["lr"] = float(_cfg["lr"])
    return config


def experiment_main(data_name, gnn_type, missing_rate,
                    hidden, dropout, num_layers, norm, residual,
                    prune_type, imp_type,

                    runs=3, amp=True, mpnn_amp=False, cugraph=False,
                    optim="Adam", gpu_idx=0,
                    ):

    de_config = get_config(data_name)

    return experiment_loop(gnn_type=gnn_type, hidden=hidden, dropout=dropout, num_layers=num_layers,
                           norm=norm, residual=residual, missing_rate=missing_rate,
                           prune_type=prune_type, imp_type=imp_type,

                           runs=runs, amp=amp, mpnn_amp=mpnn_amp,
                           cugraph=cugraph, optim=optim, gpu_idx=gpu_idx, **de_config)



def experiment_loop(seed, dataset, split,
                    missing_rate, optim,

                    gnn_type, hidden, num_layers, dropout, residual, norm,
                    epochs, lr, weight_decay, prune_type, imp_type, purne_rate=0.5,
                    label_num_per_class=20, valid_num=500, test_num=1000,
                    runs=3, amp=True, mpnn_amp=False, cugraph=False, gpu_idx=0, **kwargs):

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
                                               gnn_type, hidden, num_layers, dropout, residual, norm,
                                               epochs, lr, weight_decay,
                                               prune_type, imp_type, purne_rate,

                                               amp, mpnn_amp, cugraph, gpu_idx)

        exp_acc.append(best_acc)
        exp_time.append(train_time)

        torch.cuda.empty_cache()
        gc.collect()
    return sum(exp_acc) / len(exp_acc), sum(exp_time) / len(exp_time)

def experiment_step(run_idx, seed, dataset, optim,
                    gnn_type, hidden, num_layers, dropout, residual, norm,
                    epochs, lr, weight_decay,
                    prune_type, imp_type, purne_rate=0.5,
                    amp=True, mpnn_amp=False, cugraph=False, gpu_idx=0):

    run_seed = seed * (run_idx + 1)
    fix_seed(run_seed)
    start = perf_counter()

    model = MainModule(dataset.x.size(1), hidden, dataset.num_classes, num_layers, cugraph=cugraph,
                       dropout=dropout, res=residual, norm=norm, gnn_type=gnn_type, gnn_cls=False)

    gnnmlp = mirror_projection(model, gnn2mlp=True, cugraph=cugraph)
    prune_trainer = TrainerPrune(args=None, model=gnnmlp,
                         epochs=200, lr=lr, weight_decay=weight_decay,
                         early_stop=-1,
                         mpp_step=1, prune_rate=purne_rate,
                         prune_type=prune_type, imp_type=imp_type, isomorphic=True, global_pruning=True,
                         optim=optim, gpu_idx=gpu_idx,
                         amp=amp, device=None, _print=False)

    prune_trainer.fit(dataset)

    model_pruned = mirror_projection(prune_trainer.model, gnn2mlp=False, cugraph=cugraph)

    gnn_trainer = TrainerCLS(args=None, model=model_pruned,
                       epochs=epochs, lr=lr, weight_decay=weight_decay,
                       early_stop=-1, optim=optim, gpu_idx=gpu_idx,
                       amp=mpnn_amp, device=None, _print=True)

    gnn_trainer.fit(dataset)

    train_time = perf_counter() - start
    best_acc = gnn_trainer._best_metrics_test
    return best_acc, train_time


def optuna_objective(trial, dataset_name, model_name, runs, gpu_idx):
    missing_rate = 0.0

    # ---- 常見連續 / 離散參數 ----
    hidden_power = trial.suggest_int("hidden_power", 6, 9)
    hidden = 2 ** hidden_power
    num_layers = trial.suggest_int("num_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.1)
    norm = trial.suggest_categorical("norm", ["layer", "batch", None])
    residual = trial.suggest_categorical("residual", [True, False])

    prune_type = trial.suggest_categorical("prune_type", ["base", "group", "bnscale"])
    imp_type = trial.suggest_categorical("imp_type", ["group", "lamp", "fpgm"])


    avg_acc, avg_time = experiment_main(dataset_name, model_name, missing_rate=missing_rate,
                                        hidden=hidden, dropout=dropout, num_layers=num_layers,
                                        norm=norm, residual=residual,prune_type=prune_type,imp_type=imp_type,

                                        runs=runs, amp=True, mpnn_amp=False, cugraph=False,
                                        optim="Adam", gpu_idx=gpu_idx

                                        )

    print(f"\nTrial {trial.number} finished with accuracy: {avg_acc:.4f} and time: {avg_time:.4f}s\n")
    return avg_acc


def optuna_optimize(dataset_name, gnn_type, runs=3, n_trials=50, gpu_idx=0):
    import os
    root = "new_optuna_model"
    os.makedirs(os.path.join(root, "db", dataset_name.lower()), exist_ok=True)
    os.makedirs(os.path.join(root, "csv", dataset_name.lower()), exist_ok=True)

    study = optuna.create_study(
        study_name=f"{dataset_name.lower()}_{gnn_type.lower()}",
        directions=["maximize"],
        storage=f"sqlite:///./{root}/db/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.db",  # 本地儲存路徑
        load_if_exists=True  # 避免重複建立新 study
    )
    if os.path.exists(f"./{root}/csv/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv"):
        print(f"File {dataset_name.lower()} Done. Skipping optimization.")
        return

    study.optimize(partial(optuna_objective, dataset_name=dataset_name,
                           model_name=gnn_type, runs=runs, gpu_idx=gpu_idx), n_trials=n_trials)

    df = study.trials_dataframe()
    df.to_csv(f"./{root}/csv/{dataset_name.lower()}/{gnn_type.lower()}_{n_trials}.csv", index=False)

    pareto_trials = study.best_trials
    for t in pareto_trials:
        print(f"Trial#{t.number}, f1={t.values[0] * 100 :.2f}, params={t.params}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument("--gnn", choices=["gin", "cheb"], type=str, default="gcn")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    for dataset in ["cora", "citeseer", "pubmed", "photo", "computers", "cs", "physics", "wikics"]:
        optuna_optimize(dataset_name=dataset, gnn_type=args.gnn, runs=3, n_trials=100, gpu_idx=args.device)
