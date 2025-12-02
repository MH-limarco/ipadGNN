import os

import optuna

sorted_keys = ["prune_rate", "momentum", "momentum", "loss_type",
               "kd_lr", "kd_weight_decay", "lamb", "alpha",
               "kd_warmup", "kd_scheduler", "kd_T_0", "label_smoothing"]


def optuna_optimize(dataset_name, gnn_type, feat=False, runs=3, n_trials=100):


    root = "optuna_apcfi_full_structural" # + ('' if not feat else "_feat")
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "db", dataset_name.lower()), exist_ok=True)
    os.makedirs(os.path.join(root, "csv", dataset_name.lower()), exist_ok=True)

    study = optuna.load_study(
        study_name=f"{dataset_name.lower()}_{gnn_type.lower()}_0.995",
        storage=f"sqlite:///./{root}/db/0.995/{dataset_name.lower()}_{gnn_type.lower()}_{n_trials}.db",  # 本地儲存路徑
        )
    print(f"Study loaded successfully!")
    #fig_importance = optuna.visualization.plot_param_importances(study)
    #fig_importance.show()

    key_params_to_plot = ['alpha', 'beta']
    #fig_slice = optuna.visualization.plot_slice(study, params=key_params_to_plot)
    #fig_slice.show()

    fig_contour = optuna.visualization.plot_contour(study, key_params_to_plot)
    fig_contour.write_image(f"./{dataset_name}_apcfi_contour_plot.png")
                                                    #fig_slice.write_image(f"./{study_name}_slice_plot.png")
    print("Best hyperparameters: ")


if __name__ == "__main__":
    for dataset in ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]:  # [::-1]:
        for conv_name in ["sage", "gcn"]:
            optuna_optimize(dataset_name=dataset, gnn_type=conv_name, feat=True, runs=3, n_trials=100)
            break
        break
