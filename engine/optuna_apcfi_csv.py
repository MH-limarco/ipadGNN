import os

sorted_keys = ["alpha", "beta", "star"]

def convert_nan_to_none(data):
    import math
    """
    Recursively converts float('nan') values to None in dictionaries and lists.
    """
    if isinstance(data, dict):
        return {k: convert_nan_to_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_nan_to_none(elem) for elem in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    else:
        return data


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


def _read_df(dir_roots, data_name, gnn_type, runs=125):
    import pandas as pd

    dfs = []
    for root in dir_roots:
        df = pd.read_csv(f"./{root}/csv/{data_name}/{data_name}_{gnn_type}_{runs}.csv")
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["number"] + [c for c in df.columns if "." in c], inplace=True)
    df_sorted = df.sort_values(by=df.columns[0], ascending=False)
    return df_sorted

def _read_yaml(dir_root, data_name, config_root):
    import yaml, os
    with open(os.path.join(config_root, dir_root, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def mixup_csv(data_name, model, runs=100):
    import yaml
    roots = ["optuna_apcfi"]
    missing = [0.3, 0.6, 0.9, 0.995]
    yaml_root = "./best_config"

    cfg_path = os.path.join(yaml_root, "apcfi")
    os.makedirs(cfg_path, exist_ok=True)
    gnn_path = os.path.join(cfg_path, f"{data_name}.yaml")

    df = _read_df(roots, data_name, model, runs)

    df_col = [c for c in df.columns if "params_" in c]
    df = df[df_col].iloc[0:1]
    df.columns = [c.replace("params_", "") for c in df.columns]
    prune_rate = df["prune_rate"].iloc[0]

    gnn_dir = "gnns/prune_cfg" if prune_rate > 0 else "gnns/new_best_cfg"
    cfg = _read_yaml(gnn_dir, data_name, yaml_root)


    model_cfg = cfg.pop(model)
    model_cfg.update(model_cfg[prune_rate])
    model_cfg = {k:v for k,v in model_cfg.items() if not isinstance(v, dict)}

    idk_cfg = round_floats(df.iloc[0].to_dict())
    model_cfg.update(idk_cfg)

    cfg[model] = model_cfg

    if os.path.exists(gnn_path):
        org_cfg = yaml.safe_load(open(gnn_path, 'r', encoding="utf-8"))
        org_cfg.pop(model)
        cfg.update(org_cfg)

    cfg = convert_nan_to_none(cfg)
    with open(gnn_path, 'w', encoding="utf-8") as file:
        yaml.dump(cfg, file, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    runs = 125
    for dataset in ["cora", "citeseer", "pubmed","photo", "computers", "wikics", "cs", "physics"]:
        for m in ["sage", "gcn"]:
            build_ikd_yaml(dataset, m, runs)