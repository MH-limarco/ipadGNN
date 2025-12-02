import os
import re
import pandas as pd
import numpy as np

DATASETS = [
    'citeseer', 'computers', 'cora', 'cs', 'photo', 'physics', 'pubmed', 'wikics'
]
SRC_MODELS = ['gcn', 'sage']
MISSING_RATES = [0.0, 0.3, 0.6, 0.9, 0.995]
COL_NAMES = ['full_feat', '30% missing', '60% missing', '90% missing', '99.5% missing']

BASELINE_DIR = 'baseline_exp'
EDGE_MISSING_DIR = 'missing_exp/uniform_missing'

CSV_DIR = "tuned_eff_u"

#os.makedirs(f"./_csv/{CSV_DIR}")

def parse_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        line = f.read().strip()
    parts = re.split(r'[,\s]+', line)
    kd_acc = float(parts[0]) if parts[0] != 'nan' else None
    gnn_acc = float(parts[1]) if parts[1] != 'nan' else None
    return kd_acc, gnn_acc

def get_missing_rate(fname):
    m = re.search(r'[_\-]?([0-9.]+)', fname)
    if m:
        return float(m.group(1)[:-1] if m.group(1).endswith(".") else m.group(1))
    return 0.0

def get_src_model(fname):
    for m in SRC_MODELS:
        if m in fname:
            return m
    return None

def get_model_type(path, fname, kd_acc):
    if 'classic' in path or 'classic' in fname:
        return 'classic_gnn'
    if kd_acc is not None and kd_acc > 0.0:
        return 'PRISM'
    return 'truned_gnn(mpp)'

for dataset in DATASETS:
    data_dict = {}
    # baseline_exp (full edge)
    for root, dirs, files in os.walk(BASELINE_DIR):
        for file in files:
            if file.endswith('.txt') and dataset in file:
                fpath = os.path.join(root, file)
                kd_acc, gnn_acc = parse_txt(fpath)
                src_model = get_src_model(file)
                model_type = get_model_type(root, file, kd_acc)
                model_name = f"{model_type}({src_model})"

                if model_name not in data_dict:
                    data_dict[model_name] = [np.nan, np.nan, np.nan, np.nan, np.nan]

                if model_type == 'classic_gnn':
                    data_dict[model_name][0] = gnn_acc
                elif model_type == 'PRISM':
                    data_dict[model_name][0] = kd_acc
                    data_dict[f"truned_gnn(mpp)({src_model})"][0] = gnn_acc

    # edge_missing
    for root, dirs, files in os.walk(EDGE_MISSING_DIR):
        for file in files:
            if "zero_" not in root:
                continue

            if file.endswith('.txt') and dataset in file:
                fpath = os.path.join(root, file)
                missing_rate = get_missing_rate(file)
                if missing_rate not in MISSING_RATES[1:]:
                    continue

                col_idx = MISSING_RATES[1:].index(missing_rate) + 1
                kd_acc, gnn_acc = parse_txt(fpath)


                src_model = get_src_model(file)
                model_type = get_model_type(root, file, kd_acc)
                model_name = f"{model_type}({src_model})"
                if model_name not in data_dict:
                    data_dict[model_name] = [np.nan, np.nan, np.nan, np.nan, np.nan]

                if model_type == 'classic_gnn':
                    data_dict[model_name][col_idx] = gnn_acc
                elif model_type == 'PRISM':
                    data_dict[model_name][col_idx] = kd_acc
                    data_dict[f"truned_gnn(mpp)({src_model})"][col_idx] = gnn_acc

    # 產生 DataFrame
    models = sorted(data_dict.keys())
    rows = []
    for m in models:
        rows.append([m] + data_dict[m])
    df = pd.DataFrame(rows, columns=['model'] + COL_NAMES)
    os.makedirs(f"./_csv/{CSV_DIR}", exist_ok=True)
    df.to_csv(f"./_csv/{CSV_DIR}/{dataset}.csv", index=False)
    print(f"已產生：{dataset}.csv")
