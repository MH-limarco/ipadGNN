import os
import re
import pandas as pd
import numpy as np
#'citeseer', ,
DATASETS = [
    'cora', 'citeseer', 'pubmed', 'computers', 'photo', 'wikics', 'cs', 'physics'
]
SRC_MODELS = ['gcn', 'sage']
MISSING_RATES = [0.0, 0.3, 0.6, 0.9]
COL_NAMES = ['full_edge', '30% missing', '60% missing', '90% missing']

#missing_type = "uniform" # uniform, structural


DATA_DIR = f'_csv/edge_missing/dataset'
CSV_DIR = f"edge_missing"


def parse_csv(filepath, target_col):
    _df = pd.read_csv(filepath, index_col=0)
    #_df.index = [f"{imp_type}_{ind}" for ind in _df.index]
    return _df.loc[:, target_col].to_dict()


df = pd.DataFrame(columns=[DATASETS])
for target_mr in MISSING_RATES[1:]:
    target_col_names = COL_NAMES[MISSING_RATES.index(target_mr)]
    for dataset in DATASETS:
        _dict = {}

        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(".csv") and dataset in file:
                    fpath = os.path.join(root, file)
                    #_imp_type = parse_imp(fpath)
                    _value = parse_csv(fpath, target_col_names)
                    _dict.update(_value)

        df[dataset] = pd.Series(_dict)

    os.makedirs(f"./_csv/{CSV_DIR}", exist_ok=True)
    df.to_csv(f"./_csv/{CSV_DIR}/edge_mr_{target_mr}.csv", index=True)
    print(f"已產生：edge_mr_{target_mr}.csv")

