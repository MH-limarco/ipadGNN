import os
import re
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASETS = ["cora", "citeseer", "pubmed", "photo", "computers", "wikics", "cs", "physics"]
SRC_MODELS = ['gcn', 'sage']
_imp = ["zero", "fp", "apcfi"]
MISSING_RATES = [0.0, 0.3, 0.6, 0.9, 0.995]
COL_NAMES = ['full_feat', '30% missing', '60% missing', '90% missing', '99.5% missing']

missing_type = "structural" # uniform, structural
model = "GCN" # GCN, SAGE

CSV_DIR = f"feat_missing_{missing_type}"

file_name = ["apcfi", "fp", "zero"]

offset = 0 if model.lower() == "gcn" else 1

DIR = f'{missing_type}_{model.lower()}'

os.makedirs(f"./_fig/{DIR}", exist_ok=True)
for dataset in DATASETS:
    _apcfi = pd.read_csv(f"./_csv/{CSV_DIR}/dataset/{file_name[0]}_{dataset}.csv")
    _apcfi = _apcfi.iloc[[0 + offset, 2 + offset, 4 + offset]]
    #_apcfi = _apcfi.iloc[[2 + offset]]

    _apcfi['model'] = _apcfi['model'].replace(["PRISM(gcn)", "PRISM(sage)",
                                               "classic_gnn(gcn)", "classic_gnn(sage)",
                                               "truned_gnn(mpp)(gcn)",
                                               "truned_gnn(mpp)(sage)"], ["iPad-GCN", "iPad-SAGE",
                                                                          "GCN w APCFI", "GraphSAGE w APCFI",
                                                                          "MPP-GCN(tuned) w APCFI",
                                                                          "MPP-SAGE(tuned) w APCFI"])


    _fp = pd.read_csv(f"./_csv/{CSV_DIR}/dataset/{file_name[1]}_{dataset}.csv")
    _fp = _fp.iloc[[2 + offset]]
    _fp['model'] = _fp['model'].replace(["classic_gnn(gcn)", "classic_gnn(sage)"], ["GCN w FP", "GraphSAGE w FP"])

    _zero = pd.read_csv(f"./_csv/{CSV_DIR}/dataset/{file_name[2]}_{dataset}.csv")
    _zero = _zero.iloc[[2 + offset]]
    _zero['model'] = _zero['model'].replace(["classic_gnn(gcn)", "classic_gnn(sage)"], ["GCN w Zero", "GraphSAGE w Zero"])


    df = pd.concat([_apcfi, _fp, _zero], ignore_index=True)

    df_long = df.melt(id_vars='model', var_name='missing_type', value_name='score')
    df_long['missing_type'] = df_long['missing_type'].replace(["full_feat", "30% missing" ,'60% missing', '90% missing', '99.5% missing'], [0.0, 0.3, 0.6, 0.9, 0.995])

    df_long['linestyle'] = df_long['model'].apply(lambda x: 'solid' if 'PRISM' in x or "APCFI" in x else 'dashed') #dashed
    plt.figure(figsize=(9,5))

    y_lim = (_fp.iloc[0, -1] // 0.05) * 0.05

    # 畫出每條線
    for idx, (model, group) in enumerate(df_long.groupby('model')):
        style = 'solid' if 'iPad' in model or "APCFI" in model else 'dashed'
        plt.plot(
                group['missing_type'], group['score'],
                label=model,
                linestyle=style,
                marker= 'x' if style == 'dashed' else 'o'
            )
    plt.ylim(y_lim,)
    plt.xlabel('Missing Ratio')
    plt.ylabel('Accuracy')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"./_fig/{DIR}/mr_{missing_type}_{dataset}.png")