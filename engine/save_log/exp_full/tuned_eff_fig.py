import os
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [
    'citeseer', 'computers', 'cora', 'cs', 'photo', 'physics', 'pubmed', 'wikics'
]
SRC_MODELS = ['classic_gcn', 'classic_sage', 'truned_gcn(mpp)', 'truned_sage(mpp)']
MISSING_RATES = [0.0, 0.3, 0.6, 0.9, 0.995]
COL_NAMES = ['full_feat', '30% missing', '60% missing', '90% missing', '99.5% missing']

missing_type = "uniform" # uniform, structural

DIR = f'tuned_eff_{missing_type}'

os.makedirs(f"./_fig/{DIR}", exist_ok=True)
for dataset in DATASETS:
    _df = pd.read_csv(f"./_csv/{DIR}/{dataset}.csv")
    _df = _df.iloc[2:, :-1]
    _df.iloc[:, 0] = SRC_MODELS

    df_long = _df.melt(id_vars='model', var_name='missing_type', value_name='score')
    df_long['missing_type'] = df_long['missing_type'].replace(["full_feat", "30% missing" ,'60% missing', '90% missing', '99.5% missing'], [0.0, 0.3, 0.6, 0.9, 0.995])

    df_long['linestyle'] = df_long['model'].apply(lambda x: 'dashed' if 'classic' in x else 'solid')

    plt.figure(figsize=(9,5))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 設定線條顏色

    # 畫出每條線
    for idx, (model, group) in enumerate(df_long.groupby('model')):
        style = 'dashed' if 'classic' in model else 'solid'
        plt.plot(
            group['missing_type'], group['score'],
            label=model,
            color=palette[idx],
            linestyle=style,
            marker='x'
        )

    plt.xlabel('Missing Ratio')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"./_fig/{DIR}/{missing_type}_{dataset}.png")

