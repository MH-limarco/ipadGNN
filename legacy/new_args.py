import argparse

parser = argparse.ArgumentParser(description="MLP to GNN")
parser.add_argument('--methood', type=str,
                    choices=["gnn", "prune_gnn", "prune_gnn_mlpinit"],
                    default='prune_gnn')

parser.add_argument('--runs', type=int,
                    default=10)

parser.add_argument('--lr_rate', type=float,
                    default=1)

parser.add_argument('--missing_rate', type=float,
                    default=0.0)

parser.add_argument('--model', type=str,
                    choices=["GCN", "SAGE", "GIN"],
                    default="SAGE")

parser.add_argument("--fill", type=str, default="fp",
                    choices=["fp", "apcfi", "zero"],
                    help="filling method")

parser.add_argument('--pmlp', type=bool,
                    default=False)
parser.add_argument('--norm', type=bool,
                    default=False)

parser.add_argument('--classic', action='store_true')


