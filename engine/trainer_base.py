from trainer import TrainerCLS, LimelightModule
import torch.nn as nn
from torch_geometric.nn import GCNConv


def read_config(data_name, dir, config_root="./best_config"):
    import yaml, os
    with open(os.path.join(config_root, dir, f"{data_name.lower()}.yaml"), 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def read_best_config(data_name, model_type):
    gnn_dir = "save_log/baselines"
    gnn_config = read_config(data_name, gnn_dir)

    config = {"seed": gnn_config.get("seed", 42)}

    config.update(gnn_config[model_type.lower()])
    config["epochs"] = int(config["epochs"])
    config["lr"] = float(config["lr"])
    config["weight_decay"] = float(config["weight_decay"])
    return config

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = nn.functional.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):

        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x=conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



if __name__ == "__main__":
    from limelight.utils import fix_seed
    from limelight.api.dataset import load_dataset
    from engine.save_log.baselines.sgformer import SGFormer

    model = "sgformer"
    dataname = "computers"

    config = read_best_config(dataname, model)

    split = "fixed" if dataname not in ["cora", "citeseer", "pubmed"] else "class"

    fix_seed(config["seed"])
    dataset, _ = load_dataset(dataname, split, missing_rate=0.0, label_num_per_class=20)

    model = LimelightModule(dataset.x.size(1), config["hidden"], config["hidden"], config["num_layers"],
                       dropout=config["dropout"], res=False, norm="batch", gnn_type=config["backbone"],
                       gnn_cls=False)

    model = SGFormer(
        in_channels=dataset.x.size(1),
        hidden_channels=config["hidden"],
        out_channels=dataset.num_classes,
        num_layers=config["ours_layers"],
        alpha=config["alpha"],
        dropout=config["ours_dropout"],
        use_residual=config["residual"],
        graph_weight=config["graph_weight"],
        use_graph=True,
        gnn=model,
        aggregate='add',
    )
    print(model)



    trainer = TrainerCLS(args=None, model=model,
                         epochs=config["epochs"], lr=config["lr"],
                         early_stop=-1, optim="Adam", gpu_idx=0,
                         amp=False, device=None, _print=True)

    trainer.fit(dataset)

