import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


from legacy.gmlp_models import GMLP
from legacy.gmlp_utils import accuracy, load_data, get_batch, save_log

def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def _train_step(adj, data, train_index, model, optimizer, tau, alpha):
    model.train()
    optimizer.zero_grad()


    if alpha > 0:
        features_batch, edge_index_batch, adj_label_batch = get_batch(adj, data.x, data.edge_index, train_index,
                                                                      batch_size=batch_size)
        train_index = train_index[train_index < len(features_batch)]

        output, x_dis = model(features_batch, edge_index_batch)
        loss_train_class = F.cross_entropy(output[:len(train_index)], data.y[train_index])

        loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau=tau)

        loss_train = loss_train_class + loss_Ncontrast * alpha
    else:
        output, x_dis = model(data.x, data.edge_index)
        loss_train_class = F.cross_entropy(output[train_index], data.y[train_index])
        loss_train = loss_train_class

    acc_train = accuracy(output[train_index], data.y[train_index])

    loss_train.backward()
    optimizer.step()
    return acc_train

def _eval_step(model, data):
    model.eval()
    output = model(data.x, data.edge_index)

    loss_val = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])

    return loss_val, acc_val, acc_test

def train(adj, data, model, optimizer, epochs, tau, alpha):
    train_index = torch.arange(end=data.train_mask.size(0),
                               dtype=torch.long, device=data.train_mask.device)
    train_index = train_index[data.train_mask]

    best_train_acc = 0
    best_val_acc = 0
    best_epoch = 0
    #print('\n' + 'training configs', epochs)
    for epoch in tqdm(range(epochs)):
        acc_train = _train_step(adj, data, train_index, model, optimizer, tau, alpha)
        loss_val, acc_val, acc_test = _eval_step(model, data)
        if acc_val > best_val_acc:
            best_epoch = epoch
            best_train_acc = acc_train
            best_val_acc = acc_val
            test_acc = acc_test

    print(f"Best epoch: {best_epoch} Best Train Acc {best_train_acc:.4f} Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {test_acc:.4f}")
    return test_acc

def mlp2gnn(mlp):
    from copy import deepcopy
    from torch_geometric.nn import GCNConv
    model = deepcopy(mlp)
    model.use_edge = True
    model.embedding.in_fc = GCNConv(model.embedding.in_fc.lin.in_features, model.embedding.in_fc.lin.out_features)
    for i in range(len(model.embedding.layers)):
            model.embedding.layers[i] = GCNConv(model.embedding.layers[i].lin.in_features, model.embedding.layers[i].lin.out_features)
    return model


if __name__ == "__main__":
    from limelight.utils import fix_seed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs.')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size')
    parser.add_argument('--order', type=int, default=2,
                        help='to compute order-th power of adj')
    parser.add_argument('--pmlp', type=bool, default=False)
    parser.add_argument('--graph', type=bool, default=False)

    data_args = {
        "cora": ["class", 10, 2.0],
        "citeseer": ["class", 1.0, 0.5],
        "pubmed": ["class", 100, 1],
        "photo": ["random", 10, 1],
        "computers": ["random", 10, 1],
        "cs": ["random", 10, 1],
    }

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    runs = args.runs
    for data_name, exp_args in data_args.items():
        order = args.order
        alpha = exp_args[1] if args.graph else 0
        tau = exp_args[2]
        name = data_name
        print("\nUsing dataset:", data_name)

        hidden = args.hidden
        epochs = args.epochs
        dropout = args.dropout
        layers = 2

        lr = args.lr
        weight_decay = args.weight_decay
        batch_size = args.batch_size

        fix_seed(42)
        adj, data, num_classes = load_data(data_name, exp_args[0], order=order)
        adj = adj.to(device)
        data = data.to(device)

        run_test_acc = []
        for run in range(runs):
            _seed = 42 * (run + 1)
            fix_seed(_seed)
            model = GMLP(num_features=data.x.shape[1],
                         hidden_dim=hidden,
                         num_classes=num_classes,
                         dropout=dropout,
                         embedding_layers=layers,
                         pmlp=args.pmlp,)


            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            model = model.to(device)
            test_acc = train(adj, data, model, optimizer, epochs, tau=tau, alpha=alpha)
            run_test_acc.append(test_acc.item())
        save_log(run_test_acc, data_name)


