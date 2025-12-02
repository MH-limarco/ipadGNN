import numpy as np
import torch
from torch_geometric import utils
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from limelight.api.dataset import load_dataset

def load_data(name, split_method, order=2):
    #_data = Planetoid(root='data', name=name, transform=NormalizeFeatures()) #,
    _data, mask = load_dataset(name, split_method, label_num_per_class=20, missing_rate=0.0)
    indices, values = utils.normalize_edge_index(_data.edge_index)
    _adj = utils.to_scipy_sparse_matrix(indices, values, num_nodes=_data.x.size(0))

    adj = _adj.copy()
    for _ in range(1, order):
        adj = adj @ _adj

    adj = adj.todense()
    adj = torch.FloatTensor(adj)

    return adj, _data, _data.num_classes


def get_batch(adj_label, features, edge_index, mask_index, batch_size):
    """
    get a batch of feature & adjacency matrix
    """
    rand_index = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    if len(mask_index) > batch_size:
        rand_index = mask_index
    else:
        rand_index[:len(mask_index)] = mask_index
    features_batch = features[rand_index]
    edge_index_batch = edge_index[:, rand_index].to(features.device)
    adj_label_batch = adj_label[rand_index, :][: ,rand_index]
    return features_batch, edge_index_batch, adj_label_batch


def accuracy(output, labels):
    preds = torch.max(output, axis=1)[1].type_as(labels)
    correct = preds.eq(labels).sum()
    return correct / len(labels)

def save_log(run_test_acc, data_name, log_name="log"):
    run_test_acc = np.array(run_test_acc)
    avg_test_acc = np.mean(run_test_acc)
    std_test_acc = np.std(run_test_acc)

    output_test_acc = f"{avg_test_acc:.4f} ± {std_test_acc:.3f}"

    print(f"Dataset - {data_name}: Test Acc: {output_test_acc}")

    log_file = open(f"gmlp_{log_name}.txt", encoding="utf-8", mode="a+")
    with log_file as txt:
        print('data', 'test_acc',
              file=txt)

        print(data_name,
              output_test_acc, file=txt, sep=',')