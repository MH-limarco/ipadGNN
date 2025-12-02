
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv

EPS = 1e-6
DEFAULT_ACT_FUNC = nn.GELU


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1-mask) * x_dis
    return x_dis

class GNNMLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super(GNNMLPLayer, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.normalize = normalize
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.normal_(self.bias, std=EPS)

    def forward(self, x, *args):
        out = self.lin(x)
        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_layers=2, dropout=0.5, pmlp=False):
        from torch_geometric.nn import GCNConv
        super(EmbeddingLayer, self).__init__()
        self.pmlp = pmlp
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')
        self.num_features = num_features
        self.act = DEFAULT_ACT_FUNC()
        self.dropout = nn.Dropout(dropout)
        layer = nn.Linear

        self.in_fc = layer(num_features, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim, eps=EPS)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if embedding_layers > 2 else None
        for idx in range(1, embedding_layers):
            self.layers.append(layer(hidden_dim, hidden_dim))
            if idx != embedding_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim, eps=EPS))

        self.reset_parameters()

    #def _init_fc(self):
    #    self.in_fc
        #nn.init.xavier_uniform_(self.in_fc.weight)
        #nn.init.normal_(self.in_fc.bias, std=EPS)
        #for layer in self.layers:
        #    nn.init.xavier_uniform_(layer.weight)
        #    nn.init.normal_(layer.bias, std=EPS)

    def reset_parameters(self):
        self.in_fc.reset_parameters()
        self.in_norm.reset_parameters()
        if self.norms is not None:
            for norm in self.norms:
                norm.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        #self._init_fc()

    def forward(self, x, *args):
        x = self.in_fc(x)
        x = self.act(x)
        x = self.in_norm(x)
        x = self.dropout(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None and idx < len(self.norms):
                x = self.act(x)
                x = self.norms[idx](x)
                x = self.dropout(x)

        return x


class GMLP(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout, embedding_layers=2, pmlp=False):
        super(GMLP, self).__init__()
        self.num_feature = num_features
        self.embedding = EmbeddingLayer(num_features, hidden_dim, embedding_layers, dropout, pmlp=pmlp)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, *args):
        emb = self.embedding(x, *args)

        cls = self.classifier(emb)
        #cls = F.log_softmax(cls, dim=1)

        if self.training:
            x_dis = get_feature_dis(emb)
            return cls, x_dis
        return cls