

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, GCNConv, SAGEConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from U2B.generate import *
from U2B.Graph_Converter import Graph_Converter
from U2B.Node_Converter import Node_Converter

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, n_class, K_g , K_v, head, topk):
        super(Encoder, self).__init__()
        
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.muti_nc = torch.nn.ModuleList()
        self.fc1 = Linear(dim * 5, dim * 5)
        self.fc2 = Linear(dim * 5 * 5, dim * 5)
        self.encoder = FFN(dim, dim)
        self.GC = Graph_Converter(5 * dim, head, K_g, topk)  # 24
        self.alpha = torch.nn.Parameter(torch.tensor(0.))
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.))
        
        self.beta = torch.nn.Parameter(torch.tensor(0.))
        self.decoder1 = FFN(5 * dim, 5 * dim)
        self.decoder = FFN(5 * dim, n_class)
        
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(5 * dim, 5 * dim), ReLU(), Linear(5 * dim, 5 * dim))
            else:
                nn = Sequential(Linear(num_features, 5 * dim), ReLU(), Linear(5 * dim, 5 * dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(5 * dim)
            NC = Node_Converter(5 * dim, head, K_v) 
            self.convs.append(conv)
            self.bns.append(bn)
            self.muti_nc.append(NC)
    
    def forward(self, x, edge_index, batch, node_index, bias):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            x = self.muti_nc[i](x, node_index)
            xs.append(x)

        xs = torch.cat(xs, dim = 1)  
        xs = self.fc2(xs)
        x = global_add_pool(xs, batch)  
        x1 = F.relu(self.fc1(x))
        attn, topk_indices = self.GC(x1, bias)
        x2 = F.sigmoid(self.alpha) * attn
        x3 = self.decoder(x1)

        return x2, xs, F.log_softmax(x3, dim=1), topk_indices