
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from U2B.generate import *
from U2B.Sink import Static_Sink




class Node_Converter(nn.Module):
    def __init__(self, dim, head, K_v):
        super(Node_Converter, self).__init__()
        self.static = Static_Sink(dim, dim, K_v, head)
        self.FFN = FFN(dim, dim)
        self.norm1 = RMSNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(math.log(9)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))
        self.gamma = nn.Parameter(torch.tensor(math.log(9)))
        self.delta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

    def forward(self, x, node_index):  
        x = self.norm1(F.sigmoid(self.gamma) * x )
        attn = self.static(x, node_index)
        return F.sigmoid(self.alpha) * self.FFN(x) + F.sigmoid(self.beta) * (attn)