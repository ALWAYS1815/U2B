import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from U2B.generate import *
from U2B.Dynamic_Cons import Dynamic_Cons

class Graph_Converter(nn.Module):
    def __init__(self, dim, head, K_g, topk):
        super(Graph_Converter, self).__init__()
        
        self.dynamic = Dynamic_Cons(dim, dim, K_g, head, topk)
        self.FFN = FFN(dim, dim)
        self.norm1 = RMSNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(math.log(9)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

        self.gamma = nn.Parameter(torch.tensor(math.log(9)))
        self.delta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

    def forward(self, x, bias):  
        x = self.norm1(F.sigmoid(self.gamma) * x)
        attn, topk_indices = self.dynamic(x, bias)
        return F.sigmoid(self.alpha) * self.FFN(x) + F.sigmoid(self.beta) * (attn), topk_indices