import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def sinkhorn(out, num_iters, epsilon):
    Q = torch.exp(out / epsilon).t() 
    B = Q.shape[1]
    K = Q.shape[0] 
    sum_Q = torch.sum(Q)
    Q  =  Q / sum_Q
    for it in range(num_iters):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q = Q / sum_of_rows
        Q = Q / K
        Q = Q /  torch.sum(Q, dim=0, keepdim=True)
        Q = Q / B
    Q =  Q * B 
    
    return Q.t()

def multihead_sinkhorn(A, num_iters=3, epsilon=0.05):
    N, H, K = A.shape
    outputs = []
    for h in range(H):
        Q_h = sinkhorn(A[:, h, :], num_iters=num_iters, epsilon=epsilon) 
        outputs.append(Q_h.unsqueeze(1))  
    return torch.cat(outputs, dim=1)


class Static_Sink(nn.Module):
    def __init__(self, dim, dim_attn, K_v, head):
        super(Static_Sink, self).__init__()
        self.head_dim = dim_attn // head     
        self.query = nn.Linear(dim, dim_attn)
        self.key = nn.Parameter(torch.randn((K_v, head, self.head_dim)))
        self.value = nn.Linear(dim, dim)
        self.emb = nn.Parameter(torch.full((head, K_v), 0).unsqueeze(0).float())  
        self.alpha = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))
        self.beta = nn.Parameter(torch.tensor([math.log(0.01) for _ in range(head)]).unsqueeze(-1))
        self.K_v = K_v
        self.head = head
        self.struct_bias_proj = nn.Linear(4, self.head * self.K_v)
        
        
    def forward(self, x, node_index):  
        n, f = x.shape  
        q = self.query(x).reshape(n, self.head, self.head_dim) 
        attn = torch.einsum('nhd, rhd -> nhr', q, self.key) / (self.head_dim ** 0.5)   
        bias = self.struct_bias_proj(node_index)  
        bias = bias.view(n, self.head, self.K_v)  
        attn = attn + bias
        x = self.value(x).reshape(n, self.head, self.head_dim)
        v = torch.einsum('nhr, nhd -> rhd', F.sigmoid(attn), x)  
        Cons = multihead_sinkhorn(F.softmax(attn, dim = -1)) 
        v = torch.einsum('nhr, rhd -> nhd', Cons, v)
        v = F.sigmoid(self.alpha) * x + F.sigmoid(self.beta) * v 
        return v.reshape(n, f)