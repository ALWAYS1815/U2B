
from losses import local_global_loss_
from model import FF, PriorDiscriminator
from U2B.Encoder import Encoder
import torch
import torch.nn as nn


class U2B(nn.Module):
  def __init__(self, args, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(U2B, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior
    K_g = args.K_g
    K_v = args.K_v
    head = args.head
    topk = args.topk
    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(args.dataset_num_features, hidden_dim, num_gc_layers, args.n_class, K_g , K_v, head, topk)
    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

  def forward(self, x, edge_index, batch, node_index, bias):
    if x is None:
        x = torch.ones(batch.shape[0]).to(edge_index.device)
    y, M, logits, topk_indices = self.encoder(x, edge_index, batch, node_index, bias)
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    return local_global_loss + PRIOR, logits, topk_indices