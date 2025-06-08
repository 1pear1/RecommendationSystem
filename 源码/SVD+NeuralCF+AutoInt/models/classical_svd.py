import torch
import torch.nn as nn

class ClassicalSVD(nn.Module):
    def __init__(self, n_users, n_items, k=32):
        super().__init__()
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        self.mu = nn.Parameter(torch.zeros(1))
        
        nn.init.normal_(self.P.weight, 0, .05)
        nn.init.normal_(self.Q.weight, 0, .05)

    def forward(self, u, i):
        return (self.P(u) * self.Q(i)).sum(1) + self.mu
