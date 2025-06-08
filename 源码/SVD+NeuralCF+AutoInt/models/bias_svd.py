import torch
import torch.nn as nn

class BiasSVD(nn.Module):
    def __init__(self, n_users, n_items, k=32):
        super().__init__()
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.zeros(1))

        for emb in (self.P, self.Q, self.bu, self.bi):
            nn.init.normal_(emb.weight, 0, .05)

    def forward(self, u, i):
        dot = (self.P(u) * self.Q(i)).sum(1)
        return dot + self.bu(u).squeeze() + self.bi(i).squeeze() + self.mu
