import torch
import torch.nn as nn

class SVDattrModel(nn.Module):
    def __init__(self, n_users, n_items, n_attributes, k=32, attr_dim=16):
        super().__init__()
        self.P = nn.Embedding(n_users, k)
        self.Q = nn.Embedding(n_items, k)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.zeros(1))
        self.attr_embedding = nn.Embedding(n_attributes, attr_dim)
        self.attr_fusion = nn.Linear(attr_dim, k)
        
        for emb in (self.P, self.Q, self.bu, self.bi, self.attr_embedding):
            nn.init.normal_(emb.weight, 0, 0.05)
        nn.init.normal_(self.attr_fusion.weight, 0, 0.05)

    def forward(self, u, i, item_attrs=None):
        user_factors = self.P(u)  
        item_factors = self.Q(i)  
        
        if item_attrs is not None:
            attr_embeds = self.attr_embedding(item_attrs)
            avg_attr_embed = torch.mean(attr_embeds, dim=1)  
            attr_factors = self.attr_fusion(avg_attr_embed)  
            item_factors = item_factors + attr_factors
        
        dot_product = (user_factors * item_factors).sum(1)
        prediction = dot_product + self.bu(u).squeeze() + self.bi(i).squeeze() + self.mu
        
        return prediction

    def forward_without_attrs(self, u, i):
        return self.forward(u, i, item_attrs=None) 