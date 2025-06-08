import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出变换
        output = self.W_o(context)
        return output

class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    使用多头注意力机制自动学习特征交互
    """
    def __init__(self, n_users, n_items, embed_dim=64, num_layers=3, num_heads=8):
        super().__init__()
        
        # 嵌入层
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        
        # 多层自注意力
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # 层标准化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # 预测层
        self.prediction = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for emb in [self.user_emb, self.item_emb]:
            nn.init.normal_(emb.weight, 0, 0.05)
    
    def forward(self, u, i):
        # 获取嵌入
        user_vec = self.user_emb(u)  # [batch_size, embed_dim]
        item_vec = self.item_emb(i)  # [batch_size, embed_dim]
        
        # 堆叠特征，形状为 [batch_size, 2, embed_dim]
        x = torch.stack([user_vec, item_vec], dim=1)
        
        # 多层自注意力
        for attention_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            # 残差连接 + 层标准化
            x = layer_norm(x + attention_layer(x))
        
        # 聚合特征
        user_final = x[:, 0, :]  # [batch_size, embed_dim]
        item_final = x[:, 1, :]  # [batch_size, embed_dim]
        
        # 拼接并预测
        concat_features = torch.cat([user_final, item_final], dim=1)
        prediction = self.prediction(concat_features).squeeze()
        
        return prediction 