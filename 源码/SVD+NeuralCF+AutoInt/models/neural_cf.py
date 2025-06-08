import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) 模型
    结合了矩阵分解和多层感知机的优点
    """
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        # GMF (Generalized Matrix Factorization) 分支
        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)
        
        # MLP (Multi-Layer Perceptron) 分支
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)
        
        # MLP 层
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 最终预测层
        self.prediction = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb, self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, 0, 0.05)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, u, i):
        # GMF 分支
        gmf_user_vec = self.gmf_user_emb(u)
        gmf_item_vec = self.gmf_item_emb(i)
        gmf_output = gmf_user_vec * gmf_item_vec
        
        # MLP 分支
        mlp_user_vec = self.mlp_user_emb(u)
        mlp_item_vec = self.mlp_item_emb(i)
        mlp_input = torch.cat([mlp_user_vec, mlp_item_vec], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # 融合两个分支
        concat_output = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.prediction(concat_output).squeeze()
        
        return prediction 