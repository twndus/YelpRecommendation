import torch
import torch.nn as nn

from models.base_model import BaseModel

class MatrixFactorization(BaseModel):

    def __init__(self, cfg, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, cfg.embed_size, dtype=torch.float32) 
        self.item_embedding = nn.Embedding(num_items, cfg.embed_size, dtype=torch.float32) 

    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.normal_(child.weights)

    def forward(self, user_id, item_id):
        return torch.matmul(self.user_embedding(user_id), self.item_embedding(item_id).T)
