import torch
import torch.nn as nn

from models.base_model import BaseModel

class NGCF(BaseModel):

    def __init__(self, cfg, num_users, num_items, laplacian_matrix):
        super().__init__()
        self.num_users = num_users
        self.laplacian_matrix = laplacian_matrix
        self.embedding = nn.Embedding(
            num_users+num_items, cfg.embed_size, dtype=torch.float32)

        self.W1 = nn.ModuleList([
            nn.Linear(cfg.embed_size, cfg.embed_size, bias=False) for _ in range(cfg.num_orders) 
            ])
        self.W2 = nn.ModuleList([
            nn.Linear(cfg.embed_size, cfg.embed_size, bias=False) for _ in range(cfg.num_orders)
            ])

    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.xavier_uniform_(child.weight)k

    def forward(self, user_id, item_id):
        user_embed_list, item_embed_list = [self.embedding(user_id),], [self.embedding(self.num_users+item_id)]
        last_embed = self.embedding
        for w1, w2 in zip(self.W1, self.W2):
            last_embed = embedding_propagation(last_embed, w1, w2)
            user_embed_list.append(last_embed(user_id))
            item_embed_list.append(last_embed(self.num_users+item_id))

        user_embed = torch.concat(user_embed_list, dim=1)
        item_embed = torch.concat(item_embed_list, dim=1)

        return torch.sum(user_emb * item_emb, dim=1)

    def embedding_propagation(self, last_embed, w1, w2):
        identity_matrix = torch.eye(*self.laplacian_matrix.size())
        term1 = torch.matmul(self.laplacian_matrix + identity_matrix, last_embed)
        term1 = w1(term1)

        neighbor_embeddings = torch.matmul(self.laplacian_matrix, last_embed)
        term2 = torch.mul(neighbor_embeddings, last_embed)
        term2 = w2(term2)

        return nn.functional.leaky_relu(term1 + term2)

