import torch
import torch.nn as nn

from models.base_model import BaseModel
from loguru import logger

class NGCF(BaseModel):

    def __init__(self, cfg, num_users, num_items, laplacian_matrix):
        super().__init__()
        self.cfg = cfg
        self.num_users = num_users
        self.num_items = num_items
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
                nn.init.xavier_uniform_(child.weight)

    def forward(self, user_id, item_id):
        user_embed_list, item_embed_list = [self.embedding(user_id),], [self.embedding(self.num_users+item_id)]
        last_embed = self.embedding.weight
        for w1, w2 in zip(self.W1, self.W2):
            last_embed: torch.Tensor = self.embedding_propagation(last_embed, w1, w2)
            user_embed_list.append(last_embed[user_id])
            item_embed_list.append(last_embed[self.num_users + item_id])

        user_embed = torch.concat(user_embed_list, dim=1)
        item_embed = torch.concat(item_embed_list, dim=1)

        return torch.sum(user_embed * item_embed, dim=1)

    def embedding_propagation(self, last_embed: torch.Tensor, w1, w2):
        identity_matrix = torch.eye(*self.laplacian_matrix.size())
        matrix = self.laplacian_matrix.to('cpu') + identity_matrix

        # split calcuclation GPU memory shortage
        chunk_size = 32
        embed_list = []
        for chunk_idx in range(0, self.num_users + self.num_items, chunk_size):
            matrix_concat = matrix[chunk_idx : (chunk_idx + chunk_size)]
            term1 = torch.matmul(matrix_concat.to(self.cfg.device), last_embed)
            term1 = w1(term1)

            laplacian_concat = self.laplacian_matrix[chunk_idx : (chunk_idx + chunk_size)]
            neighbor_embeddings = torch.matmul(laplacian_concat, last_embed)

            last_embed_concat = last_embed[chunk_idx : (chunk_idx + chunk_size)]
            term2 = torch.mul(neighbor_embeddings, last_embed_concat) 
            term2 = w2(term2)
            embed_list.append(term1 + term2)

        embed_list = torch.concat(embed_list, dim=0)

        return nn.functional.leaky_relu(embed_list)

