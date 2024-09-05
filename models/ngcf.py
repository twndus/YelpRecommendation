import torch
import torch.nn as nn

from models.base_model import BaseModel
from loguru import logger

class NGCF(BaseModel):

    def __init__(self, cfg, num_users, num_items): #, laplacian_matrix):
        super().__init__()
        self.cfg = cfg
        self.num_users = num_users
        self.num_items = num_items
        # self.laplacian_matrix = laplacian_matrix
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

    def bpr_forward(self, user_id, pos_item_ids, neg_item_ids, laplacian_matrix):
        user_embed_list, pos_item_embed_list, neg_item_embed_list = \
            [self.embedding(user_id),], [self.embedding(self.num_users+pos_item_ids)], [self.embedding(self.num_users+neg_item_ids)]
        last_embed = self.embedding.weight

        for w1, w2 in zip(self.W1, self.W2):
            last_embed: torch.Tensor = self.embedding_propagation(last_embed, w1, w2, laplacian_matrix)
            user_embed_list.append(last_embed[user_id])
            pos_item_embed_list.append(last_embed[self.num_users + pos_item_ids])
            neg_item_embed_list.append(last_embed[self.num_users + neg_item_ids])

        user_embed = torch.concat(user_embed_list, dim=1)
        pos_item_embed = torch.concat(pos_item_embed_list, dim=1)
        neg_item_embed = torch.concat(neg_item_embed_list, dim=1)

        return torch.sum(user_embed * pos_item_embed, dim=1), torch.sum(user_embed * neg_item_embed, dim=1)

    def forward(self, user_id, item_id, laplacian_matrix):
        user_embed_list, item_embed_list = [self.embedding(user_id),], [self.embedding(self.num_users+item_id)]
        last_embed = self.embedding.weight
        for w1, w2 in zip(self.W1, self.W2):
            last_embed: torch.Tensor = self.embedding_propagation(last_embed, w1, w2, laplacian_matrix)
            user_embed_list.append(last_embed[user_id])
            item_embed_list.append(last_embed[self.num_users + item_id])

        user_embed = torch.concat(user_embed_list, dim=1)
        item_embed = torch.concat(item_embed_list, dim=1)

        return torch.sum(user_embed * item_embed, dim=1)

    def embedding_propagation(self, last_embed: torch.Tensor, w1, w2, laplacian_matrix):
        identity_matrix = torch.eye(*laplacian_matrix.size(), dtype=torch.float32).to_sparse().to(self.cfg.device)
        matrix = laplacian_matrix + identity_matrix

        term1 = torch.sparse.mm(matrix, last_embed)
        term1 = w1(term1)

        neighbor_embeddings = torch.sparse.mm(laplacian_matrix, last_embed)

        term2 = torch.mul(last_embed, neighbor_embeddings) 
        term2 = w2(term2)

        return nn.functional.leaky_relu(term1 + term2)
