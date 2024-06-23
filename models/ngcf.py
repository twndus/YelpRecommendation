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

    def _embedding_propagation(self, last_embed: torch.Tensor, w1, w2):
        identity_matrix = torch.eye(*self.laplacian_matrix.size())
        matrix = (self.laplacian_matrix.to('cpu') + identity_matrix).to(self.cfg.device)

        # split calcuclation GPU memory shortage
#        chunk_size = 32
#        embed_list = []
#        for chunk_idx in range(0, self.num_users + self.num_items, chunk_size):
#            matrix_concat = matrix[chunk_idx : (chunk_idx + chunk_size)]
#            term1 = torch.matmul(matrix_concat.to(self.cfg.device), last_embed)
#            term1 = w1(term1)
#
#            laplacian_concat = self.laplacian_matrix[chunk_idx : (chunk_idx + chunk_size)]
#            neighbor_embeddings = torch.matmul(laplacian_concat, last_embed)
#
#            last_embed_concat = last_embed[chunk_idx : (chunk_idx + chunk_size)]
#            term2 = torch.mul(neighbor_embeddings, last_embed_concat) 
#            term2 = w2(term2)
#            embed_list.append(term1 + term2)
#
#        embed_list = torch.concat(embed_list, dim=0)

        term1 = torch.matmul(matrix, last_embed)
        term1 = w1(term1)

        neighbor_embeddings = torch.matmul(self.laplacian_matrix, last_embed)

        term2 = torch.mul(neighbor_embeddings, last_embed) 
        term2 = w2(term2)

        # return nn.functional.leaky_relu(embed_list)
        return nn.functional.leaky_relu(term1 + term2)

    def embedding_propagation(self, last_embed: torch.Tensor, w1, w2, laplacian_matrix):
        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')

        # Split last_embed into two parts for each GPU
        mid = last_embed.size(0) // 2

        # Prepare identity matrix and laplacian matrix on each GPU
        identity_matrix = torch.eye(last_embed.size(0))
        matrix = laplacian_matrix + identity_matrix

        # Compute term1 on GPU0
        term1_part0 = torch.matmul(matrix.to(device0), last_embed.to(device0))
        term1_part0 = w1(term1_part0)

        # Compute term2 on GPU1
        w2 = w2.to(device1)
        neighbor_embeddings = torch.matmul(laplacian_matrix.to(device1), last_embed.to(device1))
        term2_part1 = torch.mul(neighbor_embeddings, last_embed.to(device1))
        term2_part1 = w2(term2_part1)

        # Transfer term2_part1 to GPU0
        term2_part1 = term2_part1.to(device0)

        # Combine term1 and term2 on GPU0
        combined_result = term1_part0 + term2_part1
        #combined_result = term1_part0

        # Apply activation function
        result = nn.functional.leaky_relu(combined_result)

        return result  # Return result to device0
