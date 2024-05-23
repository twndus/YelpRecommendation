import torch
import torch.nn as nn

from models.base_model import BaseModel
from loguru import logger

class DCN(BaseModel):
    def __init__(self, cfg, num_users, num_items, attributes_count: list):
        super().__init__()
        self._calculate_emb_size = lambda x: round(6 * (x ** 0.25))
        self.user_embedding = nn.Embedding(num_users, self._calculate_emb_size(num_users), dtype=torch.float32) 
        self.item_embedding = nn.Embedding(num_items, self._calculate_emb_size(num_items), dtype=torch.float32)
        self.attributes_embeddings = nn.ModuleList([
            nn.Embedding(count+1, self._calculate_emb_size(count), dtype=torch.float32) for count in attributes_count
        ])
        self.input_size = self._calculate_emb_size(num_users) + self._calculate_emb_size(num_items) +\
                sum([self._calculate_emb_size(count) for count in attributes_count])
        self.hidden_dims = [self.input_size] + cfg.hidden_dims
        self.cross_dims = [self.input_size] * cfg.cross_orders
        self.deep = self._deep()
        self.cross_weights, self.cross_bias = self._cross()
        self.output_layer = nn.Linear(self.hidden_dims[-1] + self.cross_dims[-1], 1)
        self.device = cfg.device
        self._init_weights()
    
    def _deep(self):
        deep = nn.Sequential()
        for idx in range(len(self.hidden_dims)-1): 
            deep.append(nn.Linear(self.hidden_dims[idx], self.hidden_dims[idx+1]))
            deep.append(nn.ReLU())
        return deep
    
    def _cross(self):
        cross_weights = nn.ParameterList([nn.Parameter(torch.rand(dim)) for dim in self.cross_dims])
        cross_bias = nn.ParameterList([nn.Parameter(torch.rand(dim)) for dim in self.cross_dims])
        return cross_weights, cross_bias

    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.kaiming_normal_(child.weight)
            elif isinstance(child, nn.Linear):
                nn.init.kaiming_normal_(child.weight)
                nn.init.zeros_(child.bias)

    def forward(self, user_id, item_id, *attributes):
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)

        attributes_emb = []
        for idx, embedding in enumerate(self.attributes_embeddings):
            emb = embedding(attributes[idx])
            if len(attributes[idx].size()) > 1:
                emb = torch.mean(emb, dim=1)
            attributes_emb.append(emb)

        input_x = torch.cat([user_emb, item_emb] + attributes_emb, dim=1)
        input_x = torch.cat([self.deep(input_x), self._forward_cross(input_x)], dim=1)

        return torch.sigmoid(self.output_layer(input_x))

    def _forward_cross(self, x):
        prev_x = x
        for weight, bias in zip(self.cross_weights, self.cross_bias):
            input_x = torch.einsum('bi,bj->bij', (x, prev_x))
            prev_x = torch.matmul(input_x, weight) + bias + prev_x
        return prev_x
