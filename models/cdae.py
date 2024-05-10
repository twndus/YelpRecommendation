import torch
import torch.nn as nn

from models.base_model import BaseModel
from loguru import logger

class CDAE(BaseModel):

    def __init__(self, cfg, num_items, num_users):
        super().__init__()
        self.num_items = num_items
        self.num_users = num_users
        self.hidden_size = cfg.hidden_size
        self.device = cfg.device
        self.corruption_level = cfg.corruption_level

        self.dropout_layer = nn.Dropout(p=self.corruption_level)
        self.hidden_layer = nn.Linear(
            in_features=self.num_items, out_features=self.hidden_size, 
            bias=True, device=self.device, dtype=torch.float32
        )
        self.user_nodes = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.hidden_size,
            device=self.device, dtype=torch.float32
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.num_items, 
            bias=True, device=self.device, dtype=torch.float32
        )
        self.hidden_activation = self._activation_module(cfg.hidden_activation)
        self.output_activation = self._activation_module(cfg.output_activation)

        self._init_weights()

    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                nn.init.uniform_(child.bias)
            elif isinstance(child, nn.Embedding):
                nn.init.uniform_(child.weight)

    def add_noise(self, x):
        return self.dropout_layer(x)

    def forward(self, user_id, x):
        if self.train:
            x_corrupted = self.add_noise(x)
            z = self.hidden_activation(self.hidden_layer(x_corrupted) + self.user_nodes(user_id))
        else:
            z = self.hidden_activation(self.hidden_layer(x) + self.user_nodes(user_id))
        return self.output_activation(self.output_layer(z))
