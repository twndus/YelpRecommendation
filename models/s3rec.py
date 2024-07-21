import torch
import torch.nn as nn

from models.base_model import BaseModel

from loguru import logger
class S3Rec(BaseModel):

    def __init__(self, cfg, num_users, num_items, attributes_count):
        super().__init__()
        self.cfg = cfg
        # self.user_embedding = nn.Embedding(num_users, cfg.embed_size, dtype=torch.float32) 
        self.item_embedding = nn.Embedding(num_items + 1, self.cfg.embed_size, dtype=torch.float32)
        # self.attribute_embedding = nn.Embedding(attributes_count, self.cfg.embed_size, dtype=torch.float32)
        self.positional_encoding = nn.Parameter(torch.rand(self.cfg.max_seq_len, self.cfg.embed_size))
        
        # self.query = nn.ModuleList([nn.Linear(self.cfg.embed_size / self.num_heads) for _ in range(self.cfg.num_heads)])
        # self.key = nn.ModuleList([nn.Linear(self.cfg.embed_size) for _ in range(self.cfg.num_heads)])
        # self.value = nn.ModuleList([nn.Linear(self.cfg.embed_size) for _ in range(self.cfg.num_heads)])
        self.ffn1s = nn.ModuleList([nn.Linear(self.cfg.embed_size, self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        self.ffn2s = nn.ModuleList([nn.Linear(self.cfg.embed_size, self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        self.multihead_attns = nn.ModuleList([nn.MultiheadAttention(self.cfg.embed_size, self.cfg.num_heads) for _ in range(self.cfg.num_blocks)])
        self._init_weights()

    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.xavier_uniform_(child.weight)

    def _embedding_layer(self, X):
        return self.item_embedding(X) + self.positional_encoding
        
    def _self_attention_block(self, X):
        for multihead_attn, ffn1, ffn2 in zip(self.multihead_attns, self.ffn1s, self.ffn2s):
            attn_output, attn_output_weights = multihead_attn(X, X, X)
            X = ffn2(nn.functional.relu(ffn1(attn_output)))
        return X

    def _prediction_layer(self, item, self_attn_output):
        return torch.matmul(item.T, self_attn_output)

    def forward(self, X, pos_item, neg_item):
        X = self._embedding_layer(X)
        X = self._self_attention_block(X)
        pos_pred = self._prediction_layer(self.item_embedding(pos_item), X[:, -1])
        neg_pred = self._prediction_layer(self.item_embedding(neg_item), X[:, -1])
        return pos_pred, neg_pred
