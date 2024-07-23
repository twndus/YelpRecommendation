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
        self.attribute_embedding = nn.Embedding(attributes_count, self.cfg.embed_size, dtype=torch.float32)
        self.positional_encoding = nn.Parameter(torch.rand(self.cfg.max_seq_len, self.cfg.embed_size))
        
        # self.query = nn.ModuleList([nn.Linear(self.cfg.embed_size / self.num_heads) for _ in range(self.cfg.num_heads)])
        # self.key = nn.ModuleList([nn.Linear(self.cfg.embed_size) for _ in range(self.cfg.num_heads)])
        # self.value = nn.ModuleList([nn.Linear(self.cfg.embed_size) for _ in range(self.cfg.num_heads)])
        self.ffn1s = nn.ModuleList([nn.Linear(self.cfg.embed_size, self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        self.ffn2s = nn.ModuleList([nn.Linear(self.cfg.embed_size, self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        self.multihead_attns = nn.ModuleList([nn.MultiheadAttention(self.cfg.embed_size, self.cfg.num_heads) for _ in range(self.cfg.num_blocks)])
        self.aap_weight = nn.Linear(self.cfg.embed_size, self.cfg.embed_size, bias=False)

        self._init_weights()


    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.xavier_uniform_(child.weight)
            elif isinstance(child, nn.ModuleList): # nn.Linear):
                for sub_child in child.children():
                    if not isinstance(sub_child, nn.MultiheadAttention):
                        nn.init.xavier_uniform_(sub_child.weight)
            elif isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
            else:
                logger.info(f"other type: {child} / {type(child)}")

    def _embedding_layer(self, X):
        return self.item_embedding(X) + self.positional_encoding
        
    def _self_attention_block(self, X):
        for multihead_attn, ffn1, ffn2 in zip(self.multihead_attns, self.ffn1s, self.ffn2s):
            attn_output, attn_output_weights = multihead_attn(X, X, X)
            X = ffn2(nn.functional.relu(ffn1(attn_output)))
        return X

    def _prediction_layer(self, item, self_attn_output):
        return torch.einsum('bi,bi->b', (item, self_attn_output))

    def forward(self, X, pos_item, neg_item):
        X = self._embedding_layer(X)
        X = self._self_attention_block(X)
        pos_pred = self._prediction_layer(self.item_embedding(pos_item), X[:, -1])
        neg_pred = self._prediction_layer(self.item_embedding(neg_item), X[:, -1])
        return pos_pred, neg_pred

    def evaluate(self, X, pos_item, neg_items):
        X = self._embedding_layer(X)
        X = self._self_attention_block(X)
        pos_pred = self._prediction_layer(self.item_embedding(pos_item), X[:, -1]).view(pos_item.size(0), -1)
        neg_preds = [self._prediction_layer(
            self.item_embedding(neg_items[:,i]), X[:, -1]).view(neg_items.size(0), -1) for i in range(neg_items.size(-1))]
        neg_preds = torch.concat(neg_preds, dim=1)
        return pos_pred, neg_preds
    
    def aap(self, items):
        # item
        item_embeddings = self.item_embedding(items)
        return torch.matmul(self.aap_weight(item_embeddings), self.attribute_embedding.weight.T) # (batch, embed_size) * (attribute_size, embed_size) (batch, attribute_size)
