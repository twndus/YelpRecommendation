import torch
import torch.nn as nn

from models.base_model import BaseModel

from loguru import logger


class S3Rec(BaseModel):

    def __init__(self, cfg, num_items, attributes_count):
        super().__init__()
        self.cfg = cfg
        self.item_embedding = nn.Embedding(num_items + 1, self.cfg.embed_size, dtype=torch.float32)
        self.attribute_embedding = nn.Embedding(attributes_count, self.cfg.embed_size, dtype=torch.float32)
        self.positional_encoding = nn.Parameter(torch.rand(self.cfg.max_seq_len, self.cfg.embed_size))

        self.multihead_attns = nn.ModuleList(
            [nn.MultiheadAttention(self.cfg.embed_size, self.cfg.num_heads) for _ in range(self.cfg.num_blocks)])
        self.layernorm1s = nn.ModuleList(
            [nn.LayerNorm(self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        
        self.ffn1s = nn.ModuleList(
            [nn.Linear(self.cfg.embed_size, self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        self.ffn2s = nn.ModuleList(
            [nn.Linear(self.cfg.embed_size, self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])
        self.layernorm2s = nn.ModuleList(
            [nn.LayerNorm(self.cfg.embed_size) for _ in range(self.cfg.num_blocks)])

        self.dropout = nn.Dropout(self.cfg.dropout_ratio)

        self.aap_weight = nn.Linear(self.cfg.embed_size, self.cfg.embed_size, bias=False)
        self.mip_weight = nn.Linear(self.cfg.embed_size, self.cfg.embed_size, bias=False)
        self.map_weight = nn.Linear(self.cfg.embed_size, self.cfg.embed_size, bias=False)
        self.sp_weight = nn.Linear(self.cfg.embed_size, self.cfg.embed_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for child in self.children():
            if isinstance(child, nn.Embedding):
                nn.init.xavier_uniform_(child.weight)
            elif isinstance(child, nn.ModuleList): # nn.Linear):
                for sub_child in child.children():
                    if isinstance(sub_child, nn.Linear):
                        nn.init.xavier_uniform_(sub_child.weight)
            elif isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
            else:
                logger.info(f"other type: {child} / {type(child)}")

    def _embedding_layer(self, X):
        return self.item_embedding(X) + self.positional_encoding
        
    def _self_attention_block(self, X):
        for multihead_attn, ffn1, ffn2, layernorm1, layernorm2 in zip(
                self.multihead_attns, self.ffn1s, self.ffn2s, self.layernorm1s, self.layernorm2s):
            # multi-head self-attention
            attn_output, attn_output_weights = multihead_attn(X, X, X)
            # dropout
            attn_output = self.dropout(attn_output)
            # add & norm
            normalized_attn_output = layernorm1(X + attn_output)
            # feed-forward network
            ffn_output = ffn2(nn.functional.relu(ffn1(normalized_attn_output)))
            # dropout
            ffn_output = self.dropout(ffn_output)
            # add & norm
            X = layernorm2(X + ffn_output)
        return X

    def _prediction_layer(self, item, self_attn_output):
        return torch.einsum('bi,bi->b', (item, self_attn_output))

    def finetune(self, X, pos_item, neg_item):
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

    def encode(self, X):
        return self._self_attention_block(self._embedding_layer(X))
    
    def pretrain(self, item_masked_sequences, subsequence_masked_sequences, pos_subsequences, neg_subsequences):
        # encode
        attention_output = self.encode(item_masked_sequences)
        subsequence_attention_output = self.encode(subsequence_masked_sequences)
        pos_subsequence_attention_output = self.encode(pos_subsequences)
        neg_subsequence_attention_output = self.encode(neg_subsequences)
        # aap
        aap_output = self.aap(attention_output) # (B, L, A)
        # mip
        mip_output = self.mip(attention_output)
        # map
        map_output = self.map(attention_output)
        # sp
        sp_output_pos = self.sp(attention_output, pos_subsequence_attention_output) # pos 1
        sp_output_neg = self.sp(attention_output, neg_subsequence_attention_output) # neg 1
        return aap_output, mip_output, map_output, (sp_output_pos, sp_output_neg)

    def aap(self, attention_output):
        '''
        inputs:
            attention_output: [ B, L, H ]
        output: 
            [ B, L, A ]
        '''
        FW = self.aap_weight(attention_output) # [ B L H ]
        return torch.matmul(FW, self.attribute_embedding.weight.T) # [ B L H ] [ H A ] -> [ B L A ]

    def mip(self, attention_output):
        '''
        inputs:
            attention_output: [ B, L, H ]
        output: 
        '''
        FW = self.mip_weight(attention_output) # [ B L H ]
        return torch.matmul(FW, self.item_embedding.weight.t()) # [ B L H ] [ H I ] -> [ B L I ]

    def map(self, attention_output):
        '''
        inputs:
            attention_output: [ B, L, H ]
        output: 
            [ B, L, A ]
        '''
        FW = self.aap_weight(attention_output) # [ B L H ]
        return torch.matmul(FW, self.attribute_embedding.weight.T) # [ B L H ] [ H A ] -> [ B L A ]

    def sp(self, context_attention_output, subsequence_attention_output):
        '''
        inputs:
            context_attention_output: [ B, L, H ]
            subsequence_attention_output: [ B, len_subsequence, H ]
        output: 
            [ B ]

        s - input [ i1, i2, mask, mask, mask, ..., in ]
        s~ - input [ i3, i4, i5 ]

        '''
        s = context_attention_output[:, -1, :] # [ B H ]
        s_tilde = subsequence_attention_output[:, -1, :] # [ B H ]
        SW = self.sp_weight(s)
        return torch.einsum('bi,bi->b', SW, s_tilde) # [ B ]
