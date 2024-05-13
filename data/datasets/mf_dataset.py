import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

class MFDataset(Dataset):

    def __init__(self, data, mode='train', num_items=None): 
        super().__init__()
        self.data = data
        self.mode = mode 
        self.num_items = num_items

    def __len__(self):
        return len(self.data.keys())
    
    def _negative_sampling(self, input_item, user_positives):
        neg_item = np.random.randint(self.num_items)
        while neg_item in user_positives:
            neg_item = np.random.randint(self.num_items)
        return neg_item
    
    def __getitem__(self, index):
        data = self.data.iloc[index,:]
        if self.mode == 'train':
            pos_item = data['business_id'].astype('int64')
            user_pos_items = data['pos_items']
            return {
                'user_id': data['user_id'].astype('int64'),
                'pos_item': pos_item,
                'neg_item': self._negative_sampling(pos_item, user_pos_items)
                }
        elif self.mode == 'valid':
            pos_item = data['business_id'].astype('int64')
            user_pos_items = data['pos_items']
            return {
                'user_id': data['user_id'].astype('int64'),
                'pos_item': pos_item,
                'neg_item': self._negative_sampling(pos_item, user_pos_items)
                }
        else:
            user_pos_items = data['pos_items'].astype('int64')
            return {
                'user_id': self.data[index]['user_id'].astype('int64'),
                'pos_items': pos_items,
                }

