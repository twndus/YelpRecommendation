import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

class MFDataset(Dataset):

    def __init__(self, data, num_items=None): 
        super().__init__()
        self.data = data
        self.num_items = num_items

    def __len__(self):
        return self.data.shape[0]
    
    def _negative_sampling(self, user_positives):
        neg_item = np.random.randint(self.num_items)
        while neg_item in user_positives:
            neg_item = np.random.randint(self.num_items)
        return neg_item
    
    def __getitem__(self, index):
        data = self.data.iloc[index,:]
        pos_item = data['business_id'].astype('int64')
        user_pos_items = data['pos_items']
        return {
            'user_id': data['user_id'].astype('int64'),
            'pos_item': pos_item,
            'neg_item': self._negative_sampling(user_pos_items)
            }
