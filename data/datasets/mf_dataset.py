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
    
    def __getitem__(self, user_id):
        input_mask = self.data[user_id]['input_mask'].astype('float32')
        if self.mode == 'train':
            pos_item = self.data[user_id]['business_id'].astype('float32')
            user_pos_items = self.data[user_id]['pos_items']
            return {
                'user_id': user_id,
                'pos_item': input_item,
                'neg_item': self._negative_sampling(input_item, user_positives)
                }
        elif self.mode == 'valid':
            pos_item = self.data[user_id]['business_id'].astype('float32')
            user_pos_items = self.data[user_id]['pos_items']
            return {
                'user_id': user_id,
                'pos_item': input_item,
                'neg_item': self._negative_sampling(input_item, user_positives)
                }
        else:
            user_pos_items = self.data[user_id]['pos_items'].astype('float32')
            return {
                'user_id': user_id,
                'pos_items': pos_items,
                }

