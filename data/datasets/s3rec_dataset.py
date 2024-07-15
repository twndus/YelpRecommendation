import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

class S3RecDataset(Dataset):

    def __init__(self, data, num_items=None, train=True): 
        super().__init__()
        self.data = data
        self.num_items = num_items
        self.train = train

    def __len__(self):
        return self.data.shape[0]
    
    def _negative_sampling(self, behaviors):
        sample_size = 1 if self.train else 99
        neg_items = []
        for _ in range(sample_size):
            neg_item = np.random.randint(1, self.num_items+1)
            while (neg_item in behaviors) or (neg_item in neg_items):
                neg_item = np.random.randint(1, self.num_items+1)
            neg_items.append(neg_item)
        return neg_items
    
    def __getitem__(self, user_id):
        data = self.data.iloc[user_id,:]
        pos_item = data['y'].astype('int64')
        if self.train:
            return {
                'user_id': user_id,
                'X': data['X'],
                'pos_item': pos_item,
                'neg_item': self._negative_sampling(data['behaviors'])[0]
                }
        else:
            return {
                'user_id': user_id,
                'X': data['X'],
                'pos_item': pos_item,
                'neg_items': self._negative_sampling(data['behaviors'])
                }
