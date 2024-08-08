import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

class PopRecDataset(Dataset):

    def __init__(self, data, item2attribute, attributes_count, num_items=None): 
        super().__init__()
        self.data = data
        self.num_items = num_items
        self.item2attribute = item2attribute
        self.attributes_count = attributes_count

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, user_id):
        data = self.data.iloc[user_id,:]
        pos_item = data['y'].astype('int64')
        return {
            'user_id': user_id,
            'X': np.array(data['X'], dtype='int64'),
            'pos_item': pos_item,
            }
