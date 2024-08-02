import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

class S3RecDataset(Dataset):

    def __init__(self, data, item2attribute, attributes_count, num_items=None, train=True): 
        super().__init__()
        self.data = data
        self.num_items = num_items
        self.train = train
        self.item2attribute = item2attribute
        self.attributes_count = attributes_count

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
        aap_actual = np.array([[1 if attriute in self.item2attribute[item]['categories'] else 0 \
                                for attriute in range(self.attributes_count)] for item in data['X']], dtype='float')
        mip_actual = np.zeros((len(data['X']), self.num_items+1), dtype='float')
        for i, item in enumerate(data['X']):
            mip_actual[i, item] = 1
        if self.train:
            return {
                'user_id': user_id,
                'X': np.array(data['X'], dtype='int64'),
                'pos_item': pos_item,
                'neg_item': self._negative_sampling(data['behaviors'])[0],
                'aap_actual': aap_actual,
                'mip_actual': mip_actual,
                }
        else:
            return {
                'user_id': user_id,
                'X': np.array(data['X'], dtype='int64'),
                'pos_item': pos_item,
                'neg_items': np.array(self._negative_sampling(data['behaviors']), dtype='int64'),
                'aap_actual': aap_actual,
                'mip_actual': mip_actual,
                }
