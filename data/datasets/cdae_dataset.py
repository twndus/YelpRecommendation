import numpy as np

import torch
from torch.utils.data import Dataset

from loguru import logger

class CDAEDataset(Dataset):

    def __init__(self, data, mode='train', neg_times: int=5): 
        super().__init__()
        self.data = data
        self.mode = mode 
        if self.mode != 'test':
            self.neg_times = neg_times

    def __len__(self):
        return len(self.data.keys())
    
    def _negative_sampling(self, input_mask):
        # Calculate the number of positive samples.
        num_pos = int(input_mask.sum())
        # Flip zeros and ones to generate candidates for negative sampling.
        flipped_mask = 1-input_mask
        # Retrieve indexes of the negative candidates.
        negative_indexes = flipped_mask.nonzero()[0]
        # Sample from negative indexes, selecting multiple times the number of positive samples.
        negative_samples = np.random.choice(negative_indexes, num_pos*self.neg_times, replace=False)
        # Create a negative mask of the same shape as input_mask
        negative_mask = np.zeros_like(input_mask)
        # Set sampled indexes to 1 in the negative mask
        # Only the masked indexes need to be computed for the loss
        negative_mask[negative_samples] = 1.
        return negative_mask
    
    def __getitem__(self, user_id):
        input_mask = self.data[user_id]['input_mask'].astype('float32')
        if self.mode == 'train':
            return {
                'user_id': user_id,
                'input_mask': input_mask,
                'negative_mask': self._negative_sampling(input_mask)
                }
        elif self.mode == 'valid':
            valid_mask = self.data[user_id]['valid_mask'].astype('float32')
            return {
                'user_id': user_id,
                'input_mask': input_mask,
                'valid_mask': valid_mask,
                'negative_mask': self._negative_sampling(input_mask + valid_mask)
                }
        else:
            test_mask = self.data[user_id]['test_mask'].astype('float32')
            return {
                'user_id': user_id,
                'input_mask': input_mask,
                'test_mask': test_mask,
                }

