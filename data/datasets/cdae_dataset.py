import torch
from torch.utils.data import Dataset

class CDAEDataset(Dataset):

    def __init__(self, data, mode='train'): 
        super().__init__()
        self.data = data
        self.mode = mode 

    def __len__(self):
        return len(self.data.keys())
    
    def __getitem__(self, user_id):
        if self.mode == 'train':
            return {
                'user_id': user_id,
                'input_mask': self.data[user_id]['input_mask'].astype('float32'),
                }
        elif self.mode == 'valid':
            return {
                'user_id': user_id,
                'input_mask': self.data[user_id]['input_mask'].astype('float32'),
                'loss_mask': self.data[user_id]['loss_mask'].astype('float32'),
                }
        else:
            return {
                'user_id': user_id,
                'input_mask': self.data[user_id]['input_mask'].astype('float32'),
                'train_valid_mask': self.data[user_id]['train_valid_mask'].astype('float32')
                }

