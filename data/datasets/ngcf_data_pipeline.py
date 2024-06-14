import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from loguru import logger

import torch

from .mf_data_pipeline import MFDataPipeline

class NGCFDataPipeline(MFDataPipeline):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.laplacian_matrix = None

    def _set_laplacian_matrix(self, df):
        logger.info('set laplacian matrix...')
        # transform df to user-item interaction (R)
        logger.info('transform df to user-item interaction')
        user_item_interaction = df.pivot_table(index='user_id', columns=['business_id'], values=['rating'])
        user_item_interaction = user_item_interaction.droplevel(0, 1)

        # adjacency matrix
        logger.info('create adjacency matrix')
        adjacency_matrix = np.zeros((self.num_items+self.num_users, self.num_items+self.num_users))
        adjacency_matrix[:self.num_users,self.num_users:] = user_item_interaction
        adjacency_matrix[self.num_users:,:self.num_users] = user_item_interaction.T

        # diagonal degree matrix (n+m) x (m+n)
        logger.info('create diagonal degree matrix')
        diagonal_degree_matrix = np.diag(1/np.sqrt(adjacency_matrix.sum(axis=0)))

        # set laplacian matrix
        logger.info('set laplacian matrix')
        diagonal_degree_matrix = torch.tensor(diagonal_degree_matrix).float().to('cuda')
        adjacency_matrix = torch.tensor(adjacency_matrix).float().to('cuda')
        self.laplacian_matrix = torch.matmul(diagonal_degree_matrix, adjacency_matrix)
        adjacency_matrix = adjacency_matrix.cpu().detach()
        self.laplacian_matrix = torch.matmul(self.laplacian_matrix, diagonal_degree_matrix)
        logger.info('done...')

    def preprocess(self) -> pd.DataFrame:
        df = super().preprocess()
        self._set_laplacian_matrix(df)
        return df 
