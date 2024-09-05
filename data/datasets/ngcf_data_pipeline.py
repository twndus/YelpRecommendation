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
        user_item_interaction = user_item_interaction.droplevel(0, 1).fillna(0)

        # adjacency matrix
        logger.info('create adjacency matrix')
        adjacency_matrix = np.zeros((self.num_items+self.num_users, self.num_items+self.num_users), dtype=np.float32)
        adjacency_matrix[:self.num_users,self.num_users:] = user_item_interaction
        adjacency_matrix[self.num_users:,:self.num_users] = user_item_interaction.T

        # diagonal degree matrix (n+m) x (m+n)
        logger.info('create diagonal degree matrix')
        diagonal_degree_matrix = np.diag(1/np.sqrt(adjacency_matrix.sum(axis=0))).astype(np.float32)

        # set laplacian matrix
        logger.info('set laplacian matrix')
        diagonal_degree_matrix = torch.from_numpy(diagonal_degree_matrix).to_sparse().to('cuda')
        adjacency_matrix = torch.from_numpy(adjacency_matrix).to_sparse().to('cuda')
        self.laplacian_matrix = torch.sparse.mm(diagonal_degree_matrix, adjacency_matrix)
        del adjacency_matrix
        self.laplacian_matrix = torch.sparse.mm(self.laplacian_matrix, diagonal_degree_matrix)
        del diagonal_degree_matrix 
        logger.info('done...')

    def preprocess(self) -> pd.DataFrame:
        df = super().preprocess()
        self._set_laplacian_matrix(df)
        return df 
