import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from loguru import logger
from omegaconf.dictconfig import DictConfig
import wandb

from utils import log_metric
from metric import *

class PopRecTrainer(nn.Module):
    def __init__(self, cfg: DictConfig, item_list: list) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_list = item_list
    
    @log_metric
    def evaluate(self, dataloader: DataLoader) -> tuple[float]:
        actual, predicted = [], []
        for data in tqdm(dataloader):
            batch_actual = np.expand_dims(data['pos_item'].numpy(), -1).tolist()
            # Filter
            batch_predicted = []
            for bi in range(data['X'].size(0)):
                user_predicted = []
                ii = 0
                while len(user_predicted) < self.cfg.top_n:
                    if self.item_list[ii] not in data['X'][bi]:
                        user_predicted.append(self.item_list[ii])
                    ii += 1
                batch_predicted.append(user_predicted)

            actual.extend(batch_actual)
            predicted.extend(batch_predicted)

        predicted = np.array(predicted)
        logger.info(f'actual: {np.array(actual).shape}')
        logger.info(f'predicted: {predicted.shape}')

        test_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        test_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        test_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        test_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)

        logger.info(f'''\n[Trainer] Test > 
                        precision@{self.cfg.top_n} : {test_precision_at_k:.4f} / 
                        Recall@{self.cfg.top_n}: {test_recall_at_k:.4f} / 
                        MAP@{self.cfg.top_n}: {test_map_at_k:.4f} / 
                        NDCG@{self.cfg.top_n}: {test_ndcg_at_k:.4f}''')

        return (test_precision_at_k,
                test_recall_at_k,
                test_map_at_k,
                test_ndcg_at_k)

    def _generate_target_and_top_k_recommendation(self, scores: Tensor, pos_item: Tensor) -> tuple[list]:
        actual = pos_item.cpu().detach().numpy()

        # create item index information
        scores_idx = np.zeros_like(scores.cpu().detach().numpy())
        scores_idx[:, 0] = pos_item.cpu().detach()

        # sort topK probs and find their indexes
        sorted_indices = np.argsort(-scores.cpu().detach().numpy(), axis=1)[:, :self.cfg.top_n]
        # apply sorted indexes to item indexes to get sorted topK item indexes by user
        predicted = np.take_along_axis(scores_idx, sorted_indices, axis=1)
    
        return actual.reshape(pos_item.size(0),1).tolist(), predicted[:, :self.cfg.top_n]
