import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module, BCELoss
from torch.optim import Optimizer, Adam, AdamW

from loguru import logger
from omegaconf.dictconfig import DictConfig
from abc import ABC, abstractmethod

from models.cdae import CDAE
from .base_trainer import BaseTrainer
from metric import *

class CDAETrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, num_items: int, num_users: int) -> None:
        super().__init__(cfg)
        self.model = CDAE(self.cfg, num_items, num_users)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)

    def train(self, train_dataloader: DataLoader) -> float:
        self.model.train()
        train_loss = 0
        for data in tqdm(train_dataloader):
            user_id, input_mask = data['user_id'].to(self.device), data['input_mask'].to(self.device)
            pred = self.model(user_id, input_mask)

            self.optimizer.zero_grad()
            loss = self.loss(pred, input_mask)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def validate(self, valid_dataloader: DataLoader) -> tuple[float]:
        self.model.eval()
        valid_loss = 0
        actual, predicted = [], []
        for data in tqdm(valid_dataloader):
            user_id, input_mask = data['user_id'].to(self.device), data['input_mask'].to(self.device)
            loss_mask = data['loss_mask'].to(self.device)
            pred = self.model(user_id, input_mask)

            self.optimizer.zero_grad()
            loss = self.loss(pred, input_mask.add(loss_mask)) # train + valid 1
            loss.backward()
            self.optimizer.step()

            valid_loss += loss.item()
            
            # argpartition top k
            # sort
            actual.extend([
                np.nonzero(user_actual.numpy())[0] for user_actual in input_mask.cpu().detach()
            ])

            # mask to train items
            pred = pred * torch.logical_not(input_mask)
            # find the largest topK item indexes by user
            topn_index = np.argpartition(pred.cpu().detach().numpy(), -self.cfg.top_n)[:, -self.cfg.top_n:]
            # take probs from predictions using above indexes
            topn_prob = np.take_along_axis(pred.cpu().detach().numpy(), topn_index, axis=1)
            # sort topK probs and find their indexes
            sorted_indices = np.argsort(-topn_prob, axis=1)
            # apply sorted indexes to item indexes to get sorted topK item indexes by user
            topn_index_sorted = np.take_along_axis(topn_index, sorted_indices, axis=1)

            predicted.append(topn_index_sorted)

        logger.info(f"len actual: {len(actual)}")
        logger.info(f"actual first: {actual[0]}")
        predicted = np.concatenate(predicted, axis=0)

        valid_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        valid_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        valid_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        valid_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)

        return (valid_loss,
             valid_precision_at_k,
             valid_recall_at_k,
             valid_map_at_k,
             valid_ndcg_at_k)

    def evaluate(self, k: int=20) -> tuple[float]:
        pass
