import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from loguru import logger
from omegaconf.dictconfig import DictConfig

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
            valid_mask = data['valid_mask'].to(self.device)
            pred = self.model(user_id, input_mask)

            loss = self.loss(pred, input_mask.add(valid_mask)) # train + valid 1
            valid_loss += loss.item()

            batch_actual, batch_predicted = self._generate_target_and_top_k_recommendation(pred, valid_mask, input_mask)
            actual.extend(batch_actual)
            predicted.extend(batch_predicted)

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

    def evaluate(self, test_dataloader: DataLoader) -> None:
        self.model.eval()
        actual, predicted = [], []
        for data in tqdm(test_dataloader):
            input_mask, user_id, test_mask = data['input_mask'].to(self.device), \
                data['user_id'].to(self.device), data['test_mask'].to(self.device)

            pred = self.model(user_id, input_mask)

            batch_actual, batch_predicted = \
                self._generate_target_and_top_k_recommendation(pred, test_mask, input_mask)
            actual.extend(batch_actual)
            predicted.extend(batch_predicted)

        predicted = np.concatenate(predicted, axis=0)

        test_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        test_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        test_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        test_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)

        logger.info(f'''\n[Trainer] Test > 
                        precision@{self.cfg.top_n} : {test_precision_at_k:.4f} / 
                        Recall@{self.cfg.top_n}: {test_recall_at_k:.4f} / 
                        MAP@{self.cfg.top_n}: {test_map_at_k:.4f} / 
                        NDCG@{self.cfg.top_n}: {test_ndcg_at_k:.4f}''')
        
    def _generate_target_and_top_k_recommendation(self, pred: Tensor, actual_mask, pred_mask) -> tuple[list]:
        actual, predicted = [], []

        # make actual set
        actual.extend([
            np.nonzero(user_actual.numpy())[0] for user_actual in actual_mask.cpu().detach()
        ])

        # mask to train items
        pred = pred * torch.logical_not(pred_mask)
        # find the largest topK item indexes by user
        topn_index = np.argpartition(pred.cpu().detach().numpy(), -self.cfg.top_n)[:, -self.cfg.top_n:]
        # take probs from predictions using above indexes
        topn_prob = np.take_along_axis(pred.cpu().detach().numpy(), topn_index, axis=1)
        # sort topK probs and find their indexes
        sorted_indices = np.argsort(-topn_prob, axis=1)
        # apply sorted indexes to item indexes to get sorted topK item indexes by user
        topn_index_sorted = np.take_along_axis(topn_index, sorted_indices, axis=1)

        predicted.append(topn_index_sorted)
    
        return actual, predicted
