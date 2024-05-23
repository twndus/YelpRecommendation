import numpy as np
import pandas as pd

import torch
from torch import Tensor
from tqdm import tqdm

from collections import ChainMap

from loguru import logger

from .base_trainer import BaseTrainer
from models.wdn import WDN
from metric import *

class WDNTrainer(BaseTrainer):
    def __init__(self, cfg, num_items, num_users):
        super().__init__(cfg)
        self.num_items = num_items
        self.num_users = num_users
        self.model = WDN(self.cfg, self.num_items, self.num_users)
        self.loss = self._loss(self.cfg.loss_name)
        self.optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)
        self.train_mask = np.zeros([self.num_users, self.num_items])
        self.valid_mask = np.zeros([self.num_users, self.num_items])

    def train(self, train_dataloader):
        logger.info("[Trainer] train...")
        self.model.train()
        train_loss = .0

        # train_pos_X = []

        # for users, predict for train items and train model.
        for data in tqdm(train_dataloader):
            # positive
            pos_X, pos_y = data['pos_X'].to(self.device), data['pos_y'].to(self.device)
            pos_pred = self.model(pos_X)

            self.optimizer.zero_grad()
            batch_pos_loss: Tensor = self.loss(pos_pred, pos_y)
            batch_pos_loss.backward()
            self.optimizer.step()

            train_loss += batch_pos_loss.item()

            # train_pos_X.append({
            #     'user_id': data['pos_X'][0],
            #     'business_id': data['pos_X'][1],
            # })

            print(self.train_mask.shape)
            print(data['pos_X'])
            for row in data['pos_X']:
                self.train_mask[row[0]][row[1]] = 1

            # negative
            neg_X, neg_y = data['neg_X'].to(self.device), data['neg_y'].to(self.device)
            neg_pred = self.model(neg_X)

            self.optimizer.zero_grad()
            batch_neg_loss: Tensor = self.loss(neg_pred, neg_y)
            batch_neg_loss.backward()
            self.optimizer.step()

            train_loss += batch_neg_loss.item()

        # if self.train_mask is None:
        #     mask = pd.DataFrame(train_pos_X)
        #     mask = mask.groupby('user_id').agg(set).to_numpy()
        #     zeros = np.zeros([self.num_users, self.num_items])
        #     self.train_mask = mask

        return train_loss

    def validate(self, valid_dataloader):
        logger.info("[Trainer] validate...")
        self.model.eval()
        valid_loss = .0

        valid_X = list()

        # calculate validation loss
        for data in tqdm(valid_dataloader):
            X, y = data['X'].to(self.device), data['y'].to(self.device)
            pred = self.model(X)
            batch_loss: Tensor = self.loss(pred, y)
            valid_loss += batch_loss.item()
            # valid_X.append({
            #     'user_id': data['X'][0],
            #     'business_id': data['X'][1],
            # })

            for row in data['X']:
                self.valid_mask[row[0]][row[1]] = 1
        
        # if self.valid_mask is None:
        #     self.valid_mask = pd.DataFrame(valid_X)

        # for users, predict total items predictions, remove train set and then rank for validation items.
        actual, predicted = [], []
        for user_id in tqdm(range(self.num_users)):
            X = torch.tensor([[user_id, item_id] for item_id in range(self.num_items)]).to(self.device)
            pred: Tensor = self.model(X)
            
            batch_actual, batch_predicted = self._generate_target_and_top_k_recommendation(pred, self.valid_mask, self.train_mask)
            actual.extend(batch_actual)
            predicted.extend(batch_predicted)
        
        valid_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        valid_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        valid_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        valid_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)

        return (valid_loss,
             valid_precision_at_k,
             valid_recall_at_k,
             valid_map_at_k,
             valid_ndcg_at_k)

    def evaluate(self, test_dataloader):
        logger.info("[Trainer] validate...")
        self.model.eval()

        test_mask = np.zeros([self.num_users, self.num_items])
        for data in test_dataloader:
            # test_X.append({
            #     'user_id': data['X'][0],
            #     'business_id': data['X'][1],
            # })
            for row in data['X']:
                test_mask[row[0]][row[1]] = 1
        
        actual, predicted = [], []

        # for users, predict total items predictions, remove train set and then rank for validation items.
        for user_id in tqdm(range(self.num_users)):
            X = torch.tensor([[user_id, item_id] for item_id in range(self.num_items)]).to(self.device)
            pred: Tensor = self.model(X)

            batch_actual, batch_predicted = self._generate_target_and_top_k_recommendation(pred, test_mask, self.train_mask + self.valid_mask)
            actual.extend(batch_actual)
            predicted.extend(batch_predicted)
        
        valid_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        valid_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        valid_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        valid_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)

        return (valid_precision_at_k,
             valid_recall_at_k,
             valid_map_at_k,
             valid_ndcg_at_k)
    
    def _sampled(self, num:int=200):
        return np.random.choice(np.arange(self.num_users), num)
    
    def _generate_target_and_top_k_recommendation(self, pred: Tensor, actual_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[list]:
        actual, predicted = [], []

        # make actual set
        actual.extend([np.nonzero(user_actual.numpy())[0] for user_actual in actual_mask])

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
