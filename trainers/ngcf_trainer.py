import wandb

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from loguru import logger
from omegaconf.dictconfig import DictConfig
import wandb

from models.ngcf import NGCF
from .base_trainer import BaseTrainer
from metric import *
from loss import BPRLoss

class NGCFTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, num_items: int, num_users: int, laplacian_matrix: torch.Tensor) -> None:
        super().__init__(cfg)
        logger.info(f'[DEVICE] device = {self.device}')
        self.num_items = num_items
        self.num_users = num_users
        self.model = NGCF(self.cfg, num_users, num_items).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr, self.cfg.weight_decay)
        self.loss = self._loss()
        self.laplacian_matrix = laplacian_matrix

    def _loss(self):
        return BPRLoss()
    
    def run(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, valid_eval_data: pd.DataFrame):
        logger.info(f"[Trainer] run...")

        best_valid_loss: float = 1e+6
        best_valid_precision_at_k: float = .0
        best_valid_recall_at_k: float = .0
        best_valid_map_at_k: float = .0
        best_valid_ndcg_at_k: float = .0
        best_epoch: int = 0
        endurance: int = 0

        # train
        for epoch in range(self.cfg.epochs):
            train_loss: float = self.train(train_dataloader)
            valid_loss: float = self.validate(valid_dataloader)
            (valid_precision_at_k,
             valid_recall_at_k,
             valid_map_at_k,
             valid_ndcg_at_k) = self.evaluate(valid_eval_data, 'valid')
            logger.info(f'''\n[Trainer] epoch: {epoch} > train loss: {train_loss:.4f} / 
                        valid loss: {valid_loss:.4f} / 
                        precision@K : {valid_precision_at_k:.4f} / 
                        Recall@K: {valid_recall_at_k:.4f} / 
                        MAP@K: {valid_map_at_k:.4f} / 
                        NDCG@K: {valid_ndcg_at_k:.4f}''')
            
            # wandb logging 
            if self.cfg.wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'valid_Precision@K': valid_precision_at_k,
                    'valid_Recall@K': valid_recall_at_k,
                    'valid_MAP@K': valid_map_at_k,
                    'valid_NDCG@K': valid_ndcg_at_k,
                })

            # update model
            if self._is_surpass_best_metric(
                current=(valid_loss,
                         valid_precision_at_k,
                         valid_recall_at_k,
                         valid_map_at_k,
                         valid_ndcg_at_k),
                best=(best_valid_loss,
                      best_valid_precision_at_k,
                      best_valid_recall_at_k,
                      best_valid_map_at_k,
                      best_valid_ndcg_at_k)):
                logger.info(f"[Trainer] update best model...")
                best_valid_loss = valid_loss
                best_valid_precision_at_k = valid_precision_at_k
                best_valid_recall_at_k = valid_recall_at_k
                best_valid_ndcg_at_k = valid_ndcg_at_k
                best_valid_map_at_k = valid_map_at_k
                best_epoch = epoch
                endurance = 0

                torch.save(self.model.state_dict(), f'{self.cfg.model_dir}/best_model.pt')
            else:
                endurance += 1
                if endurance > self.cfg.patience: 
                    logger.info(f"[Trainer] ealry stopping...")
                    break


    def train(self, train_dataloader: DataLoader) -> float:
        self.model.train()
        train_loss = 0
        for data in tqdm(train_dataloader):
            user_id, pos_item, neg_item = data['user_id'].to(self.device), data['pos_item'].to(self.device), \
                 data['neg_item'].to(self.device)
            pos_pred,neg_pred = self.model.bpr_forward(user_id, pos_item, neg_item, self.laplacian_matrix)

            self.optimizer.zero_grad()
            loss = self.loss(pos_pred, neg_pred)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def validate(self, valid_dataloader: DataLoader) -> tuple[float]:
        self.model.eval()
        valid_loss = 0
        actual, predicted = [], []
        for data in tqdm(valid_dataloader):
            user_id, pos_item, neg_item = data['user_id'].to(self.device), data['pos_item'].to(self.device), \
                data['neg_item'].to(self.device)
            pos_pred,neg_pred = self.model.bpr_forward(user_id, pos_item, neg_item, self.laplacian_matrix)

            loss = self.loss(pos_pred, neg_pred)

            valid_loss += loss.item()

        return valid_loss
    
    def evaluate(self, eval_data: pd.DataFrame, mode='valid') -> tuple:
        self.model.eval()
        actual, predicted = [], []
        batch_size = 64

        for idx in tqdm(range(0, eval_data.shape[0], batch_size)):
            user_id = [i for i in range(idx, min(idx+batch_size, self.num_users))]
            item_input = torch.tensor([item_id for item_id in range(self.num_items)]*len(user_id)).to(self.device)
            
            batch_row = eval_data.iloc[user_id, :]
            user_id = np.concatenate([[id,] * self.num_items for id in user_id], axis=-1)
            user_input = torch.tensor(user_id).to(self.device)

            batch_pred = self.model(user_input, item_input, self.laplacian_matrix)
            
            for i, row in batch_row.iterrows():
                pred = batch_pred[i:i+self.num_items].cpu()
                user_predicted = self._generate_top_k_recommendation(pred, row['mask_items'])

                actual.append(row['pos_items'])
                predicted.append(user_predicted)

        test_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        test_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        test_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        test_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)
        
        if mode == 'test': 
            logger.info(f'''\n[Trainer] Test > 
                            precision@{self.cfg.top_n} : {test_precision_at_k:.4f} / 
                            Recall@{self.cfg.top_n}: {test_recall_at_k:.4f} / 
                            MAP@{self.cfg.top_n}: {test_map_at_k:.4f} / 
                            NDCG@{self.cfg.top_n}: {test_ndcg_at_k:.4f}''')
            if wandb.run is not None: # validate wandb initialization
                logger.info("[Trainer] logging test results...")
                wandb.log({
                    'test_Precision@K': test_precision_at_k,
                    'test_Recall@K': test_recall_at_k,
                    'test_MAP@K': test_map_at_k,
                    'test_NDCG@K': test_ndcg_at_k,
                })

        return (test_precision_at_k,
             test_recall_at_k,
             test_map_at_k,
             test_ndcg_at_k)

    def _evaluate(self, eval_data: pd.DataFrame, mode='valid') -> tuple:

        self.model.eval()
        actual, predicted = [], []
        item_input = torch.tensor([item_id for item_id in range(self.num_items)]).to(self.device)

        for idx in tqdm(np.random.randint(eval_data.shape[0], size=100), total=100):
            user_id = eval_data.iloc[[idx], :].index[0]
            row = eval_data.iloc[idx, :]

            user_input = torch.tensor([user_id,]*self.num_items).to(self.device)
            pred = self.model(user_input, item_input, self.laplacian_matrix)

            batch_predicted = \
                self._generate_top_k_recommendation(pred, row['mask_items'])
            actual.append(row['pos_items'])
            predicted.append(batch_predicted)

        test_precision_at_k = precision_at_k(actual, predicted, self.cfg.top_n)
        test_recall_at_k = recall_at_k(actual, predicted, self.cfg.top_n)
        test_map_at_k = map_at_k(actual, predicted, self.cfg.top_n)
        test_ndcg_at_k = ndcg_at_k(actual, predicted, self.cfg.top_n)
        
        if mode == 'test': 
            logger.info(f'''\n[Trainer] Test > 
                            precision@{self.cfg.top_n} : {test_precision_at_k:.4f} / 
                            Recall@{self.cfg.top_n}: {test_recall_at_k:.4f} / 
                            MAP@{self.cfg.top_n}: {test_map_at_k:.4f} / 
                            NDCG@{self.cfg.top_n}: {test_ndcg_at_k:.4f}''')

        return (test_precision_at_k,
             test_recall_at_k,
             test_map_at_k,
             test_ndcg_at_k)
        
    def _generate_top_k_recommendation(self, pred: Tensor, mask_items) -> tuple[list]:

        # mask to train items
        pred = pred.cpu().detach().numpy()
        pred[mask_items] = -3.40282e+38 # finfo(float32)

        # find the largest topK item indexes by user
        topn_index = np.argpartition(pred, -self.cfg.top_n)[-self.cfg.top_n:]
        # take probs from predictions using above indexes
        topn_prob = np.take_along_axis(pred, topn_index, axis=0)
        # sort topK probs and find their indexes
        sorted_indices = np.argsort(-topn_prob)
        # apply sorted indexes to item indexes to get sorted topK item indexes by user
        topn_index_sorted = np.take_along_axis(topn_index, sorted_indices, axis=0)

        return topn_index_sorted
