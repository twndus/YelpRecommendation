import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from loguru import logger
from omegaconf.dictconfig import DictConfig

from models.cdae import CDAE
from utils import log_metric
from .base_trainer import BaseTrainer
from metric import *
from loss import BPRLoss

class CDAETrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, num_items: int, num_users: int) -> None:
        super().__init__(cfg)
        self.model = CDAE(self.cfg, num_items, num_users) ##
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)
        self.loss = self._loss()

    def _loss(self):
        return BPRLoss()

    def run(self, train_dataloader: DataLoader, valid_dataloader: DataLoader):
        logger.info(f"[Trainer] run...")

        best_valid_loss: float = 1e+6
        best_epoch: int = 0
        endurance: int = 0

        # train
        for epoch in range(self.cfg.epochs):
            train_loss: float = self.train(train_dataloader)
            valid_loss = self.validate(valid_dataloader)
            logger.info(f'''\n[Trainer] epoch: {epoch} > train loss: {train_loss:.4f} / 
                        valid loss: {valid_loss:.4f}''') 
            
            if self.cfg.wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                })

            # update model
            if self._is_surpass_best_metric(
                current=(valid_loss,)
                best=(best_valid_loss,)):
                
                logger.info(f"[Trainer] update best model...")
                best_valid_loss = valid_loss
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
            X, pos_item, neg_item = data['X'].to(self.device), data['pos_item'].to(self.device), data['neg_item'].to(self.device)
            pos_pred, neg_pred = self.model(X, pos_item, neg_item)

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
            X, pos_item, neg_item = data['X'].to(self.device), data['pos_item'].to(self.device), data['neg_item'].to(self.device)
            pos_pred, neg_pred = self.model(X, pos_item, neg_item)

            self.optimizer.zero_grad()
            loss = self.loss(pos_pred, neg_pred)

            valid_loss += loss.item()

        return valid_loss
    
    @log_metric
    def evaluate(self, test_dataloader: DataLoader) -> tuple[float]:
        self.model.eval()
        actual, predicted = [], []
        for data in tqdm(test_dataloader):
            X, pos_item, neg_items = data['X'].to(self.device), data['pos_item'].to(self.device), data['neg_items'].to(self.device)
            scores = self.model.evaluate(X, pos_item, neg_items)

            batch_actual, batch_predicted = \
                self._generate_target_and_top_k_recommendation(pred, test_mask, input_mask)
            actual.append(batch_actual)
            predicted.append(batch_predicted)

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

        return (test_precision_at_k,
                test_recall_at_k,
                test_map_at_k,
                test_ndcg_at_k)

    def _generate_target_and_top_k_recommendation(self, scores: Tensor, pos_item) -> tuple[list]:
        actual = [pos_item,]

        # create item index information
        scores_idx = np.zeros_like(scores.cpu().detach().numpy())
        scores_idx[0,:] = pos_item

        # sort topK probs and find their indexes
        sorted_indices = np.argsort(-scores.cpu().detach().numpy(), axis=1)[:self.cfg.top_n]
        # apply sorted indexes to item indexes to get sorted topK item indexes by user
        predicted = np.take_along_axis(scores_idx, sorted_indices, axis=1)
    
        return actual, predicted
