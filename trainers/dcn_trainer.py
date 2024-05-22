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

from models.dcn import DCN
from .base_trainer import BaseTrainer
from metric import *
from loss import BPRLoss

class DCNTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, num_items: int, num_users: int, item2attributes: dict, attributes_count: list) -> None:
        super().__init__(cfg)
        self.num_items = num_items
        self.num_users = num_users
        self.model = DCN(self.cfg, num_users, num_items, attributes_count).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr, self.cfg.weight_decay)
        self.loss = self._loss()
        self.item2attributes = item2attributes
    
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
            
            # update model
            if best_valid_loss > valid_loss:
                logger.info(f"[Trainer] update best model...")
                best_valid_loss = valid_loss
                best_valid_precision_at_k = valid_precision_at_k
                best_recall_k = valid_recall_at_k
                best_valid_ndcg_at_k = valid_ndcg_at_k
                best_valid_map_at_k = valid_map_at_k
                best_epoch = epoch
                endurance = 0

                # TODO: add mlflow

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
            # pos_item_categories, pos_item_statecity, neg_item_categories, neg_item_statecity = \
            #     data['pos_item_categories'].to(self.device), data['pos_item_statecity'].to(self.device), \
            #     data['neg_item_categories'].to(self.device), data['neg_item_statecity'].to(self.device)
            pos_item_categories = torch.tensor([self.item2attributes[item.item()]['categories'] for item in data['pos_item']]).to(self.device)
            pos_item_statecity = torch.tensor([self.item2attributes[item.item()]['statecity'] for item in data['pos_item']]).to(self.device)
            neg_item_categories = torch.tensor([self.item2attributes[item.item()]['categories'] for item in data['neg_item']]).to(self.device)
            neg_item_statecity = torch.tensor([self.item2attributes[item.item()]['statecity'] for item in data['neg_item']]).to(self.device)

            # logger.info(f"pos_categories: {torch.equal(pos_item_categories, torch.tensor([self.item2attributes[item.item()]['categories'] for item in data['pos_item']]).to(self.device))}")
            # logger.info(f"pos_statecity: {torch.equal(pos_item_statecity, torch.tensor([self.item2attributes[item.item()]['statecity'] for item in data['pos_item']]).to(self.device))}")
            # logger.info(f"neg_categories: {torch.equal(neg_item_categories, torch.tensor([self.item2attributes[item.item()]['categories'] for item in data['neg_item']]).to(self.device))}")
            # logger.info(f"neg_statecity: {torch.equal(neg_item_statecity, torch.tensor([self.item2attributes[item.item()]['statecity'] for item in data['neg_item']]).to(self.device))}")
            pos_pred = self.model(user_id, pos_item, pos_item_categories, pos_item_statecity)
            neg_pred = self.model(user_id, neg_item, neg_item_categories, neg_item_statecity)

            self.optimizer.zero_grad()
            loss = self.loss(pos_pred, neg_pred)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def validate(self, valid_dataloader: DataLoader) -> tuple[float]:
        self.model.eval()
        valid_loss = 0
        for data in tqdm(valid_dataloader):
            user_id, pos_item, neg_item = data['user_id'].to(self.device), data['pos_item'].to(self.device), \
                data['neg_item'].to(self.device)
            # pos_item_categories, pos_item_statecity, neg_item_categories, neg_item_statecity = \
            #     data['pos_item_categories'].to(self.device), data['pos_item_statecity'].to(self.device), \
            #     data['neg_item_categories'].to(self.device), data['neg_item_statecity'].to(self.device)
            pos_item_categories = torch.tensor([self.item2attributes[item.item()]['categories'] for item in data['pos_item']]).to(self.device)
            pos_item_statecity = torch.tensor([self.item2attributes[item.item()]['statecity'] for item in data['pos_item']]).to(self.device)
            neg_item_categories = torch.tensor([self.item2attributes[item.item()]['categories'] for item in data['neg_item']]).to(self.device)
            neg_item_statecity = torch.tensor([self.item2attributes[item.item()]['statecity'] for item in data['neg_item']]).to(self.device)

            pos_pred = self.model(user_id, pos_item, pos_item_categories, pos_item_statecity)
            neg_pred = self.model(user_id, neg_item, neg_item_categories, neg_item_statecity)

            loss = self.loss(pos_pred, neg_pred)

            valid_loss += loss.item()

        return valid_loss

    def evaluate(self, eval_data: pd.DataFrame, mode='valid') -> tuple:
        self.model.eval()
        actual, predicted = [], []
        logger.info(f"Before inference #0: {torch.cuda.memory_allocated(self.device)} allocated and {torch.cuda.memory_reserved(self.device)} reserved")
        item_input = torch.tensor([item_id for item_id in range(self.num_items)], dtype=torch.int32).to(self.device)
        # item_categories = torch.tensor([self.item2attributes[item]['categories'] for item in range(self.num_items)], dtype=torch.int32).to(self.device)
        # item_statecity = torch.tensor([self.item2attributes[item]['statecity'] for item in range(self.num_items)], dtype=torch.int32).to(self.device)
        chunk_size = 32 # self.cfg.batch_size
        # logger.info(f"Before inference #1: {torch.cuda.memory_allocated(self.device)} allocated and {torch.cuda.memory_reserved(self.device)} reserved")
        torch.cuda.empty_cache()
        # logger.info(f"Before inference #2: {torch.cuda.memory_allocated(self.device)} allocated and {torch.cuda.memory_reserved(self.device)} reserved")
        for user_id, row in tqdm(eval_data[:10].iterrows(), total=eval_data.shape[0]):
            pred = []
            for idx in range(0, eval_data.shape[0], chunk_size):
                chunk_item_input = item_input[idx:idx+chunk_size]
                chunk_item_categories = torch.tensor([self.item2attributes[item]['categories'] for item in range(idx, min(self.num_items, idx+chunk_size))], dtype=torch.int32).to(self.device)
                chunk_item_statecity = torch.tensor([self.item2attributes[item]['statecity'] for item in range(idx, min(self.num_items, idx+chunk_size))], dtype=torch.int32).to(self.device)
                # print(f"{chunk_size}, {chunk_item_input.size()}, {chunk_item_categories.size()}, {chunk_item_statecity.size()}")
                # logger.info(f"{torch.cuda.memory_allocated(self.device)} allocated and {torch.cuda.memory_reserved(self.device)} reserved")

                chunk_pred: Tensor = self.model(torch.tensor([user_id,]*len(chunk_item_input), dtype=torch.int32).to(self.device), chunk_item_input, chunk_item_categories, chunk_item_statecity)
                pred.extend(chunk_pred.detach().cpu().numpy())

                # torch.cuda.empty_cache()
            # pred = self.model(torch.tensor([user_id,]*self.num_items).to(self.device), item_input, item_categories, item_statecity)
            batch_predicted = \
                self._generate_top_k_recommendation(np.array(pred).reshape(-1), row['mask_items'])
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
    
    def _generate_top_k_recommendation(self, pred: np.ndarray, mask_items) -> tuple[list]:
        # mask to train items
        # pred = pred.cpu().detach().numpy()
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
