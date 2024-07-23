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

from models.s3rec import S3Rec
from utils import log_metric
from .base_trainer import BaseTrainer
from metric import *
from loss import BPRLoss

class S3RecPreTrainer:
    def __init__(self, cfg: DictConfig, num_items: int, num_users: int, item2attributes, attributes_count: int) -> None:
        self.cfg = cfg
        self.device = self.cfg.device
        self.model = S3Rec(self.cfg, num_items, num_users, attributes_count).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)
        self.loss = self._loss()
        self.item2attribute =  item2attributes
        self.num_items = num_items
        self.num_users = num_users
        self.attributes_count = attributes_count

    def _loss(self): 
        # AAP + MIP + MAP + SP
        return nn.BCEWithLogitsLoss()
    
    def _optimizer(self, optimizer_name: str, model: nn.Module, learning_rate: float, weight_decay: float=0) -> Optimizer:
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.error(f"Optimizer Not Exists: {optimizer_name}")
            raise NotImplementedError(f"Optimizer Not Exists: {optimizer_name}")
    
    def _is_surpass_best_metric(self, **metric) -> bool:
        (valid_loss,
             ) = metric['current']
        
        (best_valid_loss,
            ) = metric['best']
        
        if self.cfg.best_metric == 'loss':
            return valid_loss < best_valid_loss
        else:
            return False

    def pretrain(self, train_dataset, valid_dataset):
        logger.info(f"[Trainer] run...")

        best_valid_loss: float = 1e+6
        best_epoch: int = 0
        endurance: int = 0

        # train
        for epoch in range(self.cfg.pretrain_epochs):
            train_loss: float = self.train(torch.tensor([i for i in range(1, self.num_items+1)], dtype=torch.int32).to(self.device), train_dataset)
            valid_loss = self.validate(torch.tensor([i for i in range(1, self.num_items+1)], dtype=torch.int32).to(self.device), valid_dataset)
            logger.info(f'''\n[Trainer] epoch: {epoch} > train loss: {train_loss:.4f} / 
                        valid loss: {valid_loss:.4f}''') 
            
            if self.cfg.wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                })

            # update model
            if self._is_surpass_best_metric(
                current=(valid_loss,),
                best=(best_valid_loss,)):
                
                logger.info(f"[Trainer] update best model...")
                best_valid_loss = valid_loss
                best_epoch = epoch
                endurance = 0

                torch.save(self.model.state_dict(), f'{self.cfg.model_dir}/best_pretrain_model.pt')
            else:
                endurance += 1
                if endurance > self.cfg.patience: 
                    logger.info(f"[Trainer] ealry stopping...")
                    break

    def train(self, item_datasets, sequence_datasets) -> float:
        self.model.train()
        train_loss = 0
        
        for iter_num in tqdm(range(self.cfg.iter_nums)): # sequence
            item_chunk_size = self.num_items // self.cfg.iter_nums
            items = item_datasets[item_chunk_size * iter_num: item_chunk_size * (iter_num + 1)]

            sequence_chunk_size = self.num_users // self.cfg.iter_nums
            # sequences = sequence_datasets[sequence_chunk_size * iter_num: sequence_chunk_size * (iter_num + 1)]

            # AAP: item + atrributes
            pred = self.model.aap(items) # (item_chunk_size, attributes_count)
            actual = torch.Tensor([[1 if attriute in self.item2attribute[item.item()] else 0 for attriute in range(self.attributes_count)] for item in items]).to(self.device) # (item_chunk_size, attributes_count)
            aap_loss = nn.functional.binary_cross_entropy_with_logits(pred, actual)
            
            # MIP: sequence + item
            # mask
            # def random_mask(sequence):
            #     # mask = torch.Tensor([0] * sequence.size(0))
            #     non_zero_count = torch.nonzero(sequence, as_tuple=True)[0].size(0)
            #     mask_indices = torch.randint(sequence.size(0) - non_zero_count, sequence.size(0), size=1)
            #     # mask[mask_indices] = 1
            #     return mask_indices

            # masks = torch.Tensor([random_mask(sequence) for sequence in sequences]) # ()
            # masked_sequences = sequences * (1 - masks)
            # pred = self.model.mip(masked_sequences, ) # (sequence_chunk_size, mask_count, sequence_len) item idx pred
            # nn.functional.binary_cross_entropy
            # # MAP: sequence + attributes
            # map_loss = self.loss()
            # # SP: sequence + segment
            # sp_loss = self.loss()
            # # X, pos_item, neg_item = data['X'].to(self.device), data['pos_item'].to(self.device), data['neg_item'].to(self.device)
            # # pos_pred, neg_pred = self.model(X, pos_item, neg_item)

            self.optimizer.zero_grad()
            # loss = self.loss(pos_pred, neg_pred)
            loss = aap_loss # + mip_loss + map_loss + sp_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def validate(self, item_datasets, sequence_datasets) -> float:
        self.model.eval()
        valid_loss = 0
        
        for iter_num in tqdm(range(self.cfg.iter_nums)): # sequence
            item_chunk_size = self.num_items // self.cfg.iter_nums
            items = item_datasets[item_chunk_size * iter_num: item_chunk_size * (iter_num + 1)]

            sequence_chunk_size = self.num_users // self.cfg.iter_nums
            # sequences = sequence_datasets[sequence_chunk_size * iter_num: sequence_chunk_size * (iter_num + 1)]

            # AAP: item + atrributes
            pred = self.model.aap(items) # (item_chunk_size, attributes_count)
            actual = torch.Tensor([[1 if attriute in self.item2attribute[item.item()] else 0 for attriute in range(self.attributes_count)] for item in items]).to(self.device) # (item_chunk_size, attributes_count)
            aap_loss = nn.functional.binary_cross_entropy_with_logits(pred, actual)
            
            # MIP: sequence + item
            # mask
            # def random_mask(sequence):
            #     # mask = torch.Tensor([0] * sequence.size(0))
            #     non_zero_count = torch.nonzero(sequence, as_tuple=True)[0].size(0)
            #     mask_indices = torch.randint(sequence.size(0) - non_zero_count, sequence.size(0), size=1)
            #     # mask[mask_indices] = 1
            #     return mask_indices

            # masks = torch.Tensor([random_mask(sequence) for sequence in sequences]) # ()
            # masked_sequences = sequences * (1 - masks)
            # pred = self.model.mip(masked_sequences, ) # (sequence_chunk_size, sequence_len) item idx pred
            # nn.functional.binary_cross_entropy
            # # MAP: sequence + attributes
            # map_loss = self.loss()
            # # SP: sequence + segment
            # sp_loss = self.loss()
            # # X, pos_item, neg_item = data['X'].to(self.device), data['pos_item'].to(self.device), data['neg_item'].to(self.device)
            # # pos_pred, neg_pred = self.model(X, pos_item, neg_item)

            # loss = self.loss(pos_pred, neg_pred)
            loss = aap_loss # + mip_loss + map_loss + sp_loss

            valid_loss += loss.item()

        return valid_loss
    
    def load_best_model(self):
        logger.info(f"[Trainer] Load best model...")
        self.model.load_state_dict(torch.load(f'{self.cfg.model_dir}/best_pretrain_model.pt'))

class S3RecTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, num_items: int, num_users: int, item2attributes, attributes_count: int) -> None:
        super().__init__(cfg)
        self.model = S3Rec(self.cfg, num_items, num_users, attributes_count).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)
        self.loss = self._loss()

    def _loss(self):
        return BPRLoss()
    
    def _is_surpass_best_metric(self, **metric) -> bool:
        (valid_loss,
             ) = metric['current']
        
        (best_valid_loss,
            ) = metric['best']
        
        if self.cfg.best_metric == 'loss':
            return valid_loss < best_valid_loss
        else:
            return False

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
                current=(valid_loss,),
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
        # actual, predicted = [], []
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
            pos_scores, neg_scores = self.model.evaluate(X, pos_item, neg_items)

            batch_actual, batch_predicted = \
                self._generate_target_and_top_k_recommendation(torch.concat([pos_scores, neg_scores], dim=1), pos_item)
            actual.extend(batch_actual)
            predicted.extend(batch_predicted)

        predicted = np.array(predicted)
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
