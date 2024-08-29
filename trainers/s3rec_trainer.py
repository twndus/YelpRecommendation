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

from models.s3rec import S3Rec
from utils import log_metric
from .base_trainer import BaseTrainer
from metric import *
from loss import BPRLoss

class S3RecPreTrainer:
    def __init__(self, cfg: DictConfig, num_items: int, item2attributes, attributes_count: int) -> None:
        self.cfg = cfg
        self.device = self.cfg.device
        self.model = S3Rec(self.cfg, num_items, attributes_count).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)
        self.loss = self._loss()
        self.item2attribute =  item2attributes
        self.num_items = num_items
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

    def pretrain(self, train_dataset):
        logger.info(f"[Pre-Trainer] run...")

        best_train_loss: float = 1e+6
        endurance: int = 0

        # train
        for epoch in range(self.cfg.pretrain_epochs):
            train_loss: float = self.train(train_dataset)
            logger.info(f'''\n[Pre-Trainer] epoch: {epoch} > pretrain loss: {train_loss:.4f}''') 
            
            if self.cfg.wandb:
                wandb.log({
                    'pretrain_loss': train_loss,
                })

            # update model
            if self._is_surpass_best_metric(
                current=(train_loss,),
                best=(best_train_loss,)):
                
                logger.info(f"[Trainer] update best model...")
                best_train_loss = train_loss
                endurance = 0

                torch.save(self.model.state_dict(), f'{self.cfg.model_dir}/best_pretrain_model.pt')
            else:
                endurance += 1
                if endurance > self.cfg.patience: 
                    logger.info(f"[Trainer] ealry stopping...")
                    break
    
    def item_level_masking(self, sequences):
        masks = torch.rand_like(sequences, dtype=torch.float32) < self.cfg.mask_portion
        item_masked_sequences = masks * sequences
        return masks, item_masked_sequences

    def segment_masking(self, sequences):
        masks, pos_sequences, neg_sequences = torch.zeros_like(sequences), torch.zeros_like(sequences), torch.zeros_like(sequences)
        for i in range(sequences.size(0)):
            # sample segment length randomly
            segment_len = torch.randint(low=2, high=self.cfg.max_seq_len//2, size=(1,))
            # start_index
            start_idx = torch.randint(self.cfg.max_seq_len-segment_len, size=(1,))
            masks[i, start_idx:start_idx+segment_len] = 1
            # pos_sequence
            pos_sequences[i, -segment_len:] = sequences[i, start_idx:start_idx+segment_len]
            # neg_sequence 
            ## other user in same batch
            neg_user_idx = torch.randint(sequences.size(0), size=(1,))
            while neg_user_idx != i:
                neg_user_idx = torch.randint(sequences.size(0), size=(1,))
            ## start_idx
            neg_start_idx = torch.randint(self.cfg.max_seq_len-segment_len, size=(1,))
            neg_sequences[i, -segment_len:] = sequences[neg_user_idx, neg_start_idx:neg_start_idx+segment_len]
        segment_masked_sequences = (1-masks) * sequences
        return segment_masked_sequences, pos_sequences, neg_sequences

    # def train(self, item_datasets, sequence_datasets) -> float:
    def train(self, train_dataloader) -> float:
        self.model.train()
        train_loss = 0
        
        for data in tqdm(train_dataloader): # sequence
            sequences = data['X'].to(self.device)
            aap_actual = data['aap_actual'].to(self.device)
            mip_actual = data['mip_actual'].to(self.device)
            map_actual = data['aap_actual'].to(self.device)

            # item_masked_sequences
            masks, item_masked_sequences = self.item_level_masking(sequences)
            # segment_masked_sequences
            segment_masked_sequences, pos_segments, neg_segments = self.segment_masking(sequences)

            # pretrain
            aap_output, mip_output, map_output, (sp_output_pos, sp_output_neg) = self.model.pretrain(
                item_masked_sequences, segment_masked_sequences, pos_segments, neg_segments)

            # AAP: item + atrributes
            aap_actual = aap_actual * masks.unsqueeze(-1)
            ## compute unmasked area only
            aap_loss = nn.functional.binary_cross_entropy_with_logits(aap_output, aap_actual)
            
            # MIP: sequence + item
            ## compute masked area only
            mip_output = mip_output * masks.logical_not().unsqueeze(-1)
            mip_actual = mip_actual * masks.logical_not().unsqueeze(-1)
            mip_loss = nn.functional.binary_cross_entropy_with_logits(mip_output, mip_actual)

            # MAP: sequence + attribute
            ## compute masked area only
            map_output = map_output * masks.logical_not().unsqueeze(-1)
            map_actual = map_actual * masks.logical_not().unsqueeze(-1)
            map_loss = nn.functional.binary_cross_entropy_with_logits(map_output, map_actual)
            
            # SP: sequence + segment
            ## pos_segment > neg_segment
            sp_output = torch.concat([sp_output_neg, sp_output_pos], dim=0)
            sp_actual = torch.concat([torch.zeros(data['X'].size(0)), torch.ones(data['X'].size(0))]).to(self.device)
            sp_loss = nn.functional.binary_cross_entropy_with_logits(sp_output, sp_actual)

            self.optimizer.zero_grad()
            loss = aap_loss + mip_loss + map_loss + sp_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def load_best_model(self):
        logger.info(f"[Trainer] Load best model...")
        self.model.load_state_dict(torch.load(f'{self.cfg.model_dir}/best_pretrain_model.pt'))

class S3RecTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig, num_items: int, item2attributes, attributes_count: int) -> None:
        super().__init__(cfg)
        self.model = S3Rec(self.cfg, num_items, attributes_count).to(self.device)
        self.optimizer: Optimizer = self._optimizer(self.cfg.optimizer, self.model, self.cfg.lr)
        self.loss = self._loss()
        self._load_best_pretrain_model()

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
            X, pos_items, neg_items = data['X'].to(self.device), data['pos_items'].to(self.device), data['neg_items'].to(self.device)
            pos_preds, neg_preds = self.model.finetune(X, pos_items, neg_items)

            self.optimizer.zero_grad()
            loss = self.loss(pos_preds, neg_preds) # batch*max_length, 1
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss
    
    def validate(self, valid_dataloader: DataLoader) -> tuple[float]:
        self.model.eval()
        valid_loss = 0
        # actual, predicted = [], []
        for data in tqdm(valid_dataloader):
            X, pos_items, neg_items = data['X'].to(self.device), data['pos_items'].to(self.device), data['neg_items'].to(self.device)
            pos_preds, neg_preds = self.model.finetune(X, pos_items, neg_items)

            self.optimizer.zero_grad()
            loss = self.loss(pos_preds, neg_preds)

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
    
    def _load_best_pretrain_model(self):
        pretrain_model_dir = f'{self.cfg.model_dir}/best_pretrain_model.pt'
        if self.cfg.load_pretrain and os.path.exists(pretrain_model_dir):
            logger.info(f"[Trainer] Load best pretrain model...")
            self.model.load_state_dict(torch.load(f'{self.cfg.model_dir}/best_pretrain_model.pt'))

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
